"""Grid-search CV runner for peer-household DL experiments.

This runner fixes the paper-style single-target input contract:

    weather variables + all non-target household usage columns

For the current KIER data this means 17 ASOS weather variables plus 347
household columns when the source power table contains 348 households.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.project_paths import data_path
from src.legacy_model_registry import DL_MODEL_NAMES, DLTrainConfig, build_dl_model, compile_dl_model, dl_callbacks
import src.legacy_model_registry as model_registry


WEATHER_DROP_COLS = {"METER_DATE", "Unnamed: 0", "지점", "지점명"}


@dataclass(frozen=True)
class FeatureContract:
    target_household: str
    n_total_households: int
    n_peer_households: int
    n_weather_features: int
    n_input_features: int
    target_shift: int
    has_target_in_features: bool
    first_timestamp: str
    last_timestamp: str
    n_rows: int


def safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def default_power_path(resolution: str, interp_method: str) -> Path:
    return data_path(
        "KIER_3_Temporal_Resolution",
        interp_method,
        f"KIER_USAGE_ELEC_INST_INTERP_{resolution}.csv",
    )


def default_weather_path(station: str) -> Path:
    if station == "119":
        return data_path("KMA_ASOS", "119_SUWON", "ASOS_119_2010-2024_HR.csv")
    if station == "108":
        return data_path("KMA_ASOS", "108_SEOUL", "ASOS_108_2010-2023_HR.csv")
    raise ValueError(f"Unsupported station: {station}")


def read_power(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["METER_DATE"]).sort_values("METER_DATE")
    frame = frame.set_index("METER_DATE")
    return frame.apply(pd.to_numeric, errors="coerce")


def read_weather(path: Path, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["METER_DATE"]).sort_values("METER_DATE")
    keep_cols = [c for c in frame.columns if c not in WEATHER_DROP_COLS]
    frame = frame[["METER_DATE", *keep_cols]].set_index("METER_DATE")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.reindex(target_index.union(frame.index)).sort_index()
    frame = frame.interpolate(method="time").ffill().bfill()
    return frame.reindex(target_index)


def select_targets(power: pd.DataFrame, n_targets: int, seed: int, explicit_targets: list[str] | None) -> list[str]:
    usable = [c for c in power.columns if power[c].notna().any()]
    if explicit_targets:
        missing = [c for c in explicit_targets if c not in usable]
        if missing:
            raise ValueError(f"Target household columns missing or all-NaN: {missing}")
        return explicit_targets[:n_targets] if n_targets else explicit_targets
    rng = np.random.default_rng(seed)
    n_pick = min(n_targets, len(usable)) if n_targets else len(usable)
    if n_pick <= 0:
        raise ValueError("No usable target household columns found")
    return list(rng.choice(usable, size=n_pick, replace=False))


def build_feature_frame(
    power: pd.DataFrame,
    weather: pd.DataFrame,
    target_household: str,
    target_shift: int,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series, FeatureContract]:
    household_cols = list(power.columns)
    if target_household not in household_cols:
        raise ValueError(f"Unknown target household: {target_household}")

    peer_cols = [c for c in household_cols if c != target_household]
    feature_frame = pd.concat([weather, power[peer_cols]], axis=1)
    if target_shift > 0:
        target = power[target_household].shift(-target_shift)
    elif target_shift < 0:
        target = power[target_household].shift(abs(target_shift))
    else:
        target = power[target_household].copy()

    merged = feature_frame.copy()
    merged["__target__"] = target
    merged = merged.dropna()
    if max_rows > 0:
        merged = merged.tail(max_rows)

    x = merged.drop(columns=["__target__"])
    y = merged["__target__"]
    contract = FeatureContract(
        target_household=target_household,
        n_total_households=len(household_cols),
        n_peer_households=len(peer_cols),
        n_weather_features=weather.shape[1],
        n_input_features=x.shape[1],
        target_shift=target_shift,
        has_target_in_features=target_household in x.columns,
        first_timestamp=str(x.index.min()),
        last_timestamp=str(x.index.max()),
        n_rows=len(x),
    )
    validate_contract(contract)
    return x, y, contract


def validate_contract(contract: FeatureContract) -> None:
    if contract.n_total_households != 348:
        raise ValueError(f"Expected 348 household columns, got {contract.n_total_households}")
    if contract.n_peer_households != 347:
        raise ValueError(f"Expected 347 peer household features, got {contract.n_peer_households}")
    if contract.has_target_in_features:
        raise ValueError(f"Target household leaked into features: {contract.target_household}")
    if contract.n_weather_features <= 0:
        raise ValueError("No weather features were loaded")
    expected = contract.n_peer_households + contract.n_weather_features
    if contract.n_input_features != expected:
        raise ValueError(f"Expected {expected} input features, got {contract.n_input_features}")


def make_windows(x_values: np.ndarray, y_values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for start in range(0, len(y_values) - seq_len):
        end = start + seq_len
        xs.append(x_values[start:end])
        ys.append(y_values[end])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def cv_splitter(kind: str, n_splits: int, seed: int):
    if kind == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    if kind == "kfold":
        return KFold(n_splits=n_splits, shuffle=False)
    if kind == "kfold_shuffle":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raise ValueError(f"Unsupported cv type: {kind}")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def default_grid(preset: str) -> list[dict[str, Any]]:
    if preset == "smoke":
        return list(
            ParameterGrid(
                {
                    "seq_len": [3],
                    "epochs": [1],
                    "batch_size": [256],
                    "patience": [1],
                    "learning_rate": [1e-3],
                    "loss": ["huber"],
                }
            )
        )
    if preset == "small":
        return list(
            ParameterGrid(
                {
                    "seq_len": [3, 6],
                    "epochs": [20],
                    "batch_size": [64, 128],
                    "patience": [5],
                    "learning_rate": [1e-3, 5e-4],
                    "loss": ["huber"],
                }
            )
        )
    if preset == "paper":
        return list(
            ParameterGrid(
                {
                    "seq_len": [3, 6, 12],
                    "epochs": [100, 200],
                    "batch_size": [64, 128],
                    "patience": [20],
                    "learning_rate": [1e-3, 5e-4],
                    "loss": ["huber", "mse"],
                }
            )
        )
    raise ValueError(f"Unsupported grid preset: {preset}")


def parse_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.param_grid_json:
        raw = json.loads(args.param_grid_json)
        if isinstance(raw, dict):
            grid = list(ParameterGrid(raw))
        elif isinstance(raw, list):
            grid = raw
        else:
            raise ValueError("--param-grid-json must be a dict or list")
    else:
        grid = default_grid(args.grid_preset)
    if args.max_param_combos > 0:
        grid = grid[: args.max_param_combos]
    return grid


def fit_one_fold(
    model_name: str,
    x_win: np.ndarray,
    y_win: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    params: dict[str, Any],
    seed: int,
    fit_verbose: int,
) -> tuple[dict[str, float], float, int]:
    model_registry.ensure_tf()
    model_registry.tf.random.set_seed(seed)

    x_train_raw = x_win[train_idx]
    x_valid_raw = x_win[valid_idx]
    y_train_raw = y_win[train_idx].reshape(-1, 1)
    y_valid_raw = y_win[valid_idx].reshape(-1, 1)

    x_scaler = StandardScaler()
    x_train_2d = x_train_raw.reshape(-1, x_train_raw.shape[-1])
    x_scaler.fit(x_train_2d)
    x_train = x_scaler.transform(x_train_2d).reshape(x_train_raw.shape)
    x_valid = x_scaler.transform(x_valid_raw.reshape(-1, x_valid_raw.shape[-1])).reshape(x_valid_raw.shape)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train_raw).reshape(-1)
    y_valid = y_scaler.transform(y_valid_raw).reshape(-1)

    train_config = DLTrainConfig(
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        patience=int(params["patience"]),
        learning_rate=float(params["learning_rate"]),
        loss=str(params.get("loss", "huber")),
    )
    model = build_dl_model(model_name, n_features=x_train.shape[2], seq_len=x_train.shape[1])
    compile_dl_model(model, train_config)
    t0 = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        callbacks=dl_callbacks(train_config),
        verbose=fit_verbose,
    )
    elapsed = time.time() - t0
    pred_scaled = model.predict(x_valid, verbose=0).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual = y_valid_raw.reshape(-1)
    out = metric_row(actual, pred)
    model_registry.keras.backend.clear_session()
    return out, elapsed, len(history.history["loss"])


def run_grid_for_target(
    args: argparse.Namespace,
    x_df: pd.DataFrame,
    y_s: pd.Series,
    target_household: str,
    param_grid: list[dict[str, Any]],
    done_keys: set[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    done_keys = done_keys or set()
    for combo_idx, params in enumerate(param_grid, start=1):
        seq_len = int(params["seq_len"])
        x_win, y_win = make_windows(x_df.to_numpy(dtype=float), y_s.to_numpy(dtype=float), seq_len)
        splitter = cv_splitter(args.cv_type, args.cv_splits, args.seed)
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(x_win), start=1):
            if (target_household, combo_idx, fold_idx) in done_keys:
                print(f"[SKIP] model={args.model} target={target_household} combo={combo_idx} fold={fold_idx}")
                continue
            print(
                f"[CV] model={args.model} target={target_household} "
                f"combo={combo_idx}/{len(param_grid)} fold={fold_idx}/{args.cv_splits} params={params}"
            )
            metrics, elapsed, n_epochs = fit_one_fold(
                args.model,
                x_win,
                y_win,
                train_idx,
                valid_idx,
                params,
                seed=args.seed + combo_idx * 100 + fold_idx,
                fit_verbose=args.fit_verbose,
            )
            rows.append(
                {
                    "model": args.model,
                    "target_household": target_household,
                    "combo_idx": combo_idx,
                    "fold": fold_idx,
                    "cv_type": args.cv_type,
                    "cv_splits": args.cv_splits,
                    "seq_len": seq_len,
                    "epochs_requested": int(params["epochs"]),
                    "batch_size": int(params["batch_size"]),
                    "patience": int(params["patience"]),
                    "learning_rate": float(params["learning_rate"]),
                    "loss": str(params.get("loss", "huber")),
                    "n_epochs": n_epochs,
                    "elapsed_s": round(elapsed, 3),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    **metrics,
                }
            )
    return rows


def summarize(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2", "elapsed_s", "n_epochs"]
    group_cols = [
        "model",
        "target_household",
        "combo_idx",
        "seq_len",
        "epochs_requested",
        "batch_size",
        "patience",
        "learning_rate",
        "loss",
    ]
    grid = detail.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    grid.columns = ["_".join([str(x) for x in col if str(x)]) for col in grid.columns.to_flat_index()]
    best_rows = []
    for (model, target), part in grid.groupby(["model", "target_household"], dropna=False):
        best = part.sort_values(["RMSE_mean", "R2_mean"], ascending=[True, False]).iloc[0].to_dict()
        best["selected_by"] = "min_RMSE_then_max_R2"
        best_rows.append(best)
    return grid, pd.DataFrame(best_rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run peer-household DL grid-search CV")
    parser.add_argument("--model", required=True, choices=list(DL_MODEL_NAMES))
    parser.add_argument("--resolution", default="1H")
    parser.add_argument("--interp-method", default="seasonal_naive")
    parser.add_argument("--data-path", default="")
    parser.add_argument("--weather-path", default="")
    parser.add_argument("--weather-station", choices=["119", "108"], default="119")
    parser.add_argument("--target-households", default="", help="Comma-separated explicit target household columns")
    parser.add_argument("--n-target-households", type=int, default=1)
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-type", choices=["kfold", "kfold_shuffle", "timeseries"], default="kfold")
    parser.add_argument("--grid-preset", choices=["smoke", "small", "paper"], default="small")
    parser.add_argument("--param-grid-json", default="")
    parser.add_argument("--max-param-combos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--results-dir", default="results/dl_peer_grid_cv")
    parser.add_argument("--step", choices=["all", "plan", "train", "finalize"], default="all")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-targets-per-run", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    power_path = Path(args.data_path) if args.data_path else default_power_path(args.resolution, args.interp_method)
    weather_path = Path(args.weather_path) if args.weather_path else default_weather_path(args.weather_station)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    power = read_power(power_path)
    weather = read_weather(weather_path, power.index)
    explicit = [x.strip() for x in args.target_households.split(",") if x.strip()] or None
    targets = select_targets(power, args.n_target_households, args.seed, explicit)
    if args.max_targets_per_run > 0:
        targets = targets[: args.max_targets_per_run]

    model_key = safe_name(args.model)
    plan_path = results_dir / f"dl_peer_{model_key}_plan.csv"
    detail_path = results_dir / f"dl_peer_{model_key}_fold_metrics.csv"
    grid_path = results_dir / f"dl_peer_{model_key}_grid_summary.csv"
    summary_path = results_dir / f"dl_peer_{model_key}_best_summary.csv"
    contract_path = results_dir / f"dl_peer_{model_key}_feature_contract.json"
    config_path = results_dir / f"dl_peer_{model_key}_config.json"

    param_grid = parse_grid(args)
    contracts = []
    plan_rows = []
    for target in targets:
        x_df, y_s, contract = build_feature_frame(power, weather, target, args.target_shift, args.max_rows)
        contracts.append(asdict(contract))
        plan_rows.append(
            {
                **asdict(contract),
                "model": args.model,
                "grid_preset": args.grid_preset,
                "param_combos": len(param_grid),
                "cv_splits": args.cv_splits,
                "cv_type": args.cv_type,
                "power_path": str(power_path),
                "weather_path": str(weather_path),
            }
        )
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)
    write_json(contract_path, contracts)
    write_json(config_path, {"args": vars(args), "param_grid": param_grid})
    print(f"[PLAN] model={args.model} targets={len(targets)} param_combos={len(param_grid)} cv={args.cv_splits}")
    print(f"  plan: {plan_path}")
    print(f"  contract: {contract_path}")
    if args.step == "plan":
        return

    detail_existing = pd.DataFrame()
    if detail_path.exists() and not args.no_resume:
        detail_existing = pd.read_csv(detail_path)
    done_keys = set()
    if len(detail_existing) > 0:
        done_keys = set(
            zip(
                detail_existing["target_household"].astype(str),
                detail_existing["combo_idx"].astype(int),
                detail_existing["fold"].astype(int),
            )
        )

    all_rows = [detail_existing] if len(detail_existing) > 0 else []
    if args.step in ("all", "train"):
        for target in targets:
            x_df, y_s, _contract = build_feature_frame(power, weather, target, args.target_shift, args.max_rows)
            new_rows = run_grid_for_target(args, x_df, y_s, target, param_grid, done_keys=done_keys)
            if new_rows:
                all_rows.append(pd.DataFrame(new_rows))
                pd.concat(all_rows, ignore_index=True).to_csv(detail_path, index=False)
                done_keys.update(
                    (str(row["target_household"]), int(row["combo_idx"]), int(row["fold"])) for row in new_rows
                )
                print(f"[CHECKPOINT] detail: {detail_path}")
            else:
                print(f"[SKIP] target={target}: all fold rows already exist")
    if args.step == "train":
        return

    if detail_path.exists():
        detail = pd.read_csv(detail_path)
    elif all_rows:
        detail = pd.concat(all_rows, ignore_index=True)
    else:
        raise ValueError("No fold metric rows found. Run train first.")
    grid_summary, best_summary = summarize(detail)
    grid_summary.to_csv(grid_path, index=False)
    best_summary.to_csv(summary_path, index=False)
    print(f"[DONE] model={args.model}")
    print(f"  detail: {detail_path}")
    print(f"  grid  : {grid_path}")
    print(f"  best  : {summary_path}")


if __name__ == "__main__":
    main()
