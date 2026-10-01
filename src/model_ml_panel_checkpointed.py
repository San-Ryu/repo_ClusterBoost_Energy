"""
Run panel-based ML experiments for the ClusterBoost protocol.

Protocol:
  - control: train one global model on all selected households, then predict each
    household in the chronological test period.
  - experiment: train one model per cluster, then predict households belonging
    to that cluster in the chronological test period.

Hyperparameter search is checkpointed at the group level, not household level.
This keeps the meaning of the control/experiment aligned while avoiding 348
separate searches for the same global baseline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.project_paths import data_path
from model_ml_single import build_ml_features, get_spaces, mape, smape


MODEL_CHOICES = ["CatBoost", "DecisionTree", "LightGBM", "RandomForest", "XGBoost"]
PANEL_MODES = ["global", "cluster"]
BASELINE_LAGS = [1, 144, 1008]


def safe_json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: safe_json_value(value) for key, value in params.items()}


def safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "SMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def add_mase(metrics: dict[str, float], scale: float) -> dict[str, float]:
    out = dict(metrics)
    out["MASE"] = float(metrics["MAE"] / scale) if scale > 0 else np.nan
    return out


def select_households(raw: pd.DataFrame, n_households: int, seed: int) -> list[str]:
    complete_cols = [col for col in raw.columns if raw[col].isna().sum() == 0]
    if n_households <= 0 or n_households >= len(complete_cols):
        return complete_cols
    rng = np.random.default_rng(seed)
    return list(rng.choice(complete_cols, size=n_households, replace=False))


def read_raw_panel_data(path: str) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns
    dtype_map = {col: "float32" for col in columns if col != "METER_DATE"}
    return pd.read_csv(
        path,
        parse_dates=["METER_DATE"],
        index_col="METER_DATE",
        dtype=dtype_map,
    ).sort_index()


def load_cluster_groups(cluster_dir: Path, resolution: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(cluster_dir.glob(f"*_{resolution}.csv")):
        cluster_id = path.stem.split("_")[-2] if "_" in path.stem else path.stem
        cols = [col for col in pd.read_csv(path, nrows=0).columns if col != "METER_DATE"]
        groups[str(cluster_id)] = cols
    if not groups:
        raise ValueError(f"No cluster files found in {cluster_dir} for resolution={resolution}")
    return groups


def build_groups(
    selected_households: list[str],
    panel_mode: str,
    cluster_dir: Path,
    cluster_resolution: str,
) -> dict[str, list[str]]:
    selected = set(selected_households)
    if panel_mode == "global":
        return {"ALL": selected_households}

    cluster_groups = load_cluster_groups(cluster_dir, cluster_resolution)
    out = {}
    for cluster_id, members in cluster_groups.items():
        group_members = [member for member in members if member in selected]
        if group_members:
            out[cluster_id] = group_members
    if not out:
        raise ValueError("No selected households matched the cluster files.")
    return out


def split_dates(raw: pd.DataFrame, test_ratio: float) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = raw.index.drop_duplicates().sort_values()
    n_test = int(len(dates) * test_ratio)
    if n_test <= 0 or n_test >= len(dates):
        raise ValueError(f"Invalid test split: dates={len(dates)} test_ratio={test_ratio}")
    return dates[:-n_test], dates[-n_test:]


def build_panel_frames(
    raw: pd.DataFrame,
    households: list[str],
    household_codes: dict[str, int],
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frames = []
    test_frames = []
    wanted_dates = train_dates.union(test_dates)
    train_cutoff = len(train_dates)
    date_codes = pd.Series(np.arange(len(wanted_dates), dtype=np.int32), index=wanted_dates)
    for household in households:
        feat = build_ml_features(raw[household]).loc[lambda df: df.index.isin(wanted_dates)]
        feat = feat.reset_index(names="METER_DATE")
        feat.insert(1, "date_code", date_codes.reindex(feat["METER_DATE"]).to_numpy(dtype=np.int32))
        feat["household_code"] = np.int16(household_codes[household])
        float_cols = [col for col in feat.columns if col not in ("METER_DATE", "date_code", "household_code")]
        feat[float_cols] = feat[float_cols].astype("float32", copy=False)
        feat["date_code"] = feat["date_code"].astype("int32", copy=False)
        feat["household_code"] = feat["household_code"].astype("int16", copy=False)
        train_frames.append(feat[feat["date_code"] < train_cutoff].reset_index(drop=True))
        test_frames.append(feat[feat["date_code"] >= train_cutoff].reset_index(drop=True))
        del feat
    train_panel = pd.concat(train_frames, ignore_index=True, copy=False)
    test_panel = pd.concat(test_frames, ignore_index=True, copy=False)
    del train_frames, test_frames
    gc.collect()
    return train_panel, test_panel


def panel_xy(frame: pd.DataFrame, target_mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = [col for col in frame.columns if col not in ("METER_DATE", "date_code", "household", "energy")]
    y = frame["energy"].values
    if target_mode == "delta":
        y = frame["energy"].values - frame["lag_1"].values
    return frame[feature_cols], y


def to_level_prediction(pred: np.ndarray, frame: pd.DataFrame, target_mode: str) -> np.ndarray:
    if target_mode == "delta":
        return pred + frame["lag_1"].values
    return pred


def score_fold(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "r2":
        return float(r2_score(y_true, y_pred))
    if scoring == "neg_root_mean_squared_error":
        return -float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if scoring == "neg_mean_absolute_error":
        return -float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Unsupported scoring: {scoring}")


def evaluate_trial(
    base_model,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    cv_splits: int,
    scoring: str,
    target_mode: str,
) -> tuple[float, float, list[float]]:
    dates = np.array(sorted(train_frame["date_code"].drop_duplicates()))
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_scores = []
    for train_date_idx, valid_date_idx in tscv.split(dates):
        train_codes = dates[train_date_idx]
        valid_codes = dates[valid_date_idx]
        fold_train = train_frame[train_frame["date_code"].isin(train_codes)]
        fold_valid = train_frame[train_frame["date_code"].isin(valid_codes)]
        X_train, y_train = panel_xy(fold_train, target_mode)
        X_valid, y_valid = panel_xy(fold_valid, target_mode)
        model = clone(base_model).set_params(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        fold_scores.append(score_fold(y_valid, pred, scoring))
        del model, pred, X_train, y_train, X_valid, y_valid, fold_train, fold_valid
        gc.collect()
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores


def read_trials(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_trials(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("trial_index").drop_duplicates("trial_index", keep="last")
    df.to_csv(path, index=False)


def group_paths(checkpoint_dir: Path, model: str, panel_mode: str, group_id: str) -> tuple[Path, Path]:
    group_key = safe_key(group_id)
    model_dir = checkpoint_dir / model.lower() / panel_mode
    return (
        model_dir / f"{group_key}_trials.csv",
        model_dir / f"{group_key}_best_params.json",
    )


def seasonal_scale(frame: pd.DataFrame, lag: int = 144) -> float:
    col = f"lag_{lag}" if f"lag_{lag}" in frame.columns else "lag_1"
    return float(np.mean(np.abs(frame["energy"].values - frame[col].values)))


def baseline_metrics(frame: pd.DataFrame, scale: float) -> dict[int, dict[str, float]]:
    out = {}
    y_true = frame["energy"].values
    for lag in BASELINE_LAGS:
        col = f"lag_{lag}"
        if col in frame.columns:
            out[lag] = add_mase(compute_metrics(y_true, frame[col].values), scale)
    return out


def best_baseline(metrics_by_lag: dict[int, dict[str, float]]) -> tuple[str, dict[str, float]]:
    lag, metrics = min(metrics_by_lag.items(), key=lambda item: item[1]["RMSE"])
    return f"lag_{lag}", metrics


def write_predictions(
    predictions_dir: Path,
    model: str,
    panel_mode: str,
    group_id: str,
    household: str,
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> Path:
    out_dir = predictions_dir / model.lower() / panel_mode / safe_key(group_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{household}_predictions.csv"
    out = {
        "METER_DATE": frame["METER_DATE"].values,
        "household": household,
        "model": model,
        "panel_mode": panel_mode,
        "group_id": group_id,
        "actual": frame["energy"].values,
        "prediction": prediction,
        "residual": frame["energy"].values - prediction,
    }
    for lag in BASELINE_LAGS:
        col = f"lag_{lag}"
        if col in frame.columns:
            out[f"naive_lag_{lag}"] = frame[col].values
    pd.DataFrame(out).to_csv(out_path, index=False)
    return out_path


def append_or_replace(path: Path, row: dict[str, Any], subset: list[str]) -> None:
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.drop_duplicates(subset=subset, keep="last")
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def summarize(detail_path: Path, summary_path: Path, model: str, panel_mode: str, planned: int) -> None:
    if not detail_path.exists():
        raise ValueError(f"No detail rows found: {detail_path}")
    detail = pd.read_csv(detail_path)
    if len(detail) == 0:
        raise ValueError(f"No detail rows found: {detail_path}")
    metric_cols = [
        "MAE",
        "MAPE",
        "MSE",
        "RMSE",
        "SMAPE",
        "R2",
        "MASE",
        "best_naive_MAE",
        "best_naive_MSE",
        "best_naive_RMSE",
        "rmse_improvement_vs_best_naive",
        "elapsed_s",
    ]
    summary_cols = [col for col in metric_cols if col in detail.columns]
    grouped = detail.groupby("group_id", dropna=False)[summary_cols].agg(["mean", "std"])
    rows = []
    for group_id, values in grouped.iterrows():
        row = {
            "model": model,
            "panel_mode": panel_mode,
            "group_id": group_id,
            "n_households_planned": planned,
            "n_households_done": int((detail["group_id"] == group_id).sum()),
        }
        for col in summary_cols:
            row[f"{col}_mean"] = round(float(values[(col, "mean")]), 4)
            row[f"{col}_std"] = round(float(values[(col, "std")]), 4)
        rows.append(row)
    all_row = {
        "model": model,
        "panel_mode": panel_mode,
        "group_id": "ALL_GROUPS",
        "n_households_planned": planned,
        "n_households_done": len(detail),
    }
    overall = detail[summary_cols].agg(["mean", "std"]).T
    for col in summary_cols:
        all_row[f"{col}_mean"] = round(float(overall.loc[col, "mean"]), 4)
        all_row[f"{col}_std"] = round(float(overall.loc[col, "std"]), 4)
    rows.append(all_row)
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpointed panel ML search")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--panel-mode", choices=PANEL_MODES, default="global")
    parser.add_argument(
        "--data-path",
        default=str(data_path("KIER_3_Temporal_Resolution", "seasonal_naive", "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv")),
    )
    parser.add_argument("--n-households", type=int, default=348)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--results-dir", default="results/stage6_panel")
    parser.add_argument("--checkpoint-dir", default="results/stage6_panel/checkpoints")
    parser.add_argument("--predictions-dir", default="results/stage6_panel/predictions")
    parser.add_argument("--cluster-dir", default=str(data_path("KIER_4_Clustered", "seasonal_naive", "kmeans")))
    parser.add_argument("--cluster-resolution", default="10MIN")
    parser.add_argument("--target-mode", choices=["level", "delta"], default="level")
    parser.add_argument(
        "--scoring",
        default="r2",
        choices=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    )
    parser.add_argument("--step", choices=["all", "plan", "train", "finalize"], default="all")
    parser.add_argument("--start-group-index", type=int, default=0)
    parser.add_argument("--max-groups-per-run", type=int, default=0)
    parser.add_argument("--max-trials-per-run", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    predictions_dir = Path(args.predictions_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    raw = read_raw_panel_data(args.data_path)
    selected_households = select_households(raw, args.n_households, args.seed)
    household_codes = {household: idx for idx, household in enumerate(selected_households)}
    train_dates, test_dates = split_dates(raw, args.test_ratio)
    groups = build_groups(
        selected_households=selected_households,
        panel_mode=args.panel_mode,
        cluster_dir=Path(args.cluster_dir),
        cluster_resolution=args.cluster_resolution,
    )

    safe_name = args.model.lower()
    run_name = f"{safe_name}_{args.panel_mode}"
    detail_path = results_dir / f"ml_{run_name}_detail.csv"
    summary_path = results_dir / f"ml_{run_name}_summary.csv"
    params_path = results_dir / f"ml_{run_name}_best_params.json"
    plan_path = results_dir / f"ml_{run_name}_checkpoint_plan.csv"

    existing_detail = pd.DataFrame()
    if detail_path.exists() and not args.no_resume:
        existing_detail = pd.read_csv(detail_path)
    done_groups = set()
    if len(existing_detail) > 0:
        done_counts = existing_detail.groupby("group_id")["household"].nunique().to_dict()
        for group_id, members in groups.items():
            if done_counts.get(group_id, 0) >= len(members):
                done_groups.add(group_id)

    plan_rows = []
    for group_id, members in groups.items():
        trials_path, _ = group_paths(checkpoint_dir, args.model, args.panel_mode, group_id)
        trials = read_trials(trials_path)
        plan_rows.append(
            {
                "group_id": group_id,
                "n_households": len(members),
                "is_done": group_id in done_groups,
                "completed_trials": int(len(trials)),
                "pending_trials": max(args.n_iter - int(len(trials)), 0),
            }
        )
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)
    print(
        f"[PLAN] {args.model}: panel_mode={args.panel_mode} "
        f"groups={len(groups)} households={len(selected_households)} "
        f"done_groups={len(done_groups)}"
    )
    print(f"  plan: {plan_path}")
    if args.step == "plan":
        return

    pending_group_ids = [group_id for group_id in groups if args.no_resume or group_id not in done_groups]
    if args.start_group_index > 0:
        pending_group_ids = pending_group_ids[args.start_group_index :]
    if args.max_groups_per_run > 0:
        pending_group_ids = pending_group_ids[: args.max_groups_per_run]

    base_models, param_spaces = get_spaces(args.seed)
    base_model = base_models[args.model]
    sampled_params = list(
        ParameterSampler(param_spaces[args.model], n_iter=args.n_iter, random_state=args.seed)
    )
    best_params_by_group = {}
    if params_path.exists() and not args.no_resume:
        with open(params_path, "r", encoding="utf-8") as f:
            best_params_by_group = json.load(f)

    if args.step in ("all", "train"):
        print(f"[TRAIN] pending_groups={len(pending_group_ids)}")
        for ordinal, group_id in enumerate(pending_group_ids, start=1):
            group_start = time.time()
            members = groups[group_id]
            print(f"  [GROUP] ({ordinal}/{len(pending_group_ids)}) {group_id} households={len(members)}")
            train_frame, test_frame = build_panel_frames(
                raw=raw,
                households=members,
                household_codes=household_codes,
                train_dates=train_dates,
                test_dates=test_dates,
            )
            print(
                "    panel: "
                f"train_rows={len(train_frame)} "
                f"test_rows={len(test_frame)} "
                f"memory={(train_frame.memory_usage(deep=True).sum() + test_frame.memory_usage(deep=True).sum()) / 1024**3:.2f}GB"
            )
            trials_path, best_path = group_paths(checkpoint_dir, args.model, args.panel_mode, group_id)
            trials_df = pd.DataFrame() if args.no_resume else read_trials(trials_path)
            done_trials = (
                set(trials_df["trial_index"].astype(int).tolist())
                if len(trials_df) > 0 and "trial_index" in trials_df.columns
                else set()
            )
            trial_indices = [idx for idx in range(args.n_iter) if idx not in done_trials]
            if args.max_trials_per_run > 0:
                trial_indices = trial_indices[: args.max_trials_per_run]
            print(f"    trials: done={len(done_trials)} run_now={len(trial_indices)} total={args.n_iter}")

            for trial_ordinal, trial_index in enumerate(trial_indices, start=1):
                params = safe_params(sampled_params[trial_index])
                t0 = time.time()
                mean_score, std_score, fold_scores = evaluate_trial(
                    base_model=base_model,
                    params=params,
                    train_frame=train_frame,
                    cv_splits=args.cv_splits,
                    scoring=args.scoring,
                    target_mode=args.target_mode,
                )
                elapsed = time.time() - t0
                row = {
                    "model": args.model,
                    "panel_mode": args.panel_mode,
                    "group_id": group_id,
                    "trial_index": trial_index,
                    "scoring": args.scoring,
                    "mean_test_score": mean_score,
                    "std_test_score": std_score,
                    "fold_scores_json": json.dumps(fold_scores),
                    "params_json": json.dumps(params, ensure_ascii=False),
                    "elapsed_s": round(elapsed, 1),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                trials_df = pd.concat([trials_df, pd.DataFrame([row])], ignore_index=True)
                write_trials(trials_path, trials_df)
                gc.collect()
                print(
                    f"      trial {trial_index + 1}/{args.n_iter} "
                    f"({trial_ordinal}/{len(trial_indices)}) "
                    f"score={mean_score:.5f} time={elapsed/60:.1f}m"
                )

            trials_df = read_trials(trials_path)
            if len(trials_df) < args.n_iter:
                print(f"    partial {group_id}: {len(trials_df)}/{args.n_iter} trials complete")
                continue

            best_idx = trials_df["mean_test_score"].astype(float).idxmax()
            best_trial = trials_df.loc[best_idx]
            best_params = json.loads(best_trial["params_json"])
            X_train, y_train = panel_xy(train_frame, args.target_mode)
            best_model = clone(base_model).set_params(**best_params)
            fit_start = time.time()
            best_model.fit(X_train, y_train)
            fit_elapsed = time.time() - fit_start
            X_test, _ = panel_xy(test_frame, args.target_mode)
            test_pred = to_level_prediction(best_model.predict(X_test), test_frame, args.target_mode)

            group_scale = seasonal_scale(train_frame, lag=144)
            for household in members:
                mask = test_frame["household_code"].to_numpy() == household_codes[household]
                household_test = test_frame.loc[mask].copy()
                household_pred = test_pred[mask]
                metrics = add_mase(compute_metrics(household_test["energy"].values, household_pred), group_scale)
                naive_by_lag = baseline_metrics(household_test, group_scale)
                naive_name, naive_metrics = best_baseline(naive_by_lag)
                improvement = (
                    1.0 - metrics["RMSE"] / naive_metrics["RMSE"]
                    if naive_metrics["RMSE"] > 0
                    else np.nan
                )
                pred_path = write_predictions(
                    predictions_dir=predictions_dir,
                    model=args.model,
                    panel_mode=args.panel_mode,
                    group_id=group_id,
                    household=household,
                    frame=household_test,
                    prediction=household_pred,
                )
                detail_row = {
                    "model": args.model,
                    "panel_mode": args.panel_mode,
                    "group_id": group_id,
                    "household": household,
                    "n_group_households": len(members),
                    "n_train_rows": len(train_frame),
                    "n_test_rows": len(household_test),
                    "n_input_features": int(X_train.shape[1]),
                    "target_mode": args.target_mode,
                    "scoring": args.scoring,
                    "best_trial_index": int(best_trial["trial_index"]),
                    "best_cv_score": float(best_trial["mean_test_score"]),
                    "n_trials": len(trials_df),
                    "elapsed_s": round(time.time() - group_start, 1),
                    "fit_elapsed_s": round(fit_elapsed, 1),
                    "predictions_path": str(pred_path),
                    "best_naive": naive_name,
                    "rmse_improvement_vs_best_naive": float(improvement),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    **metrics,
                    **prefix_metrics("best_naive", naive_metrics),
                }
                append_or_replace(detail_path, detail_row, subset=["group_id", "household"])

            best_params_by_group[group_id] = best_params
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(best_params_by_group, f, indent=2, ensure_ascii=False)
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": args.model,
                        "panel_mode": args.panel_mode,
                        "group_id": group_id,
                        "best_trial_index": int(best_trial["trial_index"]),
                        "best_cv_score": float(best_trial["mean_test_score"]),
                        "best_params": best_params,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(
                f"    done {group_id}: households={len(members)} "
                f"best_trial={int(best_trial['trial_index']) + 1} "
                f"time={(time.time() - group_start)/60:.1f}m"
            )

    if args.step == "train":
        return

    summarize(
        detail_path=detail_path,
        summary_path=summary_path,
        model=args.model,
        panel_mode=args.panel_mode,
        planned=len(selected_households),
    )
    print(f"[DONE] model={args.model} panel_mode={args.panel_mode}")
    print(f"  detail : {detail_path}")
    print(f"  summary: {summary_path}")
    print(f"  params : {params_path}")


if __name__ == "__main__":
    main()
