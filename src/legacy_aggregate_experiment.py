"""Run legacy-compatible aggregate ML/DL experiments on current KIER outputs.

The previous project modeled aggregate targets such as ELEC_INST_SUM_ALL or
ELEC_INST_SUM_C0 on resampled wide household data. This runner recreates that
experiment shape using the current project's preprocessed and clustered CSVs,
without modifying the previous repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.project_paths import data_path
from src.legacy_clustering import load_cluster_groups
from src.legacy_model_registry import (
    DL_MODEL_NAMES,
    DLTrainConfig,
    ML_MODEL_NAMES,
    build_dl_model,
    build_ml_model,
    compile_dl_model,
    dl_callbacks,
    ensure_tf,
)
import src.legacy_model_registry as model_registry


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sum(y_true - y_pred) / y_true.size)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    try:
        msle = float(mean_squared_log_error(y_true, y_pred))
    except ValueError:
        msle = float("nan")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MSLE": msle,
        "MBE": mean_bias_error(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def data_file(interp_method: str, resolution: str) -> Path:
    return data_path(
        "KIER_3_Temporal_Resolution",
        interp_method,
        f"KIER_USAGE_ELEC_INST_INTERP_{resolution}.csv",
    )


def cluster_dir(interp_method: str, algorithm: str) -> Path:
    return data_path("KIER_4_Clustered", interp_method, algorithm)


def read_wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=["METER_DATE"]).sort_values("METER_DATE")


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt = pd.to_datetime(out["METER_DATE"])
    out["YEAR"] = dt.dt.year
    out["MONTH"] = dt.dt.month
    out["HOUR"] = dt.dt.hour
    out["MINUTE"] = dt.dt.minute
    out["DOW"] = dt.dt.dayofweek
    out["IS_WEEKEND"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def build_aggregate_frame(
    frame: pd.DataFrame,
    target_name: str,
    target_shift: int,
    include_time_features: bool,
) -> pd.DataFrame:
    work = frame.copy()
    household_cols = [c for c in work.columns if c != "METER_DATE"]
    work[target_name] = work[household_cols].sum(axis=1).shift(target_shift)
    if include_time_features:
        work = add_time_features(work)
    work = work.dropna().reset_index(drop=True)
    return work


def split_features(frame: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [c for c in frame.columns if c not in ("METER_DATE", target_col)]
    return frame[feature_cols], frame[target_col], frame["METER_DATE"]


def fit_predict_ml(
    frame: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x, y, dates = split_features(frame, target_col)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=False)
    test_dates = dates.iloc[-len(y_test) :].to_numpy()
    model = build_ml_model(model_name, seed=seed)
    t0 = time.time()
    if model_name == "XGBoost":
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    elif model_name == "LightGBM":
        model.fit(x_train, y_train, eval_metric="mae", eval_set=[(x_train, y_train)])
    elif model_name == "CatBoost":
        model.fit(x_train, y_train, cat_features=[], eval_set=[(x_train, y_train)])
    else:
        model.fit(x_train, y_train)
    elapsed = time.time() - t0
    pred = np.asarray(model.predict(x_test)).reshape(-1)
    return y_test.to_numpy(), pred, test_dates, elapsed


def cross_validate_ml(
    frame: pd.DataFrame,
    target_col: str,
    model_name: str,
    cv_folds: int,
    cv_shuffle: bool,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    x, y, _dates = split_features(frame, target_col)
    kfold = KFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=seed if cv_shuffle else None)
    fold_rows: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(frame), start=1):
        model = build_ml_model(model_name, seed=seed)
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
        t0 = time.time()
        if model_name == "XGBoost":
            model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        elif model_name == "LightGBM":
            model.fit(x_train, y_train, eval_metric="mae", eval_set=[(x_train, y_train)])
        elif model_name == "CatBoost":
            model.fit(x_train, y_train, cat_features=[], eval_set=[(x_train, y_train)])
        else:
            model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test)).reshape(-1)
        row = {"fold": fold_idx, "elapsed_s": round(time.time() - t0, 3), **compute_metrics(y_test.to_numpy(), pred)}
        fold_rows.append(row)
    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "MSLE", "MBE", "R2", "elapsed_s"]
    summary = {f"{c}_mean": float(pd.DataFrame(fold_rows)[c].mean()) for c in metric_cols}
    summary.update({f"{c}_std": float(pd.DataFrame(fold_rows)[c].std()) for c in metric_cols})
    return summary, fold_rows


def make_windows(x: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(0, len(y) - seq_len):
        xs.append(x[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def fit_predict_dl(
    frame: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_ratio: float,
    val_ratio: float,
    seq_len: int,
    train_config: DLTrainConfig,
    seed: int,
    fit_verbose: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    ensure_tf()
    import tensorflow as tf

    tf.random.set_seed(seed)
    x_df, y_s, dates = split_features(frame, target_col)
    x_values = x_df.to_numpy(dtype=float)
    y_values = y_s.to_numpy(dtype=float).reshape(-1, 1)

    n_total = len(frame)
    n_test = int(n_total * test_ratio)
    n_train_val = n_total - n_test
    n_val = int(n_train_val * val_ratio)
    n_train = n_train_val - n_val

    x_scaler = StandardScaler().fit(x_values[:n_train])
    y_scaler = MinMaxScaler().fit(y_values[:n_train])
    x_scaled = x_scaler.transform(x_values)
    y_scaled = y_scaler.transform(y_values).reshape(-1)

    x_tr, y_tr = make_windows(x_scaled[:n_train], y_scaled[:n_train], seq_len)
    x_va, y_va = make_windows(x_scaled[n_train - seq_len : n_train_val], y_scaled[n_train - seq_len : n_train_val], seq_len)
    x_te, y_te = make_windows(x_scaled[n_train_val - seq_len :], y_scaled[n_train_val - seq_len :], seq_len)
    date_te = dates.iloc[n_train_val:].iloc[: len(y_te)].to_numpy()

    model = build_dl_model(model_name, n_features=x_tr.shape[2], seq_len=x_tr.shape[1])
    compile_dl_model(model, train_config)
    t0 = time.time()
    hist = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va),
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        callbacks=dl_callbacks(train_config),
        verbose=fit_verbose,
    )
    elapsed = time.time() - t0
    pred_scaled = model.predict(x_te, verbose=0).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual = y_scaler.inverse_transform(y_te.reshape(-1, 1)).reshape(-1)
    model_registry.keras.backend.clear_session()
    return actual, pred, date_te, elapsed, len(hist.history["loss"])


def write_prediction_csv(path: Path, dates: np.ndarray, actual: np.ndarray, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "METER_DATE": dates,
            "actual": actual,
            "prediction": prediction,
            "residual": actual - prediction,
        }
    ).to_csv(path, index=False)


def run_dataset(args: argparse.Namespace, name: str, frame: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    target_col = f"ELEC_INST_SUM_{name}"
    agg = build_aggregate_frame(frame, target_col, args.target_shift, args.time_features)

    if args.step == "plan":
        return {
            "dataset": name,
            "rows": len(agg),
            "features": len([c for c in agg.columns if c not in ("METER_DATE", target_col)]),
            "target_col": target_col,
        }

    if args.family == "ml" and args.cv_folds > 1:
        cv_summary, fold_rows = cross_validate_ml(
            agg,
            target_col,
            args.model,
            cv_folds=args.cv_folds,
            cv_shuffle=args.cv_shuffle,
            seed=args.seed,
        )
        folds_path = out_dir / f"{safe_name(args.model)}_{name}_folds.csv"
        pd.DataFrame(fold_rows).to_csv(folds_path, index=False)

    if args.family == "ml":
        actual, pred, dates, elapsed = fit_predict_ml(agg, target_col, args.model, args.test_ratio, args.seed)
        n_epochs = None
    else:
        config = DLTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.lr,
            loss=args.dl_loss,
        )
        actual, pred, dates, elapsed, n_epochs = fit_predict_dl(
            agg,
            target_col,
            args.model,
            args.test_ratio,
            args.val_ratio,
            args.seq_len,
            config,
            args.seed,
            args.fit_verbose,
        )
        cv_summary = {}

    pred_path = out_dir / "predictions" / f"{safe_name(args.family)}_{safe_name(args.model)}_{name}.csv"
    write_prediction_csv(pred_path, dates, actual, pred)
    row = {
        "dataset": name,
        "family": args.family,
        "model": args.model,
        "resolution": args.resolution,
        "interp_method": args.interp_method,
        "cluster_algorithm": args.cluster_algorithm,
        "target_shift": args.target_shift,
        "test_ratio": args.test_ratio,
        "cv_folds": args.cv_folds if args.family == "ml" else 0,
        "cv_shuffle": bool(args.cv_shuffle),
        "rows": len(agg),
        "n_features": len([c for c in agg.columns if c not in ("METER_DATE", target_col)]),
        "elapsed_s": round(elapsed, 3),
        "n_epochs": n_epochs,
        "predictions_path": str(pred_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **compute_metrics(actual, pred),
    }
    if args.family == "ml" and args.cv_folds > 1:
        row.update({f"cv_{k}": v for k, v in cv_summary.items()})
    return row


def summarize_cluster_ensemble(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    cluster_rows = [r for r in rows if str(r["dataset"]).startswith("K")]
    if len(cluster_rows) < 2:
        return None
    frames = []
    for row in cluster_rows:
        df = pd.read_csv(row["predictions_path"], parse_dates=["METER_DATE"])
        df = df.rename(
            columns={
                "actual": f"actual_{row['dataset']}",
                "prediction": f"prediction_{row['dataset']}",
            }
        )[["METER_DATE", f"actual_{row['dataset']}", f"prediction_{row['dataset']}"]]
        frames.append(df)
    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on="METER_DATE", how="inner")
    actual_cols = [c for c in merged.columns if c.startswith("actual_")]
    pred_cols = [c for c in merged.columns if c.startswith("prediction_")]
    actual = merged[actual_cols].sum(axis=1).to_numpy()
    pred = merged[pred_cols].sum(axis=1).to_numpy()
    pred_path = out_dir / "predictions" / f"{safe_name(args.family)}_{safe_name(args.model)}_cluster_ensemble.csv"
    write_prediction_csv(pred_path, merged["METER_DATE"].to_numpy(), actual, pred)
    return {
        "dataset": "cluster_ensemble",
        "family": args.family,
        "model": args.model,
        "resolution": args.resolution,
        "interp_method": args.interp_method,
        "cluster_algorithm": args.cluster_algorithm,
        "target_shift": args.target_shift,
        "test_ratio": args.test_ratio,
        "cv_folds": 0,
        "cv_shuffle": bool(args.cv_shuffle),
        "rows": len(merged),
        "n_features": None,
        "elapsed_s": None,
        "n_epochs": None,
        "predictions_path": str(pred_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **compute_metrics(actual, pred),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run legacy-compatible aggregate ML/DL experiment")
    parser.add_argument("--family", choices=["ml", "dl"], default="ml")
    parser.add_argument("--model", required=True, help=f"ML: {ML_MODEL_NAMES}; DL: {DL_MODEL_NAMES}")
    parser.add_argument("--interp-method", default="seasonal_naive")
    parser.add_argument("--resolution", default="1H", choices=["10MIN", "1H", "1D", "1W", "1M"])
    parser.add_argument("--cluster-algorithm", default="kmeans")
    parser.add_argument("--scope", choices=["global", "cluster", "both"], default="both")
    parser.add_argument("--target-shift", type=int, default=1)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-features", action="store_true", default=True)
    parser.add_argument("--no-time-features", dest="time_features", action="store_false")
    parser.add_argument("--results-dir", default="results/legacy_aggregate")
    parser.add_argument("--step", choices=["plan", "run"], default="run")
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dl-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--fit-verbose", type=int, choices=[0, 1, 2], default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.family == "ml" and args.model not in ML_MODEL_NAMES:
        raise ValueError(f"--model must be one of {ML_MODEL_NAMES} for --family ml")
    if args.family == "dl" and args.model not in DL_MODEL_NAMES:
        raise ValueError(f"--model must be one of {DL_MODEL_NAMES} for --family dl")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.scope in ("global", "both"):
        rows.append(run_dataset(args, "ALL", read_wide(data_file(args.interp_method, args.resolution)), out_dir))
    if args.scope in ("cluster", "both"):
        groups = load_cluster_groups(cluster_dir(args.interp_method, args.cluster_algorithm), args.resolution)
        for group_id in sorted(groups):
            path = cluster_dir(args.interp_method, args.cluster_algorithm) / f"KIER_USAGE_ELEC_INST_{group_id}_{args.resolution}.csv"
            rows.append(run_dataset(args, group_id, read_wide(path), out_dir))

    if args.step == "plan":
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    ensemble = summarize_cluster_ensemble(rows, out_dir, args)
    if ensemble is not None:
        rows.append(ensemble)

    detail_path = out_dir / f"{safe_name(args.family)}_{safe_name(args.model)}_{args.resolution}_detail.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    summary_cols = [
        "dataset",
        "family",
        "model",
        "MAE",
        "MAPE",
        "MSE",
        "RMSE",
        "MSLE",
        "MBE",
        "R2",
        "elapsed_s",
        "predictions_path",
    ]
    summary_path = out_dir / f"{safe_name(args.family)}_{safe_name(args.model)}_{args.resolution}_summary.csv"
    pd.DataFrame(rows)[summary_cols].to_csv(summary_path, index=False)
    config_path = out_dir / f"{safe_name(args.family)}_{safe_name(args.model)}_{args.resolution}_config.json"
    config = vars(args).copy()
    config["dl_train_config"] = asdict(DLTrainConfig(args.epochs, args.batch_size, args.patience, args.lr, args.dl_loss))
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] family={args.family} model={args.model} scope={args.scope} resolution={args.resolution}")
    print(f"  detail : {detail_path}")
    print(f"  summary: {summary_path}")
    print(f"  config : {config_path}")


if __name__ == "__main__":
    main()
