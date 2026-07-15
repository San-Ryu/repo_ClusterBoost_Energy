"""
Run one ML model with household/trial-level checkpoints.

This runner keeps the research protocol close to model_ml_single.py but replaces
RandomizedSearchCV with an explicit trial loop. Each hyperparameter trial is
written to disk immediately, so interrupted runs can resume inside a household.
"""

from __future__ import annotations

import argparse
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

from model_ml_single import (
    build_ml_features,
    compute_metrics,
    get_spaces,
    split_xy,
)
from core.project_paths import data_path


MODEL_CHOICES = ["CatBoost", "DecisionTree", "LightGBM", "RandomForest", "XGBoost"]
BASELINE_LAGS = [1, 144, 1008]
FEATURE_MODES = ["single", "all_households", "cluster_households"]


def safe_json_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: safe_json_value(v) for k, v in params.items()}


def select_households(raw: pd.DataFrame, n_households: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    complete_cols = [c for c in raw.columns if raw[c].isna().sum() == 0]
    return list(
        rng.choice(
            complete_cols,
            size=min(n_households, len(complete_cols)),
            replace=False,
        )
    )


def parse_lags(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def load_cluster_map(cluster_dir: Path, resolution: str) -> dict[str, dict[str, Any]]:
    cluster_map: dict[str, dict[str, Any]] = {}
    for path in sorted(cluster_dir.glob(f"*_{resolution}.csv")):
        cluster_id = path.stem.split("_")[-2] if "_" in path.stem else path.stem
        cols = [c for c in pd.read_csv(path, nrows=0).columns if c != "METER_DATE"]
        for household in cols:
            cluster_map[household] = {"cluster_id": cluster_id, "members": cols}
    return cluster_map


def feature_households_for_mode(
    raw: pd.DataFrame,
    household: str,
    feature_mode: str,
    cluster_map: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    complete_cols = [c for c in raw.columns if raw[c].isna().sum() == 0]
    if feature_mode == "single":
        return [household], "single"
    if feature_mode == "all_households":
        return complete_cols, "all"
    if feature_mode == "cluster_households":
        if household not in cluster_map:
            raise ValueError(f"Household not found in cluster files: {household}")
        members = [h for h in cluster_map[household]["members"] if h in complete_cols]
        return members, str(cluster_map[household]["cluster_id"])
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def feature_cache_path(
    cache_dir: Path,
    household: str,
    feature_mode: str,
    peer_lags: list[int],
    cluster_id: str,
) -> Path:
    if feature_mode == "single":
        return cache_dir / f"{household}_features.parquet"
    lag_key = "-".join(str(lag) for lag in peer_lags)
    mode_key = safe_key(f"{feature_mode}_{cluster_id}_peerlags-{lag_key}")
    return cache_dir / mode_key / f"{household}_features.parquet"


def build_feature_df(
    raw: pd.DataFrame,
    household: str,
    feature_households: list[str],
    peer_lags: list[int],
) -> pd.DataFrame:
    feat_df = build_ml_features(raw[household])
    peer_cols = [h for h in feature_households if h != household]
    if len(peer_cols) == 0:
        return feat_df

    peer_parts = []
    for lag in peer_lags:
        peer = raw[peer_cols].shift(lag).reindex(feat_df.index)
        peer.columns = [f"peer__{col}__lag_{lag}" for col in peer.columns]
        peer_parts.append(peer)
    out = pd.concat([feat_df, *peer_parts], axis=1).dropna()
    feature_cols = [c for c in out.columns if c != "energy"]
    out[feature_cols] = out[feature_cols].astype("float32")
    return out


def load_feature_df(
    raw: pd.DataFrame,
    household: str,
    cache_dir: Path,
    use_cache: bool,
    force_cache: bool,
    feature_mode: str,
    feature_households: list[str],
    peer_lags: list[int],
    cluster_id: str,
) -> pd.DataFrame:
    path = feature_cache_path(cache_dir, household, feature_mode, peer_lags, cluster_id)
    if use_cache and path.exists() and not force_cache:
        return pd.read_parquet(path)

    feat_df = build_feature_df(raw, household, feature_households, peer_lags)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(path)
    return feat_df


def score_fold(y_true: np.ndarray, y_pred: np.ndarray, scoring: str) -> float:
    if scoring == "r2":
        return float(r2_score(y_true, y_pred))
    if scoring == "neg_root_mean_squared_error":
        return -float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if scoring == "neg_mean_absolute_error":
        return -float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Unsupported scoring: {scoring}")


def read_trials(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def write_trials(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("trial_index").drop_duplicates("trial_index", keep="last")
    df.to_csv(path, index=False)


def trial_paths(checkpoint_dir: Path, model: str, household: str, feature_mode: str) -> tuple[Path, Path]:
    model_dir = checkpoint_dir / model.lower() / feature_mode
    return (
        model_dir / f"{household}_trials.csv",
        model_dir / f"{household}_best_params.json",
    )


def evaluate_trial(
    base_model,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int,
    scoring: str,
) -> tuple[float, float, list[float]]:
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_scores: list[float] = []
    for train_idx, valid_idx in tscv.split(X_train):
        model = clone(base_model).set_params(**params)
        model.fit(X_train[train_idx], y_train[train_idx])
        pred = model.predict(X_train[valid_idx])
        fold_scores.append(score_fold(y_train[valid_idx], pred, scoring))
    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_scores


def append_or_replace_detail(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.drop_duplicates(subset=["household"], keep="last")
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def add_mase(metrics: dict[str, float], scale: float) -> dict[str, float]:
    out = dict(metrics)
    out["MASE"] = float(metrics["MAE"] / scale) if scale > 0 else np.nan
    return out


def seasonal_scale(train_frame: pd.DataFrame, lag: int = 144) -> float:
    col = f"lag_{lag}"
    if col not in train_frame.columns:
        col = "lag_1"
    return float(np.mean(np.abs(train_frame["energy"].values - train_frame[col].values)))


def compute_scaled_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> dict[str, float]:
    return add_mase(compute_metrics(y_true, y_pred), scale)


def safe_subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    scale: float,
) -> dict[str, float]:
    if int(mask.sum()) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "SMAPE": np.nan, "R2": np.nan, "MASE": np.nan}
    return compute_scaled_metrics(y_true[mask], y_pred[mask], scale)


def baseline_metrics(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    scale: float,
) -> dict[int, dict[str, float]]:
    out = {}
    for lag in BASELINE_LAGS:
        col = f"lag_{lag}"
        if col in test_frame.columns:
            out[lag] = compute_scaled_metrics(y_true, test_frame[col].values, scale)
    return out


def best_baseline_name(metrics_by_lag: dict[int, dict[str, float]]) -> tuple[str, dict[str, float]]:
    lag, metrics = min(metrics_by_lag.items(), key=lambda item: item[1]["RMSE"])
    return f"lag_{lag}", metrics


def peak_diagnostics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_pred: np.ndarray,
    scale: float,
    peak_quantile: float,
) -> dict[str, Any]:
    threshold = float(train_frame["energy"].quantile(peak_quantile))
    y_true = test_frame["energy"].values
    peak_mask = y_true >= threshold
    nonpeak_mask = ~peak_mask
    peak_metrics = safe_subset_metrics(y_true, y_pred, peak_mask, scale)
    nonpeak_metrics = safe_subset_metrics(y_true, y_pred, nonpeak_mask, scale)
    return {
        "peak_threshold": threshold,
        "peak_quantile": float(peak_quantile),
        "peak_count": int(peak_mask.sum()),
        "peak_rate": float(peak_mask.mean()),
        **prefix_metrics("peak", peak_metrics),
        **prefix_metrics("nonpeak", nonpeak_metrics),
    }


def train_test_frames(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = int(len(df) * test_ratio)
    if n_test <= 0 or n_test >= len(df):
        raise ValueError(
            f"Invalid test split: rows={len(df)} test_ratio={test_ratio} n_test={n_test}"
        )
    return df.iloc[:-n_test], df.iloc[-n_test:]


def to_level_prediction(
    y_pred: np.ndarray,
    lag_1: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "delta":
        return y_pred + lag_1
    return y_pred


def write_predictions(
    predictions_dir: Path,
    model: str,
    feature_mode: str,
    household: str,
    test_frame: pd.DataFrame,
    y_pred_level: np.ndarray,
) -> Path:
    model_dir = predictions_dir / model.lower() / feature_mode
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{household}_predictions.csv"
    pred_data = {
        "METER_DATE": test_frame.index,
            "household": household,
            "model": model,
            "feature_mode": feature_mode,
            "actual": test_frame["energy"].values,
        "prediction": y_pred_level,
        "residual": test_frame["energy"].values - y_pred_level,
    }
    for lag in BASELINE_LAGS:
        col = f"lag_{lag}"
        if col in test_frame.columns:
            pred_data[f"naive_lag_{lag}"] = test_frame[col].values
            pred_data[f"naive_lag_{lag}_residual"] = test_frame["energy"].values - test_frame[col].values
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(out_path, index=False)
    return out_path


def diagnose_fit(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_naive_metrics: dict[str, float],
) -> dict[str, Any]:
    r2_gap = train_metrics["R2"] - test_metrics["R2"]
    rmse_improvement = (
        1.0 - (test_metrics["RMSE"] / best_naive_metrics["RMSE"])
        if best_naive_metrics["RMSE"] > 0
        else np.nan
    )
    overfit_flag = bool(r2_gap > 0.25 and test_metrics["R2"] < 0.8)
    weak_vs_naive_flag = bool(np.isfinite(rmse_improvement) and rmse_improvement < 0.02)
    return {
        "r2_gap": float(r2_gap),
        "rmse_improvement_vs_best_naive": float(rmse_improvement),
        "overfit_flag": int(overfit_flag),
        "weak_vs_best_naive_flag": int(weak_vs_naive_flag),
    }


def write_summary(detail_path: Path, summary_path: Path, model: str, planned: int, resume: bool) -> None:
    if not detail_path.exists():
        raise ValueError(f"No detail rows found: {detail_path}")
    results_df = pd.read_csv(detail_path)
    if len(results_df) == 0:
        raise ValueError(f"No detail rows found: {detail_path}")
    metric_cols = [
        "MAE",
        "RMSE",
        "MAPE",
        "SMAPE",
        "R2",
        "MASE",
        "peak_MAE",
        "peak_RMSE",
        "peak_R2",
        "peak_MASE",
        "nonpeak_MAE",
        "nonpeak_RMSE",
        "nonpeak_R2",
        "nonpeak_MASE",
        "rmse_improvement_vs_best_naive",
        "r2_gap",
        "overfit_flag",
        "weak_vs_best_naive_flag",
        "elapsed_s",
    ]
    summary_cols = [col for col in metric_cols if col in results_df.columns]
    summary = results_df[summary_cols].agg(["mean", "std"]).T
    row = {
        "model": model,
        "n_households_planned": planned,
        "n_households_done": len(results_df),
        "resume_mode": int(resume),
    }
    for col in summary_cols:
        row[f"{col}_mean"] = round(float(summary.loc[col, "mean"]), 4)
        row[f"{col}_std"] = round(float(summary.loc[col, "std"]), 4)
    summary_out = pd.DataFrame([row])
    summary_out.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpointed ML model search")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument(
        "--data-path",
        default=str(data_path("KIER_3_Temporal_Resolution", "seasonal_naive", "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv")),
    )
    parser.add_argument("--n-households", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--n-iter", type=int, default=120)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--results-dir", default="results/models")
    parser.add_argument("--checkpoint-dir", default="results/models/checkpoints")
    parser.add_argument("--predictions-dir", default="results/models/predictions")
    parser.add_argument("--feature-cache-dir", default=str(data_path("KIER_5_Features", "ml")))
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="single")
    parser.add_argument("--peer-lags", default="1")
    parser.add_argument("--cluster-dir", default=str(data_path("KIER_4_Clustered", "seasonal_naive", "kmeans")))
    parser.add_argument("--cluster-resolution", default="10MIN")
    parser.add_argument(
        "--target-mode",
        choices=["level", "delta"],
        default="delta",
    )
    parser.add_argument(
        "--scoring",
        default="r2",
        choices=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    )
    parser.add_argument(
        "--step",
        choices=["all", "plan", "train", "finalize"],
        default="all",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-households-per-run", type=int, default=0)
    parser.add_argument("--max-trials-per-run", type=int, default=0)
    parser.add_argument("--peak-quantile", type=float, default=0.95)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--force-feature-cache", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    predictions_dir = Path(args.predictions_dir)
    feature_cache_dir = Path(args.feature_cache_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.data_path,
        parse_dates=["METER_DATE"],
        index_col="METER_DATE",
    ).sort_index()
    selected_cols = select_households(raw, args.n_households, args.seed)
    peer_lags = parse_lags(args.peer_lags)
    cluster_map = (
        load_cluster_map(Path(args.cluster_dir), args.cluster_resolution)
        if args.feature_mode == "cluster_households"
        else {}
    )

    safe_name = args.model.lower()
    run_name = f"{safe_name}_{args.feature_mode}"
    detail_path = results_dir / f"ml_{run_name}_detail.csv"
    summary_path = results_dir / f"ml_{run_name}_summary.csv"
    params_path = results_dir / f"ml_{run_name}_best_params.json"
    plan_path = results_dir / f"ml_{run_name}_checkpoint_plan.csv"

    existing_detail = pd.DataFrame()
    if detail_path.exists() and not args.no_resume:
        existing_detail = pd.read_csv(detail_path)
    done_set = set(existing_detail["household"].tolist()) if len(existing_detail) > 0 else set()

    plan_records = []
    for household in selected_cols:
        trials_path, _ = trial_paths(checkpoint_dir, args.model, household, args.feature_mode)
        trials_df = read_trials(trials_path)
        completed_trials = int(len(trials_df)) if len(trials_df) > 0 else 0
        plan_records.append(
            {
                "household": household,
                "is_done": household in done_set,
                "completed_trials": completed_trials,
                "pending_trials": max(args.n_iter - completed_trials, 0),
            }
        )
    plan_df = pd.DataFrame(plan_records)
    plan_df.to_csv(plan_path, index=False)
    print(
        f"[PLAN] {args.model}: total={len(selected_cols)} "
        f"feature_mode={args.feature_mode} "
        f"done={int(plan_df['is_done'].sum())} "
        f"pending_households={int((~plan_df['is_done']).sum())} "
        f"trial_checkpoint_dir={checkpoint_dir / safe_name / args.feature_mode}"
    )
    print(f"  plan: {plan_path}")
    if args.step == "plan":
        return

    pending = selected_cols[:] if args.no_resume else [h for h in selected_cols if h not in done_set]
    if args.start_index > 0:
        pending = pending[args.start_index :]
    if args.max_households_per_run > 0:
        pending = pending[: args.max_households_per_run]

    if args.step in ("all", "train"):
        base_models, param_spaces = get_spaces(args.seed)
        base_model = base_models[args.model]
        sampled_params = list(
            ParameterSampler(
                param_spaces[args.model],
                n_iter=args.n_iter,
                random_state=args.seed,
            )
        )
        model_best_params = {}
        if params_path.exists() and not args.no_resume:
            with open(params_path, "r", encoding="utf-8") as f:
                model_best_params = json.load(f)

        if len(pending) == 0:
            print("[TRAIN] no pending household (all skipped).")
        else:
            print(f"[TRAIN] target households: {len(pending)}")

        for h_idx, household in enumerate(pending, start=1):
            household_start = time.time()
            print(f"  [HOUSEHOLD] ({h_idx}/{len(pending)}) {household}")
            feature_households, cluster_id = feature_households_for_mode(
                raw=raw,
                household=household,
                feature_mode=args.feature_mode,
                cluster_map=cluster_map,
            )
            feat_df = load_feature_df(
                raw=raw,
                household=household,
                cache_dir=feature_cache_dir,
                use_cache=not args.no_feature_cache,
                force_cache=args.force_feature_cache,
                feature_mode=args.feature_mode,
                feature_households=feature_households,
                peer_lags=peer_lags,
                cluster_id=cluster_id,
            )
            X_tr, y_tr, X_te, y_te, y_te_level, y_te_lag1 = split_xy(
                feat_df,
                args.test_ratio,
                args.target_mode,
            )
            train_frame, test_frame = train_test_frames(feat_df, args.test_ratio)

            trials_path, best_path = trial_paths(checkpoint_dir, args.model, household, args.feature_mode)
            trials_df = pd.DataFrame() if args.no_resume else read_trials(trials_path)
            done_trials = (
                set(trials_df["trial_index"].astype(int).tolist())
                if len(trials_df) > 0 and "trial_index" in trials_df.columns
                else set()
            )
            trial_indices = [i for i in range(args.n_iter) if i not in done_trials]
            if args.max_trials_per_run > 0:
                trial_indices = trial_indices[: args.max_trials_per_run]

            print(
                f"    trials: done={len(done_trials)} "
                f"run_now={len(trial_indices)} total={args.n_iter}"
            )
            for ordinal, trial_index in enumerate(trial_indices, start=1):
                params = safe_params(sampled_params[trial_index])
                t0 = time.time()
                mean_score, std_score, fold_scores = evaluate_trial(
                    base_model=base_model,
                    params=params,
                    X_train=X_tr,
                    y_train=y_tr,
                    cv_splits=args.cv_splits,
                    scoring=args.scoring,
                )
                elapsed = time.time() - t0
                trial_row = {
                    "model": args.model,
                    "household": household,
                    "trial_index": trial_index,
                    "scoring": args.scoring,
                    "mean_test_score": mean_score,
                    "std_test_score": std_score,
                    "fold_scores_json": json.dumps(fold_scores),
                    "params_json": json.dumps(params, ensure_ascii=False),
                    "elapsed_s": round(elapsed, 1),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                trials_df = pd.concat([trials_df, pd.DataFrame([trial_row])], ignore_index=True)
                write_trials(trials_path, trials_df)
                print(
                    f"      trial {trial_index + 1}/{args.n_iter} "
                    f"({ordinal}/{len(trial_indices)}) "
                    f"score={mean_score:.5f} time={elapsed/60:.1f}m "
                    f"checkpoint={trials_path}"
                )

            trials_df = read_trials(trials_path)
            if len(trials_df) < args.n_iter:
                print(
                    f"    partial {household}: "
                    f"{len(trials_df)}/{args.n_iter} trials complete"
                )
                continue

            best_idx = trials_df["mean_test_score"].astype(float).idxmax()
            best_trial = trials_df.loc[best_idx]
            best_params = json.loads(best_trial["params_json"])
            refit_start = time.time()
            best_model = clone(base_model).set_params(**best_params)
            best_model.fit(X_tr, y_tr)
            y_train_pred = best_model.predict(X_tr)
            y_test_pred = best_model.predict(X_te)
            y_train_pred_level = to_level_prediction(
                y_train_pred,
                train_frame["lag_1"].values,
                args.target_mode,
            )
            y_test_pred_level = to_level_prediction(
                y_test_pred,
                y_te_lag1,
                args.target_mode,
            )
            mase_scale = seasonal_scale(train_frame, lag=144)
            metrics = compute_scaled_metrics(y_te_level, y_test_pred_level, mase_scale)
            train_metrics = compute_scaled_metrics(
                train_frame["energy"].values,
                y_train_pred_level,
                mase_scale,
            )
            naive_by_lag = baseline_metrics(test_frame, y_te_level, mase_scale)
            naive_metrics = naive_by_lag[1]
            best_naive, best_naive_metrics = best_baseline_name(naive_by_lag)
            fit_diagnostics = diagnose_fit(train_metrics, metrics, best_naive_metrics)
            peak_metrics = peak_diagnostics(
                train_frame=train_frame,
                test_frame=test_frame,
                y_pred=y_test_pred_level,
                scale=mase_scale,
                peak_quantile=args.peak_quantile,
            )
            predictions_path = write_predictions(
                predictions_dir=predictions_dir,
                model=args.model,
                feature_mode=args.feature_mode,
                household=household,
                test_frame=test_frame,
                y_pred_level=y_test_pred_level,
            )
            refit_elapsed = time.time() - refit_start
            household_elapsed = time.time() - household_start

            detail_row = {
                "model": args.model,
                "household": household,
                "feature_mode": args.feature_mode,
                "cluster_id": cluster_id,
                "n_feature_households": len(feature_households),
                "n_peer_households": max(len(feature_households) - 1, 0),
                "n_input_features": int(X_tr.shape[1]),
                "peer_lags": ",".join(str(lag) for lag in peer_lags),
                "target_mode": args.target_mode,
                "scoring": args.scoring,
                "best_trial_index": int(best_trial["trial_index"]),
                "best_cv_score": float(best_trial["mean_test_score"]),
                "elapsed_s": round(household_elapsed, 1),
                "refit_elapsed_s": round(refit_elapsed, 1),
                "n_trials": len(trials_df),
                "predictions_path": str(predictions_path),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "mase_scale_lag_144": mase_scale,
                "best_naive": best_naive,
                "best_naive_RMSE": best_naive_metrics["RMSE"],
                **metrics,
                **prefix_metrics("train", train_metrics),
                **prefix_metrics("naive", naive_metrics),
                **prefix_metrics("naive_lag_144", naive_by_lag[144]),
                **prefix_metrics("naive_lag_1008", naive_by_lag[1008]),
                **fit_diagnostics,
                **peak_metrics,
            }
            append_or_replace_detail(detail_path, detail_row)
            model_best_params[household] = best_params
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(model_best_params, f, indent=2, ensure_ascii=False)
            with open(best_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "household": household,
                        "best_trial_index": int(best_trial["trial_index"]),
                        "best_cv_score": float(best_trial["mean_test_score"]),
                        "best_params": best_params,
                        "updated_at": detail_row["updated_at"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(
                f"    done {household}: trials={len(trials_df)} "
                f"best_trial={int(best_trial['trial_index']) + 1} "
                f"MAE={detail_row['MAE']:.4f} RMSE={detail_row['RMSE']:.4f} "
                f"R2={detail_row['R2']:.4f} time={household_elapsed/60:.1f}m"
            )

    if args.step == "train":
        return

    write_summary(
        detail_path=detail_path,
        summary_path=summary_path,
        model=args.model,
        planned=len(selected_cols),
        resume=not args.no_resume,
    )
    print(f"[DONE] model={args.model}")
    print(f"  detail : {detail_path}")
    print(f"  summary: {summary_path}")
    print(f"  params : {params_path}")


if __name__ == "__main__":
    main()
