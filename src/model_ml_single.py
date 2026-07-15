"""
Run a single ML model experiment (split from model_ml_02_comparison.ipynb).

Usage:
  python src/model_ml_single.py --model CatBoost
  python src/model_ml_single.py --model LightGBM --n-households 10 --n-iter 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.project_paths import data_path


LAG_SHORT = [1, 2, 3, 4, 5, 6, 9, 12]
LAG_MID = [18, 24, 36, 48, 72]
LAG_DAILY = [144, 288, 432]
LAG_WEEKLY = [1008]
ROLL_WIN = [6, 12, 36, 144, 1008]


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_ml_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"energy": series})

    for lag in LAG_SHORT + LAG_MID + LAG_DAILY + LAG_WEEKLY:
        df[f"lag_{lag}"] = df["energy"].shift(lag)
        df[f"diff_lag_{lag}"] = df["energy"].diff(lag).shift(1)

    for w in ROLL_WIN:
        df[f"rolling_mean_{w}"] = df["energy"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"] = df["energy"].shift(1).rolling(w).std()
        df[f"rolling_min_{w}"] = df["energy"].shift(1).rolling(w).min()
        df[f"rolling_max_{w}"] = df["energy"].shift(1).rolling(w).max()
    df["ewm_mean_36"] = df["energy"].shift(1).ewm(span=36, adjust=False).mean()
    df["ewm_std_36"] = df["energy"].shift(1).ewm(span=36, adjust=False).std()

    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["minute_slot_sin"] = np.sin(2 * np.pi * (idx.hour * 6 + idx.minute // 10) / 144)
    df["minute_slot_cos"] = np.cos(2 * np.pi * (idx.hour * 6 + idx.minute // 10) / 144)
    return df.dropna()


def split_xy(df: pd.DataFrame, test_ratio: float, target_mode: str):
    n_test = int(len(df) * test_ratio)
    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    feat_cols = [c for c in df.columns if c != "energy"]
    y_train = train["energy"].values
    y_test = test["energy"].values
    if target_mode == "delta":
        y_train = train["energy"].values - train["lag_1"].values
        y_test = test["energy"].values - test["lag_1"].values
    return (
        train[feat_cols].values,
        y_train,
        test[feat_cols].values,
        y_test,
        test["energy"].values,
        test["lag_1"].values,
    )


def get_spaces(seed: int):
    param_spaces = {
        "CatBoost": {
            "iterations": randint(300, 1001),
            "depth": randint(4, 9),
            "learning_rate": uniform(0.01, 0.14),
            "l2_leaf_reg": randint(1, 8),
            "bagging_temperature": uniform(0, 1),
            "random_strength": uniform(0, 2),
        },
        "DecisionTree": {
            "max_depth": [4, 6, 8, 10, 12, None],
            "min_samples_split": randint(2, 21),
            "min_samples_leaf": randint(1, 17),
            "max_features": ["sqrt", "log2", None, 0.5, 0.7],
        },
        "LightGBM": {
            "n_estimators": randint(200, 1001),
            "learning_rate": uniform(0.005, 0.145),
            "num_leaves": randint(20, 128),
            "max_depth": [-1, 5, 7, 9, 11],
            "min_child_samples": randint(10, 101),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": uniform(0, 0.5),
            "reg_lambda": uniform(0, 0.5),
        },
        "RandomForest": {
            "n_estimators": randint(100, 401),
            "max_depth": [5, 8, 12, 16, None],
            "min_samples_split": randint(2, 21),
            "min_samples_leaf": randint(1, 17),
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "XGBoost": {
            "n_estimators": randint(200, 1001),
            "max_depth": randint(3, 9),
            "learning_rate": uniform(0.01, 0.14),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha": uniform(0, 0.5),
            "reg_lambda": uniform(0.5, 1.5),
            "min_child_weight": randint(1, 11),
        },
    }

    base_models = {
        "CatBoost": CatBoostRegressor(
            random_seed=seed, verbose=0, od_type="Iter", od_wait=20
        ),
        "DecisionTree": DecisionTreeRegressor(random_state=seed),
        "LightGBM": LGBMRegressor(random_state=seed, verbose=-1),
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=1),
        "XGBoost": xgb.XGBRegressor(random_state=seed, tree_method="hist", verbosity=0),
    }
    return base_models, param_spaces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ML model experiment")
    parser.add_argument(
        "--model",
        required=True,
        choices=["CatBoost", "DecisionTree", "LightGBM", "RandomForest", "XGBoost"],
    )
    parser.add_argument(
        "--data-path",
        default=str(data_path("KIER_3_Temporal_Resolution", "seasonal_naive", "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv")),
    )
    parser.add_argument("--n-households", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--results-dir", default="results/models")
    parser.add_argument(
        "--target-mode",
        choices=["level", "delta"],
        default="delta",
        help="Train target as absolute level or 1-step delta (default: delta)",
    )
    parser.add_argument(
        "--scoring",
        default="r2",
        choices=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        help="Hyperparameter search scoring metric (default: r2)",
    )
    parser.add_argument(
        "--step",
        choices=["all", "plan", "train", "finalize"],
        default="all",
        help="Execution step: plan/train/finalize (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing detail/params files and retrain from scratch",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index within pending households (for chunk runs)",
    )
    parser.add_argument(
        "--max-households-per-run",
        type=int,
        default=0,
        help="Max households to train in this run (0 means all pending)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data_path, parse_dates=["METER_DATE"], index_col="METER_DATE").sort_index()
    complete_cols = [c for c in raw.columns if raw[c].isna().sum() == 0]
    selected_cols = list(rng.choice(complete_cols, size=min(args.n_households, len(complete_cols)), replace=False))

    base_models, param_spaces = get_spaces(args.seed)
    base_model = base_models[args.model]
    param_space = param_spaces[args.model]

    safe_name = args.model.lower()
    detail_path = results_dir / f"ml_{safe_name}_detail.csv"
    summary_path = results_dir / f"ml_{safe_name}_summary.csv"
    params_path = results_dir / f"ml_{safe_name}_best_params.json"
    plan_path = results_dir / f"ml_{safe_name}_plan.csv"

    existing_df = pd.DataFrame()
    if detail_path.exists() and not args.no_resume:
        existing_df = pd.read_csv(detail_path)
    done_set = set(existing_df["household"].tolist()) if len(existing_df) > 0 else set()

    best_params_by_house = {}
    if params_path.exists() and not args.no_resume:
        with open(params_path, "r", encoding="utf-8") as f:
            best_params_by_house = json.load(f)

    plan_df = pd.DataFrame({"household": selected_cols})
    plan_df["is_done"] = plan_df["household"].isin(done_set)
    plan_df.to_csv(plan_path, index=False)
    done_count = int(plan_df["is_done"].sum())
    pending_count = int((~plan_df["is_done"]).sum())
    print(
        f"[PLAN] {args.model}: total={len(selected_cols)} "
        f"done={done_count} pending={pending_count}"
    )
    print(f"  plan: {plan_path}")
    if args.step == "plan":
        return

    if args.no_resume:
        pending = selected_cols[:]
        merged_df = pd.DataFrame()
    else:
        pending = [h for h in selected_cols if h not in done_set]
        merged_df = existing_df.copy()

    if args.start_index > 0:
        pending = pending[args.start_index :]
    if args.max_households_per_run > 0:
        pending = pending[: args.max_households_per_run]

    tscv = TimeSeriesSplit(n_splits=args.cv_splits)

    if args.step in ("all", "train"):
        if len(pending) == 0:
            print("[TRAIN] no pending household (all skipped).")
        else:
            print(f"[TRAIN] target households: {len(pending)}")

        train_start = time.time()
        completed_times: list[float] = []
        for i, col in enumerate(pending, start=1):
            run_elapsed = time.time() - train_start
            if completed_times:
                avg_one = float(np.mean(completed_times))
                eta_sec = int(avg_one * (len(pending) - (i - 1)))
                print(
                    f"  ({i}/{len(pending)}) {col} | "
                    f"elapsed={run_elapsed/60:.1f}m | eta~{eta_sec/60:.1f}m"
                )
            else:
                print(f"  ({i}/{len(pending)}) {col} | elapsed={run_elapsed/60:.1f}m")

            feat_df = build_ml_features(raw[col])
            X_tr, y_tr, X_te, y_te, y_te_level, y_te_lag1 = split_xy(
                feat_df, args.test_ratio, args.target_mode
            )

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_space,
                n_iter=args.n_iter,
                scoring=args.scoring,
                cv=tscv,
                random_state=args.seed,
                n_jobs=args.n_jobs,
                refit=True,
                error_score="raise",
            )

            t0 = time.time()
            search.fit(X_tr, y_tr)
            elapsed = time.time() - t0
            y_pred = search.best_estimator_.predict(X_te)
            if args.target_mode == "delta":
                y_pred_level = y_pred + y_te_lag1
                metrics = compute_metrics(y_te_level, y_pred_level)
            else:
                metrics = compute_metrics(y_te, y_pred)

            row = {
                "model": args.model,
                "household": col,
                "target_mode": args.target_mode,
                "scoring": args.scoring,
                "elapsed_s": round(elapsed, 1),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                **metrics,
            }
            if len(merged_df) == 0:
                merged_df = pd.DataFrame([row])
            else:
                merged_df = pd.concat([merged_df, pd.DataFrame([row])], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=["household"], keep="last")
            merged_df.to_csv(detail_path, index=False)

            best_params_by_house[col] = search.best_params_
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(best_params_by_house, f, indent=2, ensure_ascii=False)

            # keep API symmetry with notebook (ensures model can be rebuilt)
            clone(base_model).set_params(**search.best_params_)
            completed_times.append(elapsed)
            print(
                f"    done {col}: MAE={row['MAE']:.4f} "
                f"RMSE={row['RMSE']:.4f} R2={row['R2']:.4f} "
                f"time={elapsed/60:.1f}m | checkpoint={detail_path}"
            )

    if args.step == "train":
        return

    if detail_path.exists():
        results_df = pd.read_csv(detail_path)
    else:
        results_df = merged_df
    if len(results_df) == 0:
        raise ValueError("No detail rows found. Run train step first.")

    summary = results_df[["MAE", "RMSE", "MAPE", "R2", "elapsed_s"]].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    results_df.to_csv(detail_path, index=False)
    summary_out = pd.DataFrame(
        [
            {
                "model": args.model,
                "MAE_mean": round(summary.loc["MAE", "mean"], 4),
                "MAE_std": round(summary.loc["MAE", "std"], 4),
                "RMSE_mean": round(summary.loc["RMSE", "mean"], 4),
                "RMSE_std": round(summary.loc["RMSE", "std"], 4),
                "MAPE_mean": round(summary.loc["MAPE", "mean"], 2),
                "MAPE_std": round(summary.loc["MAPE", "std"], 2),
                "R2_mean": round(summary.loc["R2", "mean"], 4),
                "R2_std": round(summary.loc["R2", "std"], 4),
                "elapsed_s_mean": round(summary.loc["elapsed_s", "mean"], 1),
                "elapsed_s_std": round(summary.loc["elapsed_s", "std"], 1),
                "n_households_planned": len(selected_cols),
                "n_households_done": len(results_df),
                "resume_mode": int(not args.no_resume),
            }
        ]
    )
    summary_out.to_csv(summary_path, index=False)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_by_house, f, indent=2, ensure_ascii=False)

    print(f"[DONE] model={args.model}")
    print(f"  detail : {detail_path}")
    print(f"  summary: {summary_path}")
    print(f"  params : {params_path}")


if __name__ == "__main__":
    main()
