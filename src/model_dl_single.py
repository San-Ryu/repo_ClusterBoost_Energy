"""
Run a single DL model experiment (split from model_dl_02_comparison.ipynb).

Usage:
  python src/model_dl_single.py --model TCN
  python src/model_dl_single.py --model GRU --n-households 10 --epochs 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.project_paths import data_path
from src.legacy_model_registry import DL_MODEL_NAMES, build_dl_model

tf = None
keras = None
EarlyStopping = None
ReduceLROnPlateau = None


def ensure_tf():
    global tf, keras, EarlyStopping, ReduceLROnPlateau
    if tf is not None:
        return
    import tensorflow as _tf
    from tensorflow import keras as _keras
    from keras.callbacks import EarlyStopping as _EarlyStopping, ReduceLROnPlateau as _ReduceLROnPlateau

    tf = _tf
    keras = _keras
    EarlyStopping = _EarlyStopping
    ReduceLROnPlateau = _ReduceLROnPlateau


def make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    t = pd.DataFrame(index=index)
    t["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    t["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    t["dow_sin"] = np.sin(2 * np.pi * index.dayofweek / 7)
    t["dow_cos"] = np.cos(2 * np.pi * index.dayofweek / 7)
    t["month_sin"] = np.sin(2 * np.pi * index.month / 12)
    t["month_cos"] = np.cos(2 * np.pi * index.month / 12)
    minute_slot = index.hour * 6 + (index.minute // 10)
    t["minute_slot_sin"] = np.sin(2 * np.pi * minute_slot / 144)
    t["minute_slot_cos"] = np.cos(2 * np.pi * minute_slot / 144)
    t["is_weekend"] = (index.dayofweek >= 5).astype(int)
    return t


def build_dataset_for_household(
    series: pd.Series,
    time_feats: pd.DataFrame,
    seq_len: int,
    train_r: float,
    val_r: float,
    target_mode: str,
):
    T = len(series)
    n_train = int(T * train_r)
    n_val = int(T * val_r)

    scaler = MinMaxScaler()
    energy = series.values.reshape(-1, 1)
    scaler.fit(energy[:n_train])
    energy_scaled = scaler.transform(energy)

    feat = np.concatenate([energy_scaled, time_feats.values], axis=1)
    target = energy_scaled.ravel()
    delta_target = target - np.roll(target, 1)
    delta_target[0] = 0.0

    def make_windows(feat_arr, target_arr, start, end):
        X, y, prev_levels = [], [], []
        for i in range(start, end - seq_len):
            X.append(feat_arr[i : i + seq_len])
            y_idx = i + seq_len
            if target_mode == "delta":
                y.append(delta_target[y_idx])
            else:
                y.append(target_arr[y_idx])
            prev_levels.append(target_arr[y_idx - 1])
        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(prev_levels, dtype=np.float32),
        )

    X_tr, y_tr, prev_tr = make_windows(feat, target, 0, n_train)
    X_va, y_va, prev_va = make_windows(feat, target, n_train, n_train + n_val)
    X_te, y_te, prev_te = make_windows(feat, target, n_train + n_val, T)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, prev_tr, prev_va, prev_te, scaler


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(y_true, y_pred, scaler):
    yt = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    yp = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    mse = float(mean_squared_error(yt, yp))
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": mape(yt, yp),
        "R2": float(r2_score(yt, yp)),
    }


def train_model(
    model,
    X_tr,
    y_tr,
    X_va,
    y_va,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float,
    fit_verbose: int = 2,
):
    ensure_tf()
    model.compile(
        loss=keras.losses.Huber(),
        optimizer=keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=1.0),
        metrics=["mae"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]
    t0 = time.time()
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=fit_verbose,
    )
    elapsed = time.time() - t0
    return hist, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DL model experiment")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(DL_MODEL_NAMES),
    )
    parser.add_argument(
        "--data-path",
        default=str(data_path("KIER_3_Temporal_Resolution", "seasonal_naive", "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv")),
    )
    parser.add_argument("--n-households", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=144)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--target-mode",
        choices=["level", "delta"],
        default="delta",
        help="Train target as absolute level or 1-step delta (default: delta)",
    )
    parser.add_argument(
        "--fit-verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Keras fit verbosity (0/1/2). Default: 2",
    )
    parser.add_argument("--results-dir", default="results/models")
    parser.add_argument(
        "--step",
        choices=["all", "plan", "train", "finalize"],
        default="all",
        help="Execution step: plan/train/finalize (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing detail file and retrain from scratch",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index within pending households (for chunk run)",
    )
    parser.add_argument(
        "--max-households-per-run",
        type=int,
        default=0,
        help="Max households to train in this run (0 means all pending)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data_path, parse_dates=["METER_DATE"], index_col="METER_DATE").sort_index()
    time_feats = make_time_features(raw.index)

    complete_cols = [c for c in raw.columns if raw[c].isna().sum() == 0]
    rng = np.random.default_rng(args.seed)
    selected_cols = list(rng.choice(complete_cols, size=min(args.n_households, len(complete_cols)), replace=False))

    model_builders = {name: lambda nf, sl, model_name=name: build_dl_model(model_name, nf, sl) for name in DL_MODEL_NAMES}

    probe = raw[selected_cols[0]]
    X_tr, y_tr, X_va, y_va, X_te, y_te, prev_tr, prev_va, prev_te, scaler = build_dataset_for_household(
        probe, time_feats, args.seq_len, args.train_ratio, args.val_ratio, args.target_mode
    )
    n_features = X_tr.shape[2]
    seq_len = X_tr.shape[1]

    safe_name = args.model.lower()
    detail_path = results_dir / f"dl_{safe_name}_detail.csv"
    summary_path = results_dir / f"dl_{safe_name}_summary.csv"
    plan_path = results_dir / f"dl_{safe_name}_plan.csv"

    existing_df = pd.DataFrame()
    if detail_path.exists() and not args.no_resume:
        existing_df = pd.read_csv(detail_path)
    done_set = set(existing_df["household"].tolist()) if len(existing_df) > 0 else set()

    # Step 1) plan
    plan_df = pd.DataFrame({"household": selected_cols})
    plan_df["is_done"] = plan_df["household"].isin(done_set)
    plan_df.to_csv(plan_path, index=False)
    print(f"[PLAN] {args.model}: total={len(selected_cols)} done={plan_df['is_done'].sum()} pending={(~plan_df['is_done']).sum()}")
    print(f"  plan: {plan_path}")
    if args.step == "plan":
        return

    # Step 2) train (skip existing)
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

    if args.step in ("all", "train"):
        ensure_tf()
        tf.random.set_seed(args.seed)
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
            series = raw[col]
            X_tr, y_tr, X_va, y_va, X_te, y_te, prev_tr, prev_va, prev_te, scaler = build_dataset_for_household(
                series, time_feats, args.seq_len, args.train_ratio, args.val_ratio, args.target_mode
            )
            model = model_builders[args.model](n_features, seq_len)
            hist, elapsed = train_model(
                model,
                X_tr,
                y_tr,
                X_va,
                y_va,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                lr=args.lr,
                fit_verbose=args.fit_verbose,
            )
            y_pred = model.predict(X_te, verbose=0).ravel()
            if args.target_mode == "delta":
                y_pred_level = y_pred + prev_te
                y_true_level = y_te + prev_te
                metrics = compute_metrics(y_true_level, y_pred_level, scaler)
            else:
                metrics = compute_metrics(y_te, y_pred, scaler)
            row = {
                "model": args.model,
                "household": col,
                "target_mode": args.target_mode,
                "n_epochs": len(hist.history["loss"]),
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
            completed_times.append(elapsed)
            keras.backend.clear_session()
            print(
                f"    done {col}: epochs={row['n_epochs']} "
                f"MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f} "
                f"time={elapsed/60:.1f}m | checkpoint={detail_path}"
            )

    if args.step == "train":
        return

    # Step 3) finalize
    if detail_path.exists():
        results_df = pd.read_csv(detail_path)
    else:
        results_df = merged_df
    if len(results_df) == 0:
        raise ValueError("No detail rows found. Run train step first.")

    summary = results_df[["MAE", "MSE", "RMSE", "MAPE", "R2", "elapsed_s", "n_epochs"]].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    # ensure detail is up-to-date
    results_df.to_csv(detail_path, index=False)
    summary_out = pd.DataFrame(
        [
            {
                "model": args.model,
                "MAE_mean": round(summary.loc["MAE", "mean"], 4),
                "MAE_std": round(summary.loc["MAE", "std"], 4),
                "MSE_mean": round(summary.loc["MSE", "mean"], 4),
                "MSE_std": round(summary.loc["MSE", "std"], 4),
                "RMSE_mean": round(summary.loc["RMSE", "mean"], 4),
                "RMSE_std": round(summary.loc["RMSE", "std"], 4),
                "MAPE_mean": round(summary.loc["MAPE", "mean"], 2),
                "MAPE_std": round(summary.loc["MAPE", "std"], 2),
                "R2_mean": round(summary.loc["R2", "mean"], 4),
                "R2_std": round(summary.loc["R2", "std"], 4),
                "elapsed_s_mean": round(summary.loc["elapsed_s", "mean"], 1),
                "elapsed_s_std": round(summary.loc["elapsed_s", "std"], 1),
                "n_epochs_mean": round(summary.loc["n_epochs", "mean"], 1),
                "n_epochs_std": round(summary.loc["n_epochs", "std"], 1),
                "n_households_planned": len(selected_cols),
                "n_households_done": len(results_df),
                "resume_mode": int(not args.no_resume),
            }
        ]
    )
    summary_out.to_csv(summary_path, index=False)

    print(f"[DONE] model={args.model}")
    print(f"  detail : {detail_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
