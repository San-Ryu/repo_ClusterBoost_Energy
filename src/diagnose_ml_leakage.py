"""
Diagnose target leakage in ML feature matrices.

The check compares every feature with the configured prediction target and
reports exact identity or near-perfect correlation candidates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_ml_checkpointed import (
    FEATURE_MODES,
    feature_households_for_mode,
    load_cluster_map,
    load_feature_df,
    parse_lags,
    select_households,
)
from core.project_paths import data_path


def target_values(df: pd.DataFrame, target_mode: str) -> pd.Series:
    if target_mode == "delta":
        return df["energy"] - df["lag_1"]
    return df["energy"]


def diagnose_household(df: pd.DataFrame, household: str, target_mode: str, corr_threshold: float) -> list[dict[str, object]]:
    target = target_values(df, target_mode)
    records: list[dict[str, object]] = []
    for col in [c for c in df.columns if c != "energy"]:
        feature = df[col]
        max_abs_diff = float((feature - target).abs().max())
        corr = float(feature.corr(target)) if feature.std() > 0 and target.std() > 0 else np.nan
        exact_match = bool(max_abs_diff == 0.0)
        high_corr = bool(np.isfinite(corr) and abs(corr) >= corr_threshold)
        if exact_match or high_corr:
            records.append(
                {
                    "household": household,
                    "feature": col,
                    "target_mode": target_mode,
                    "max_abs_diff_to_target": max_abs_diff,
                    "corr_to_target": corr,
                    "exact_match": int(exact_match),
                    "high_corr": int(high_corr),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ML features for target leakage")
    parser.add_argument(
        "--data-path",
        default=str(data_path("KIER_3_Temporal_Resolution", "seasonal_naive", "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv")),
    )
    parser.add_argument("--feature-cache-dir", default=str(data_path("KIER_5_Features", "ml")))
    parser.add_argument("--n-households", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", choices=["level", "delta"], default="delta")
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, default="single")
    parser.add_argument("--peer-lags", default="1")
    parser.add_argument("--cluster-dir", default=str(data_path("KIER_4_Clustered", "seasonal_naive", "kmeans")))
    parser.add_argument("--cluster-resolution", default="10MIN")
    parser.add_argument("--corr-threshold", type=float, default=0.999)
    parser.add_argument(
        "--fail-on",
        choices=["none", "exact", "any"],
        default="exact",
        help="Exit non-zero when leakage candidates are found.",
    )
    parser.add_argument("--force-feature-cache", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--output", default="results/models/ml_leakage_diagnostics.csv")
    args = parser.parse_args()

    raw = pd.read_csv(
        args.data_path,
        parse_dates=["METER_DATE"],
        index_col="METER_DATE",
    ).sort_index()
    households = select_households(raw, args.n_households, args.seed)
    cache_dir = Path(args.feature_cache_dir)
    peer_lags = parse_lags(args.peer_lags)
    cluster_map = (
        load_cluster_map(Path(args.cluster_dir), args.cluster_resolution)
        if args.feature_mode == "cluster_households"
        else {}
    )

    records: list[dict[str, object]] = []
    for i, household in enumerate(households, start=1):
        feature_households, cluster_id = feature_households_for_mode(
            raw=raw,
            household=household,
            feature_mode=args.feature_mode,
            cluster_map=cluster_map,
        )
        feat_df = load_feature_df(
            raw=raw,
            household=household,
            cache_dir=cache_dir,
            use_cache=not args.no_feature_cache,
            force_cache=args.force_feature_cache,
            feature_mode=args.feature_mode,
            feature_households=feature_households,
            peer_lags=peer_lags,
            cluster_id=cluster_id,
        )
        household_records = diagnose_household(
            feat_df,
            household=household,
            target_mode=args.target_mode,
            corr_threshold=args.corr_threshold,
        )
        records.extend(household_records)
        print(
            f"[LEAKAGE] ({i}/{len(households)}) {household}: "
            f"candidates={len(household_records)}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    if len(out_df) == 0:
        out_df = pd.DataFrame(
            columns=[
                "household",
                "feature",
                "target_mode",
                "max_abs_diff_to_target",
                "corr_to_target",
                "exact_match",
                "high_corr",
            ]
        )
    out_df.to_csv(out_path, index=False)
    print(f"[DONE] leakage_candidates={len(out_df)} output={out_path}")
    exact_count = int(out_df["exact_match"].sum()) if len(out_df) > 0 else 0
    if args.fail_on == "exact" and exact_count > 0:
        print(f"[FAIL] exact target leakage candidates={exact_count}", file=sys.stderr)
        sys.exit(1)
    if args.fail_on == "any" and len(out_df) > 0:
        print(f"[FAIL] leakage candidates={len(out_df)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
