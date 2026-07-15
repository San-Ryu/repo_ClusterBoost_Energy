"""
Build reusable ML feature caches for Stage 6.

This script preserves the existing feature definition from model_ml_single.py and
stores one household feature matrix per file. It does not change model accuracy;
it only avoids rebuilding lag/rolling features in every model run.
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

from core.project_paths import data_path
from model_ml_single import build_ml_features


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


def cache_path(cache_dir: Path, household: str) -> Path:
    return cache_dir / f"{household}_features.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 6 ML feature cache")
    parser.add_argument(
        "--data-path",
        default=str(data_path("KIER_3_Temporal_Resolution", "seasonal_naive", "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv")),
    )
    parser.add_argument("--cache-dir", default=str(data_path("KIER_5_Features", "ml")))
    parser.add_argument("--n-households", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        args.data_path,
        parse_dates=["METER_DATE"],
        index_col="METER_DATE",
    ).sort_index()
    selected_cols = select_households(raw, args.n_households, args.seed)

    print(f"[CACHE] households={len(selected_cols)} cache_dir={cache_dir}")
    for i, col in enumerate(selected_cols, start=1):
        out_path = cache_path(cache_dir, col)
        if out_path.exists() and not args.force:
            print(f"  ({i}/{len(selected_cols)}) skip {col}: {out_path}")
            continue
        feat_df = build_ml_features(raw[col])
        feat_df.to_parquet(out_path)
        print(f"  ({i}/{len(selected_cols)}) wrote {col}: {out_path} rows={len(feat_df)}")


if __name__ == "__main__":
    main()
