"""
data_05_resample.py
====================
Stage 2 → Stage 3 중간 단계:
각 보간법 Parquet(10분)을 읽어 5가지 시간 해상도 CSV로 재생성한다.

출력 경로: 기본 ~/data/KIER_2_Temporal_Resolution/{method}/{RESOLUTION}.csv

해상도별 집계 방식: SUM
  - INST = 10분 구간의 에너지 사용량(kWh)
  - 더 긴 구간으로 합산 → 해당 기간 총 에너지 소비량

Usage:
    python core/data_05_resample.py
    python core/data_05_resample.py --methods pchip seasonal_naive
    python core/data_05_resample.py --resolutions 1H 1D
    python core/data_05_resample.py --skip-10min   # 10분 CSV 생략

History:
    2026-05-02  Created
    2026-05-02  Renamed from build_kier_temporal_resample.py → data_05_resample.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from core.project_paths import data_path
except ModuleNotFoundError:
    from project_paths import data_path

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
_IN_DIR  = data_path("KIER_S2_interp")
_OUT_DIR = data_path("KIER_2_Temporal_Resolution")

# ── 해상도 정의 ────────────────────────────────────────────────────────────────
# (출력 파일명 접미사, pandas resample 주파수 문자열, 설명)
RESOLUTIONS: dict[str, tuple[str, str]] = {
    "10MIN": ("10min",  "원본 10분 간격 (재샘플 없음)"),
    "1H":   ("1h",     "1시간 합산"),
    "1D":   ("1D",     "1일 합산"),
    "1W":   ("1W",     "1주 합산 (일요일 기준 주 마감)"),
    "1M":   ("1ME",    "1월 합산 (월말 기준)"),
}

ALL_METHODS = ["baseline", "pchip", "stl", "mstl", "seasonal_naive"]


# ══════════════════════════════════════════════════════════════════════════════
#  핵심 로직
# ══════════════════════════════════════════════════════════════════════════════

def _resample_df(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """METER_DATE를 인덱스로 하여 지정 주파수로 합산 리샘플."""
    df = df.set_index("METER_DATE")
    df_res = df.resample(freq).sum(min_count=1)   # min_count=1 → 빈 구간은 NaN
    df_res.index.name = "METER_DATE"
    return df_res.reset_index()


def process_method(method: str, resolutions: list[str],
                   skip_10min: bool) -> list[dict]:
    """단일 보간법의 Parquet을 로드하여 모든 해상도 CSV를 저장한다."""
    in_path = _IN_DIR / method / "KIER_USAGE_ELEC_INST_INTERP.parquet"
    if not in_path.exists():
        print(f"  [SKIP] Parquet 없음: {in_path}")
        return []

    print(f"\n  [{method}] Parquet 로드 중...", end=" ", flush=True)
    t0 = time.time()
    df = pd.read_parquet(in_path)
    house_cols = [c for c in df.columns if c != "METER_DATE"]
    print(f"{len(house_cols)}세대 {len(df):,}행 ({time.time()-t0:.1f}s)")

    logs = []
    for res_key in resolutions:
        if res_key == "10MIN" and skip_10min:
            print(f"    {res_key:6s}  [SKIP] --skip-10min 옵션")
            continue

        freq_str, desc = RESOLUTIONS[res_key]
        out_dir = _OUT_DIR / method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"KIER_USAGE_ELEC_INST_INTERP_{res_key}.csv"

        t1 = time.time()
        if res_key == "10MIN":
            df_out = df                              # 그대로 내보냄
        else:
            df_out = _resample_df(df, freq_str)

        df_out.to_csv(out_path, index=False)
        elapsed = time.time() - t1

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    {res_key:6s}  {len(df_out):>8,}행  "
              f"{size_mb:6.1f} MB  {elapsed:5.1f}s  → {out_path.name}")

        logs.append({
            "method":      method,
            "resolution":  res_key,
            "freq":        freq_str,
            "rows":        len(df_out),
            "house_cols":  len(house_cols),
            "size_mb":     round(size_mb, 2),
            "elapsed_s":   round(elapsed, 1),
            "path":        str(out_path),
        })
    return logs


# ══════════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="KIER ELEC INST 시간 해상도별 CSV 재생성")
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        choices=ALL_METHODS,
        help=f"처리할 보간법 (기본: 전체 {len(ALL_METHODS)}안)")
    parser.add_argument(
        "--resolutions", nargs="+", default=list(RESOLUTIONS.keys()),
        choices=list(RESOLUTIONS.keys()),
        help="생성할 해상도 (기본: 10MIN 1H 1D 1W 1M)")
    parser.add_argument(
        "--skip-10min", action="store_true",
        help="10분 CSV 생성 생략 (파일이 크므로 필요 시 제외 가능)")
    args = parser.parse_args()

    print("=" * 64)
    print("KIER ELEC INST — 시간 해상도별 CSV 재생성")
    print(f"  입력: {_IN_DIR}")
    print(f"  출력: {_OUT_DIR}")
    print(f"  보간법: {args.methods}")
    print(f"  해상도: {args.resolutions}")
    print("=" * 64)

    all_logs = []
    for method in args.methods:
        logs = process_method(method, args.resolutions, args.skip_10min)
        all_logs.extend(logs)

    if all_logs:
        log_df = pd.DataFrame(all_logs)
        log_path = _OUT_DIR / "resample_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)

        print("\n" + "=" * 64)
        print("생성 완료 요약")
        print("=" * 64)
        pivot = (log_df.pivot_table(index="method", columns="resolution",
                                     values="size_mb", aggfunc="sum")
                 .reindex(columns=list(RESOLUTIONS.keys()), fill_value=0))
        print(pivot.to_string())
        print(f"\n로그 저장: {log_path}")

    print("\n✓ 완료")


if __name__ == "__main__":
    main()
