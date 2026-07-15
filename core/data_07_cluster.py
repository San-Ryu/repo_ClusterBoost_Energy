"""
data_07_cluster.py
==================
Stage 3: KIER_3_Temporal_Resolution 내 보간법 × 시간 해상도 데이터셋에 대해
K-Means(K=2~11, 10회) 및 DBSCAN(적응형 eps 10회) 군집화 수행.

지원 보간법 : baseline · pchip · stl · mstl · seasonal_naive
지원 해상도 : 10MIN · 1H · 1D · 1W · 1M
평가 지표   : Silhouette Score · Davies-Bouldin Index · Calinski-Harabasz Score
결과 저장   : results/clustering/result_clustering.md

피처 구성 전략
  10MIN · 1H  : 일평균 소비 패턴 (시간대별 평균) → 패턴 Shape 기반 군집화
  1D · 1W · 1M: 전체 기간 전치(348 × T) → Z-score 정규화 → PCA 압축

Usage:
    # 전체 실행 (5보간법 × 5해상도 × 2알고리즘)
    python core/data_07_cluster.py

    # 특정 보간법만
    python core/data_07_cluster.py --interp-methods seasonal_naive pchip

    # 특정 해상도만 (빠른 테스트)
    python core/data_07_cluster.py --resolutions 1D 1W 1M

    # 특정 알고리즘만
    python core/data_07_cluster.py --cluster-methods kmeans
    python core/data_07_cluster.py --cluster-methods dbscan

    # 조합 예시
    python core/data_07_cluster.py --interp-methods seasonal_naive --resolutions 1H 1D

History:
    2026-05-03  Created
    2026-05-03  --interp-methods 추가 — 나머지 보간법도 일괄 처리 가능
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from core.project_paths import data_path
except ModuleNotFoundError:
    from project_paths import data_path

warnings.filterwarnings("ignore")

# =========================================================================== #
#  경로 상수
# =========================================================================== #

ROOT_DIR = Path(__file__).parent.parent
DATA_BASE = data_path("KIER_3_Temporal_Resolution")
OUT_DIR   = ROOT_DIR / "results" / "clustering"
OUT_MD    = OUT_DIR / "result_clustering.md"

ALL_INTERP_METHODS = ["baseline", "pchip", "stl", "mstl", "seasonal_naive"]

DATASETS = {
    "10MIN": "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv",
    "1H"   : "KIER_USAGE_ELEC_INST_INTERP_1H.csv",
    "1D"   : "KIER_USAGE_ELEC_INST_INTERP_1D.csv",
    "1W"   : "KIER_USAGE_ELEC_INST_INTERP_1W.csv",
    "1M"   : "KIER_USAGE_ELEC_INST_INTERP_1M.csv",
}

# =========================================================================== #
#  하이퍼파라미터
# =========================================================================== #

KMEANS_K_VALUES  = list(range(2, 12))   # k = 2 ~ 11 (10회)
KMEANS_N_INIT    = 20
DBSCAN_MIN_SAMP  = 5
N_DBSCAN_TRIALS  = 10
PCA_MAX_COMP     = 30
RANDOM_SEED      = 42


# =========================================================================== #
#  데이터 경로 탐색
# =========================================================================== #

def resolve_data_path(interp_method: str, resolution: str) -> Path | None:
    """
    보간법 + 해상도에 해당하는 CSV 경로를 탐색.
    KIER_3_Temporal_Resolution/{method}/{file} 순으로 시도.
    """
    fname = DATASETS[resolution]
    candidate = DATA_BASE / interp_method / fname
    if candidate.exists():
        return candidate
    return None


# =========================================================================== #
#  피처 행렬 구성
# =========================================================================== #

def build_feature_matrix(df: pd.DataFrame, resolution: str) -> np.ndarray:
    """
    Wide-format (시간 × 세대) DataFrame → 군집화용 특징 행렬 (세대 × 피처).

    10MIN / 1H : 일평균 소비 패턴(시간대별 평균) — 패턴 Shape 포착
    1D / 1W / 1M: 전체 기간 전치 → Z-score → PCA 압축
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if resolution == "10MIN":
        tod = df.index.hour * 6 + df.index.minute // 10
        X = df.groupby(tod).mean().T.values          # (348, 144)
    elif resolution == "1H":
        X = df.groupby(df.index.hour).mean().T.values  # (348, 24)
    else:
        X = df.T.values                               # (348, T)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = StandardScaler().fit_transform(X)

    n_comp = min(PCA_MAX_COMP, X.shape[1], X.shape[0] - 1)
    if X.shape[1] > PCA_MAX_COMP:
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        X = pca.fit_transform(X)
        print(f"    PCA {X.shape[1]}→{n_comp}D, "
              f"설명 분산: {pca.explained_variance_ratio_.sum():.1%}")

    return X


# =========================================================================== #
#  평가 지표
# =========================================================================== #

def _safe_metric(fn, X, labels):
    uniq = np.unique(labels[labels >= 0])
    if len(uniq) < 2:
        return np.nan
    mask = labels >= 0
    if mask.sum() < 2:
        return np.nan
    try:
        return fn(X[mask], labels[mask])
    except Exception:
        return np.nan


def eval_metrics(X, labels):
    return {
        "silhouette": _safe_metric(silhouette_score, X, labels),
        "dbi"       : _safe_metric(davies_bouldin_score, X, labels),
        "ch"        : _safe_metric(calinski_harabasz_score, X, labels),
    }


# =========================================================================== #
#  K-Means (k=2~11, 10회)
# =========================================================================== #

def run_kmeans(X: np.ndarray, interp: str, resolution: str) -> list[dict]:
    records = []
    for trial, k in enumerate(KMEANS_K_VALUES, start=1):
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                    random_state=RANDOM_SEED, max_iter=500)
        labels = km.fit_predict(X)
        m = eval_metrics(X, labels)
        counts = np.bincount(labels[labels >= 0])
        records.append({
            "interp"      : interp,
            "resolution"  : resolution,
            "algorithm"   : "K-Means",
            "trial"       : trial,
            "k"           : k,
            "n_clusters"  : int(np.unique(labels).size),
            "noise_ratio" : 0.0,
            "cluster_min" : int(counts.min()) if len(counts) else 0,
            "cluster_max" : int(counts.max()) if len(counts) else 0,
            "silhouette"  : m["silhouette"],
            "dbi"         : m["dbi"],
            "ch"          : m["ch"],
        })
        sil_s = f"{m['silhouette']:.3f}" if not np.isnan(m["silhouette"]) else "NaN"
        dbi_s = f"{m['dbi']:.3f}"        if not np.isnan(m["dbi"])        else "NaN"
        ch_s  = f"{m['ch']:.1f}"         if not np.isnan(m["ch"])         else "NaN"
        print(f"    K-Means k={k:2d} | Sil={sil_s} DBI={dbi_s} CH={ch_s}")
    return records


# =========================================================================== #
#  DBSCAN (적응형 eps, 10회)
# =========================================================================== #

def _adaptive_eps(X: np.ndarray, k: int = DBSCAN_MIN_SAMP) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    dists, _ = nn.kneighbors(X)
    return np.percentile(np.sort(dists[:, -1]),
                         np.linspace(10, 100, N_DBSCAN_TRIALS))


def run_dbscan(X: np.ndarray, interp: str, resolution: str) -> list[dict]:
    records = []
    for trial, eps in enumerate(_adaptive_eps(X), start=1):
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMP, n_jobs=-1).fit_predict(X)
        n_cl   = int(len(set(labels)) - (1 if -1 in labels else 0))
        n_rat  = float((labels == -1).mean())
        m      = eval_metrics(X, labels)
        counts = np.bincount(labels[labels >= 0]) if n_cl > 0 else np.array([0])
        records.append({
            "interp"      : interp,
            "resolution"  : resolution,
            "algorithm"   : "DBSCAN",
            "trial"       : trial,
            "eps"         : round(float(eps), 4),
            "min_samples" : DBSCAN_MIN_SAMP,
            "n_clusters"  : n_cl,
            "noise_ratio" : round(n_rat, 3),
            "cluster_min" : int(counts.min()) if n_cl > 0 else 0,
            "cluster_max" : int(counts.max()) if n_cl > 0 else 0,
            "silhouette"  : m["silhouette"],
            "dbi"         : m["dbi"],
            "ch"          : m["ch"],
        })
        sil_s = f"{m['silhouette']:.3f}" if not np.isnan(m["silhouette"]) else "NaN"
        print(f"    DBSCAN eps={eps:.4f} | clusters={n_cl:3d} "
              f"noise={n_rat:.1%} Sil={sil_s}")
    return records


# =========================================================================== #
#  Markdown 생성
# =========================================================================== #

_F = lambda v, d=3: f"{v:.{d}f}" if (isinstance(v, float) and not np.isnan(v)) else "—"


def _best_km(rows):
    valid = [r for r in rows if not np.isnan(r["silhouette"])]
    return max(valid, key=lambda r: r["silhouette"]) if valid else None


def _best_db(rows):
    valid = [r for r in rows if r["n_clusters"] >= 2
             and not np.isnan(r["silhouette"])]
    return max(valid, key=lambda r: r["silhouette"]) if valid else None


def _km_table(rows) -> str:
    hdr = ("| 해상도 | Trial | k | Silhouette↑ | DBI↓ | CH↑ "
           "| 최소세대 | 최대세대 |")
    sep = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['resolution']} | {r['trial']} | {r['k']} "
            f"| {_F(r['silhouette'])} | {_F(r['dbi'])} | {_F(r['ch'],1)} "
            f"| {r['cluster_min']} | {r['cluster_max']} |"
        )
    return "\n".join(lines)


def _db_table(rows) -> str:
    hdr = ("| 해상도 | Trial | eps | min_samples | n_clusters "
           "| noise_ratio | Silhouette↑ | DBI↓ | CH↑ |")
    sep = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['resolution']} | {r['trial']} | {r['eps']:.4f} "
            f"| {r['min_samples']} | {r['n_clusters']} "
            f"| {r['noise_ratio']:.1%} "
            f"| {_F(r['silhouette'])} | {_F(r['dbi'])} | {_F(r['ch'],1)} |"
        )
    return "\n".join(lines)


def _summary_table(all_km, all_db, interp_methods, resolutions) -> str:
    hdr = ("| 보간법 | 해상도 | KM 최적 k | Sil(KM) "
           "| DBSCAN 최적 eps | clusters | Sil(DB) |")
    sep = "|:---|:---|:---:|:---:|:---:|:---:|:---:|"
    lines = [hdr, sep]
    for itp in interp_methods:
        for res in resolutions:
            km_rows = [r for r in all_km
                       if r["interp"] == itp and r["resolution"] == res]
            db_rows = [r for r in all_db
                       if r["interp"] == itp and r["resolution"] == res]
            bk = _best_km(km_rows)
            bd = _best_db(db_rows)
            km_k   = f"k={bk['k']}" if bk else "—"
            km_sil = _F(bk["silhouette"]) if bk else "—"
            db_eps = f"{bd['eps']:.4f}" if bd else "—"
            db_nc  = str(bd["n_clusters"]) if bd else "—"
            db_sil = _F(bd["silhouette"]) if bd else "—"
            lines.append(
                f"| {itp} | {res} | {km_k} | {km_sil} "
                f"| {db_eps} | {db_nc} | {db_sil} |"
            )
    return "\n".join(lines)


def write_markdown(all_km, all_db, elapsed,
                   interp_methods, resolutions) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 군집화 분석 결과",
        "",
        f"**실행일시**: {now}  ",
        f"**소요시간**: {elapsed:.1f}초  ",
        f"**데이터**: `{DATA_BASE}/{{보간법}}/`  ",
        f"**보간법**: {', '.join(interp_methods)}  ",
        f"**해상도**: {', '.join(resolutions)}  ",
        "**알고리즘**: K-Means (k=2~11, 10회) · DBSCAN (적응형 eps 10회)  ",
        "**평가지표**: Silhouette(↑) · DBI(↓) · Calinski-Harabasz(↑)",
        "",
        "---",
        "",
        "## 1. 피처 공학",
        "",
        "| 해상도 | 피처 구성 | 행렬 크기 |",
        "|:---|:---|:---|",
        "| 10MIN | 일내 144 시간대 평균 소비 패턴 | 348 × 144 |",
        "| 1H   | 24시간 평균 소비 패턴 | 348 × 24 |",
        f"| 1D   | 전치 → Z-score → PCA | 348 × ≤{PCA_MAX_COMP} |",
        f"| 1W   | 전치 → Z-score → PCA | 348 × ≤{PCA_MAX_COMP} |",
        "| 1M   | 전치 → Z-score | 348 × 24 |",
        "",
        "---",
        "",
        "## 2. 결과 요약 (보간법 × 해상도 최적 설정)",
        "",
        _summary_table(all_km, all_db, interp_methods, resolutions),
        "",
        "> Silhouette 기준 최우수 설정. DBSCAN은 유효 군집 ≥ 2 조건 충족 시만 선택.",
        "",
        "---",
        "",
    ]

    # ── 보간법별 섹션 ──────────────────────────────────────────────────────────
    for sec_i, itp in enumerate(interp_methods, start=3):
        lines += [
            f"## {sec_i}. 보간법: {itp}",
            "",
        ]

        for res in resolutions:
            km_rows = [r for r in all_km
                       if r["interp"] == itp and r["resolution"] == res]
            db_rows = [r for r in all_db
                       if r["interp"] == itp and r["resolution"] == res]

            if not km_rows and not db_rows:
                lines += [f"### {itp} / {res} — 데이터 없음", ""]
                continue

            bk = _best_km(km_rows)
            bd = _best_db(db_rows)
            bk_note = (f" · 최적 k={bk['k']} (Sil={_F(bk['silhouette'])})"
                       if bk else "")
            bd_note = (f" · 최적 eps={bd['eps']:.4f} "
                       f"(clusters={bd['n_clusters']}, Sil={_F(bd['silhouette'])})"
                       if bd else " · 유효 군집 없음")

            lines += [f"### {itp} / {res}", ""]

            if km_rows:
                lines += [
                    f"**K-Means**{bk_note}",
                    "",
                    _km_table(km_rows),
                    "",
                ]
            if db_rows:
                lines += [
                    f"**DBSCAN**{bd_note}",
                    "",
                    _db_table(db_rows),
                    "",
                ]

    lines += [
        "---",
        "",
        "## 알고리즘 비교 지침",
        "",
        "| | K-Means | DBSCAN |",
        "|:---|:---|:---|",
        "| 군집 수 | 사전 지정 (k) | 자동 결정 |",
        "| 이상치 | 없음 | 노이즈(-1) 분리 |",
        "| 선택 기준 | Silhouette·CH 높음 + DBI 낮음 | noise_ratio<15% + Silhouette 최대 |",
        "",
        f"*생성: {now} by `core/data_07_cluster.py`*",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장: {OUT_MD}")


# =========================================================================== #
#  메인
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="KIER_3_Temporal_Resolution 보간법 × 해상도 군집화 비교")
    parser.add_argument(
        "--interp-methods", nargs="+",
        choices=ALL_INTERP_METHODS,
        default=ALL_INTERP_METHODS,
        metavar="METHOD",
        help=f"보간법 (기본: 전체 {ALL_INTERP_METHODS})")
    parser.add_argument(
        "--resolutions", nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        metavar="RES",
        help=f"시간 해상도 (기본: {' '.join(DATASETS.keys())})")
    parser.add_argument(
        "--cluster-methods", nargs="+",
        choices=["kmeans", "dbscan"],
        default=["kmeans", "dbscan"],
        metavar="ALG",
        help="군집화 알고리즘 (기본: kmeans dbscan)")
    args = parser.parse_args()

    import time
    t0 = time.time()
    all_km, all_db = [], []

    for itp in args.interp_methods:
        print(f"\n{'='*64}")
        print(f"  보간법: {itp}")
        print(f"{'='*64}")

        for res in args.resolutions:
            csv_path = resolve_data_path(itp, res)

            if csv_path is None:
                print(f"\n  [{res}] ⚠ 파일 없음 — 건너뜀: "
                      f"{DATA_BASE}/{itp}/{DATASETS[res]}")
                continue

            print(f"\n  [{itp}/{res}] {csv_path.name}")
            df = pd.read_csv(csv_path, parse_dates=["METER_DATE"],
                             index_col="METER_DATE")
            print(f"  로드: {df.shape[0]:,} × {df.shape[1]} | "
                  f"NaN: {df.isna().sum().sum():,}")

            print("  피처 행렬 구성 중...")
            X = build_feature_matrix(df, res)
            print(f"  특징 행렬: {X.shape}")

            if "kmeans" in args.cluster_methods:
                print(f"\n  [K-Means] k=2~11 실행...")
                all_km.extend(run_kmeans(X, itp, res))

            if "dbscan" in args.cluster_methods:
                print(f"\n  [DBSCAN] 적응형 eps 10회 실행...")
                all_db.extend(run_dbscan(X, itp, res))

    elapsed = time.time() - t0
    print(f"\n{'='*64}")
    print(f"전체 소요: {elapsed:.1f}s")

    write_markdown(all_km, all_db, elapsed,
                   args.interp_methods, args.resolutions)

    # ── 터미널 요약 ─────────────────────────────────────────────────────────────
    print("\n[K-Means 최적 k 요약]")
    for itp in args.interp_methods:
        for res in args.resolutions:
            rows = [r for r in all_km
                    if r["interp"] == itp and r["resolution"] == res]
            bk = _best_km(rows)
            if bk:
                print(f"  {itp:15s} {res:6s}: k={bk['k']:2d}  "
                      f"Sil={bk['silhouette']:.4f}  "
                      f"DBI={bk['dbi']:.4f}  CH={bk['ch']:.1f}")

    print("\n[DBSCAN 최적 설정 요약]")
    for itp in args.interp_methods:
        for res in args.resolutions:
            rows = [r for r in all_db
                    if r["interp"] == itp and r["resolution"] == res]
            bd = _best_db(rows)
            if bd:
                print(f"  {itp:15s} {res:6s}: eps={bd['eps']:.4f}  "
                      f"clusters={bd['n_clusters']:3d}  "
                      f"noise={bd['noise_ratio']:.1%}  "
                      f"Sil={bd['silhouette']:.4f}")
            elif rows:
                print(f"  {itp:15s} {res:6s}: 유효 군집 없음")


if __name__ == "__main__":
    main()
