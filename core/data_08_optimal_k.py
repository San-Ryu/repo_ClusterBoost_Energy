"""
data_08_optimal_k.py
====================
Task 1: seasonal_naive 5개 시간 해상도 데이터셋에서 K=2~15에 대한
        군집화 지표를 계산하고 최적 K를 도출한다.

지표
  Inertia       (Elbow curve — 낮을수록 좋음, 단 급격히 완만해지는 Knee)
  Silhouette    (↑, -1~1)
  Calinski-Harabasz Index  (↑)
  Dunn Index    (↑)
  Davies-Bouldin Score     (↓)

최적 K 결정: 5개 지표별 최우수 K에 대한 다수결 (Voting)
결과 저장  : results/clustering/result_clustering.md  (테이블)
            results/clustering/optimal_k.json         (스크립트 2가 읽음)

Usage:
    python core/data_08_optimal_k.py
    python core/data_08_optimal_k.py --resolutions 1D 1W 1M

History:
    2026-05-07  Created
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from core.project_paths import data_path
except ModuleNotFoundError:
    from project_paths import data_path

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = data_path("KIER_3_Temporal_Resolution", "seasonal_naive")
OUT_DIR   = ROOT / "results" / "clustering"
OUT_MD    = OUT_DIR / "result_clustering.md"
OUT_JSON  = OUT_DIR / "optimal_k.json"

DATASETS  = {
    "10MIN": "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv",
    "1H"   : "KIER_USAGE_ELEC_INST_INTERP_1H.csv",
    "1D"   : "KIER_USAGE_ELEC_INST_INTERP_1D.csv",
    "1W"   : "KIER_USAGE_ELEC_INST_INTERP_1W.csv",
    "1M"   : "KIER_USAGE_ELEC_INST_INTERP_1M.csv",
}

K_MIN, K_MAX   = 2, 15
PCA_MAX_COMP   = 30
KMEANS_N_INIT  = 20
RANDOM_SEED    = 42

# ── Dunn Index (ref_cluster_eval.py 래퍼) ────────────────────────────────────
try:
    sys.path.insert(0, str(ROOT))
    from core.ref_cluster_eval import get_dunn_index as _ext_dunn
    def dunn_index(X, labels):
        try:
            return _ext_dunn(X, labels)
        except Exception:
            return _dunn_fallback(X, labels)
except ImportError:
    def dunn_index(X, labels):
        return _dunn_fallback(X, labels)


def _dunn_fallback(X: np.ndarray, labels: np.ndarray) -> float:
    """인터-클러스터 최솟값 / 인트라-클러스터 최댓값."""
    from scipy.spatial.distance import cdist
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    clusters = [X[labels == lb] for lb in uniq]

    min_inter = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            d = cdist(clusters[i], clusters[j]).min()
            if d < min_inter:
                min_inter = d

    max_intra = 0.0
    for c in clusters:
        if c.shape[0] < 2:
            continue
        from scipy.spatial.distance import pdist
        d = pdist(c).max()
        if d > max_intra:
            max_intra = d

    return float(min_inter / max_intra) if max_intra > 0 else np.nan


# =========================================================================== #
#  피처 행렬 구성 (data_07_cluster.py 동일 방식)
# =========================================================================== #

def build_feature_matrix(df: pd.DataFrame, resolution: str) -> np.ndarray:
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if resolution == "10MIN":
        tod = df.index.hour * 6 + df.index.minute // 10
        X = df.groupby(tod).mean().T.values
    elif resolution == "1H":
        X = df.groupby(df.index.hour).mean().T.values
    else:
        X = df.T.values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = StandardScaler().fit_transform(X)

    n_comp = min(PCA_MAX_COMP, X.shape[1], X.shape[0] - 1)
    if X.shape[1] > PCA_MAX_COMP:
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        X = pca.fit_transform(X)
        print(f"    PCA → {n_comp}D  (설명분산 {pca.explained_variance_ratio_.sum():.1%})")
    return X


# =========================================================================== #
#  Elbow Knee 탐지 (수직 거리 최대화)
# =========================================================================== #

def find_elbow_k(k_values: list[int], inertias: list[float]) -> int:
    x = np.array(k_values, dtype=float)
    y = np.array(inertias, dtype=float)
    # [0,1] 정규화
    xn = (x - x[0]) / max(x[-1] - x[0], 1e-9)
    yn = (y - y[0]) / max(y[-1] - y[0], 1e-9)
    # 첫점–끝점 연결 직선까지의 수직 거리
    a, b = xn[-1] - xn[0], yn[-1] - yn[0]
    nrm = np.sqrt(a ** 2 + b ** 2) + 1e-9
    dists = [abs(b * xi - a * yi + a * yn[0] - b * xn[0]) / nrm
             for xi, yi in zip(xn, yn)]
    return k_values[int(np.argmax(dists))]


# =========================================================================== #
#  K=2~15 지표 계산
# =========================================================================== #

def compute_metrics_per_k(X: np.ndarray, k_values: list[int]) -> list[dict]:
    records = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT,
                    random_state=RANDOM_SEED, max_iter=500)
        labels = km.fit_predict(X)

        sil  = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        ch   = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan
        dbi  = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else np.nan
        di   = dunn_index(X, labels)

        records.append({
            "k"        : k,
            "inertia"  : km.inertia_,
            "silhouette": sil,
            "ch"       : ch,
            "dunn"     : di,
            "dbi"      : dbi,
        })
        print(f"    k={k:2d} | inertia={km.inertia_:.1f}  "
              f"Sil={sil:.4f}  CH={ch:.1f}  Dunn={di:.4f}  DBI={dbi:.4f}")
    return records


# =========================================================================== #
#  최적 K 결정 (5-way Voting)
# =========================================================================== #

def optimal_k_vote(records: list[dict]) -> dict:
    k_vals    = [r["k"]         for r in records]
    inertias  = [r["inertia"]   for r in records]
    sil_vals  = [r["silhouette"]for r in records]
    ch_vals   = [r["ch"]        for r in records]
    dunn_vals = [r["dunn"]      for r in records]
    dbi_vals  = [r["dbi"]       for r in records]

    # 지표별 최우수 K
    k_elbow = find_elbow_k(k_vals, inertias)
    k_sil   = k_vals[int(np.nanargmax(sil_vals))]
    k_ch    = k_vals[int(np.nanargmax(ch_vals))]
    k_dunn  = k_vals[int(np.nanargmax(dunn_vals))]
    k_dbi   = k_vals[int(np.nanargmin(dbi_vals))]

    votes = Counter([k_elbow, k_sil, k_ch, k_dunn, k_dbi])
    best_k = votes.most_common(1)[0][0]

    return {
        "optimal_k" : best_k,
        "k_elbow"   : k_elbow,
        "k_sil"     : k_sil,
        "k_ch"      : k_ch,
        "k_dunn"    : k_dunn,
        "k_dbi"     : k_dbi,
        "votes"     : dict(votes),
    }


# =========================================================================== #
#  Markdown 저장
# =========================================================================== #

_F = lambda v, d=4: f"{v:.{d}f}" if isinstance(v, float) and not np.isnan(v) else "—"


def _build_table(records: list[dict], opt: dict) -> str:
    hdr = ("| K | Inertia↓ | Silhouette↑ | CH↑ "
           "| Dunn↑ | DBI↓ | 비고 |")
    sep = "|:---:|:---:|:---:|:---:|:---:|:---:|:---|"
    lines = [hdr, sep]
    for r in records:
        k = r["k"]
        tags = []
        if k == opt["k_elbow"]   : tags.append("Elbow")
        if k == opt["k_sil"]     : tags.append("Sil↑")
        if k == opt["k_ch"]      : tags.append("CH↑")
        if k == opt["k_dunn"]    : tags.append("Dunn↑")
        if k == opt["k_dbi"]     : tags.append("DBI↓")
        tag_str = " · ".join(tags) if tags else ""
        marker  = " ← **최적**" if k == opt["optimal_k"] else ""
        lines.append(
            f"| {k} | {_F(r['inertia'],1)} | {_F(r['silhouette'])} "
            f"| {_F(r['ch'],1)} | {_F(r['dunn'])} | {_F(r['dbi'])} "
            f"| {tag_str}{marker} |"
        )
    return "\n".join(lines)


def _build_vote_row(resolution: str, opt: dict) -> str:
    return (
        f"| {resolution} | {opt['k_elbow']} | {opt['k_sil']} "
        f"| {opt['k_ch']} | {opt['k_dunn']} | {opt['k_dbi']} "
        f"| **{opt['optimal_k']}** |"
    )


def write_markdown(all_results: dict, all_opts: dict, elapsed: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 최적 K 도출 — K=2~15 군집화 지표 비교",
        "",
        f"**실행일시**: {now}  ",
        f"**소요시간**: {elapsed:.1f}초  ",
        f"**데이터**: `{DATA_DIR}`  ",
        f"**K 범위**: {K_MIN} ~ {K_MAX}  ",
        "**최적 K 결정**: Elbow · Silhouette↑ · CH↑ · Dunn↑ · DBI↓ 다수결",
        "",
        "---",
        "",
        "## 1. 해상도별 최적 K 요약",
        "",
        "| 해상도 | Elbow K | Sil Best | CH Best | Dunn Best | DBI Best | **최종 K** |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for res, opt in all_opts.items():
        lines.append(_build_vote_row(res, opt))

    lines += ["", "---", ""]

    for sec_i, (res, records) in enumerate(all_results.items(), start=2):
        opt = all_opts[res]
        lines += [
            f"## {sec_i}. {res} — 상세 지표 (K=2~15)",
            "",
            f"> 최적 K = **{opt['optimal_k']}** "
            f"(Elbow={opt['k_elbow']}, Sil={opt['k_sil']}, "
            f"CH={opt['k_ch']}, Dunn={opt['k_dunn']}, DBI={opt['k_dbi']})",
            "",
            _build_table(records, opt),
            "",
        ]

    lines += [
        "---",
        "",
        f"*생성: {now} by `core/data_08_optimal_k.py`*",
        f"*JSON 저장: `results/clustering/optimal_k.json`*",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {OUT_MD}")


# =========================================================================== #
#  메인
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="K=2~15 군집화 지표 계산 & 최적 K 도출")
    parser.add_argument(
        "--resolutions", nargs="+",
        choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
        metavar="RES",
        help=f"해상도 (기본: 전체)")
    args = parser.parse_args()

    import time
    t0 = time.time()
    k_values    = list(range(K_MIN, K_MAX + 1))
    all_results = {}   # {resolution: [metric_records]}
    all_opts    = {}   # {resolution: optimal_k_info}

    for res in args.resolutions:
        csv_path = DATA_DIR / DATASETS[res]
        print(f"\n{'='*60}")
        print(f"[{res}]  {csv_path.name}")

        df = pd.read_csv(csv_path, parse_dates=["METER_DATE"],
                         index_col="METER_DATE")
        print(f"  로드: {df.shape[0]:,} × {df.shape[1]} | "
              f"NaN: {df.isna().sum().sum():,}")

        print("  피처 행렬 구성...")
        X = build_feature_matrix(df, res)
        print(f"  특징 행렬: {X.shape}")

        print(f"\n  K=2~15 지표 계산...")
        records = compute_metrics_per_k(X, k_values)
        opt     = optimal_k_vote(records)

        all_results[res] = records
        all_opts[res]    = opt

        print(f"\n  ▶ 최적 K = {opt['optimal_k']}  "
              f"(Elbow={opt['k_elbow']} Sil={opt['k_sil']} "
              f"CH={opt['k_ch']} Dunn={opt['k_dunn']} DBI={opt['k_dbi']})")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"전체 소요: {elapsed:.1f}s")

    # JSON 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_out = {res: {"optimal_k": opt["optimal_k"], **opt}
                for res, opt in all_opts.items()}
    OUT_JSON.write_text(json.dumps(json_out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # Markdown 저장
    print("\n저장 중...")
    write_markdown(all_results, all_opts, elapsed)
    print(f"  → {OUT_JSON}")

    # 터미널 요약
    print("\n[최적 K 요약]")
    for res, opt in all_opts.items():
        print(f"  {res:6s}: K={opt['optimal_k']}  "
              f"(Elbow={opt['k_elbow']} Sil={opt['k_sil']} "
              f"CH={opt['k_ch']} Dunn={opt['k_dunn']} DBI={opt['k_dbi']})")


if __name__ == "__main__":
    main()
