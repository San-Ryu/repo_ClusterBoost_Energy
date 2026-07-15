"""
data_09_cluster_label.py
========================
Task 2: 최적 K로 5가지 군집화 알고리즘 적용 → 세대별 레이블 부여 → 데이터셋 분리 저장.

알고리즘
  K-Means   : sklearn KMeans (optimal K)
  MeanShift : sklearn MeanShift (자동 K)
  DBSCAN    : sklearn DBSCAN (적응형 eps)
  L1L2      : K-Medoids (Manhattan distance) — L1 distance 기반 Robust Clustering
  GMM       : sklearn GaussianMixture (optimal K)

입력
  results/clustering/optimal_k.json          (data_08_optimal_k.py 출력)
  ~/data/KIER_3_Temporal_Resolution/seasonal_naive/

출력
  results/clustering/label_cluster.md        (Task 2-1)
  ~/data/KIER_4_Clustered/seasonal_naive/{algo}/
    KIER_USAGE_ELEC_INST_K{n}_{res}.csv     (Task 2-2)

Usage:
    python core/data_09_cluster_label.py
    python core/data_09_cluster_label.py --resolutions 10MIN 1H
    python core/data_09_cluster_label.py --algos kmeans gmm   # 특정 알고리즘만

History:
    2026-05-07  Created
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from core.project_paths import data_path
except ModuleNotFoundError:
    from project_paths import data_path

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = data_path("KIER_3_Temporal_Resolution", "seasonal_naive")
OUT_CLUS  = data_path("KIER_4_Clustered", "seasonal_naive")
OUT_DIR   = ROOT / "results" / "clustering"
LABEL_MD  = OUT_DIR / "label_cluster.md"
OPT_JSON  = OUT_DIR / "optimal_k.json"

DATASETS = {
    "10MIN": "KIER_USAGE_ELEC_INST_INTERP_10MIN.csv",
    "1H"   : "KIER_USAGE_ELEC_INST_INTERP_1H.csv",
    "1D"   : "KIER_USAGE_ELEC_INST_INTERP_1D.csv",
    "1W"   : "KIER_USAGE_ELEC_INST_INTERP_1W.csv",
    "1M"   : "KIER_USAGE_ELEC_INST_INTERP_1M.csv",
}

ALL_ALGOS   = ["kmeans", "meanshift", "dbscan", "l1l2", "gmm"]
RANDOM_SEED = 42
PCA_MAX     = 30
KMEANS_NINIT = 20

# DBSCAN 파라미터
DBSCAN_MIN_SAMP = 5


# =========================================================================== #
#  피처 행렬 (data_08_optimal_k.py 동일)
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
    n_comp = min(PCA_MAX, X.shape[1], X.shape[0] - 1)
    if X.shape[1] > PCA_MAX:
        X = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit_transform(X)
    return X


# =========================================================================== #
#  K-Medoids (L1 Manhattan distance — L1L2 알고리즘)
# =========================================================================== #

def kmedoids_l1(X: np.ndarray, n_clusters: int,
                max_iter: int = 300, random_state: int = RANDOM_SEED) -> np.ndarray:
    """
    K-Medoids (PAM) with Manhattan (L1) distance.
    L1 norm is more robust to outliers than L2 (Euclidean).
    Combined effect with StandardScaler pre-processing → L1+L2 mixed objective.
    """
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(random_state)
    n   = X.shape[0]

    D   = pairwise_distances(X, metric="manhattan")   # (n, n)
    med = rng.choice(n, size=n_clusters, replace=False)

    for _ in range(max_iter):
        labels = D[:, med].argmin(axis=1)             # assignment
        new_med = np.empty(n_clusters, dtype=int)
        for k in range(n_clusters):
            idx = np.where(labels == k)[0]
            if idx.size == 0:
                new_med[k] = med[k]
                continue
            sub = D[np.ix_(idx, idx)]
            new_med[k] = idx[sub.sum(axis=1).argmin()]
        if np.array_equal(new_med, med):
            break
        med = new_med

    return D[:, med].argmin(axis=1)


# =========================================================================== #
#  군집화 실행 — 알고리즘별
# =========================================================================== #

def run_kmeans(X, k):
    km = KMeans(n_clusters=k, n_init=KMEANS_NINIT,
                random_state=RANDOM_SEED, max_iter=500)
    return km.fit_predict(X)


def run_meanshift(X, k_hint: int):
    """
    MeanShift: 자동 K 결정.
    bandwidth를 조정해 k_hint±2 범위 내 클러스터 수를 목표로 시도.
    """
    bw = estimate_bandwidth(X, quantile=0.2, random_state=RANDOM_SEED)
    for scale in [1.0, 0.8, 0.6, 1.2, 1.5, 0.4, 2.0]:
        ms = MeanShift(bandwidth=bw * scale, bin_seeding=True)
        ms.fit(X)
        n_found = len(np.unique(ms.labels_))
        if abs(n_found - k_hint) <= 2 or scale == 2.0:
            break
    return ms.labels_


def run_dbscan(X, k_hint: int):
    """
    DBSCAN: k-NN 거리 기반 적응형 eps.
    유효 군집 수가 k_hint에 가장 가까운 eps 선택.
    """
    nn = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMP, n_jobs=-1).fit(X)
    dists, _ = nn.kneighbors(X)
    eps_candidates = np.percentile(np.sort(dists[:, -1]),
                                   np.linspace(5, 95, 30))

    best_labels, best_diff = None, np.inf
    for eps in eps_candidates:
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMP,
                        n_jobs=-1).fit_predict(X)
        n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
        diff   = abs(n_cl - k_hint)
        if diff < best_diff and n_cl >= 2:
            best_diff   = diff
            best_labels = labels.copy()
    if best_labels is None:
        best_labels = np.zeros(len(X), dtype=int)
    return best_labels


def run_l1l2(X, k):
    return kmedoids_l1(X, n_clusters=k, random_state=RANDOM_SEED)


def run_gmm(X, k):
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         n_init=5, random_state=RANDOM_SEED, max_iter=300)
    return gm.fit_predict(X)


ALGO_FN = {
    "kmeans"   : run_kmeans,
    "meanshift": run_meanshift,
    "dbscan"   : run_dbscan,
    "l1l2"     : run_l1l2,
    "gmm"      : run_gmm,
}

ALGO_LABEL = {
    "kmeans"   : "K-Means",
    "meanshift": "Mean-Shift",
    "dbscan"   : "DBSCAN",
    "l1l2"     : "L1L2 (K-Medoids)",
    "gmm"      : "GMM",
}


# =========================================================================== #
#  Silhouette (노이즈 제외)
# =========================================================================== #

def safe_sil(X, labels):
    uniq = np.unique(labels[labels >= 0])
    if len(uniq) < 2:
        return np.nan
    mask = labels >= 0
    if mask.sum() < 2:
        return np.nan
    try:
        return silhouette_score(X[mask], labels[mask])
    except Exception:
        return np.nan


# =========================================================================== #
#  Markdown 생성 — label_cluster.md
# =========================================================================== #

def write_label_md(label_table: dict, sil_table: dict,
                   households: list[str], resolutions: list[str],
                   elapsed: float) -> None:
    """
    label_table : {resolution: {algo: labels_array(348,)}}
    sil_table   : {resolution: {algo: silhouette_score}}
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 세대별 군집 레이블 현황",
        "",
        f"**실행일시**: {now}  ",
        f"**소요시간**: {elapsed:.1f}초  ",
        f"**데이터**: `{DATA_DIR}`  ",
        f"**알고리즘**: K-Means · Mean-Shift · DBSCAN · L1L2(K-Medoids) · GMM  ",
        f"**세대 수**: {len(households)}  ",
        "",
        "---",
        "",
        "## 1. 알고리즘별 Silhouette Score 요약",
        "",
        "| 해상도 | K-Means | Mean-Shift | DBSCAN | L1L2 | GMM |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for res in resolutions:
        row_vals = []
        for algo in ALL_ALGOS:
            sil = sil_table.get(res, {}).get(algo, np.nan)
            row_vals.append(f"{sil:.4f}" if not np.isnan(sil) else "—")
        lines.append(f"| {res} | " + " | ".join(row_vals) + " |")

    lines += ["", "---", ""]

    # 해상도별 × 알고리즘별 클러스터 분포
    for res in resolutions:
        lines += [
            f"## 2. {res} — 알고리즘별 클러스터 분포",
            "",
        ]
        for algo in ALL_ALGOS:
            if res not in label_table or algo not in label_table[res]:
                continue
            labels = label_table[res][algo]
            uniq   = sorted(set(labels))
            n_cl   = len([u for u in uniq if u >= 0])
            noise  = (labels == -1).sum()
            sil    = sil_table.get(res, {}).get(algo, np.nan)
            sil_s  = f"{sil:.4f}" if not np.isnan(sil) else "—"

            lines += [
                f"### {res} / {ALGO_LABEL[algo]}",
                f"> 군집 수: {n_cl}  |  노이즈: {noise}세대  |  Silhouette: {sil_s}",
                "",
                "| 군집 ID | 세대 수 | 세대 목록 (최대 10개) |",
                "|:---:|:---:|:---|",
            ]
            for cl_id in uniq:
                hh_in = [h for h, lbl in zip(households, labels) if lbl == cl_id]
                label_disp = f"노이즈(-1)" if cl_id == -1 else f"K{cl_id}"
                preview = ", ".join(hh_in[:10])
                if len(hh_in) > 10:
                    preview += f" ... ({len(hh_in)}세대)"
                lines.append(f"| {label_disp} | {len(hh_in)} | {preview} |")
            lines.append("")

    lines += [
        "---",
        "",
        "## 3. 세대별 레이블 전체 목록",
        "",
        "> 세대 수가 많아 아래 테이블은 최대 50세대만 표시합니다.",
        f"> 전체는 `{OUT_CLUS}` 경로의 CSV 분리 파일을 참조하세요.",
        "",
    ]

    for res in resolutions[:1]:  # 10MIN 또는 첫 해상도만 전체 레이블 표시
        if res not in label_table:
            continue
        hdr = "| 세대 | " + " | ".join(ALGO_LABEL[a] for a in ALL_ALGOS) + " |"
        sep = "|:---|" + ":---:|" * len(ALL_ALGOS)
        lines += [f"### {res} 세대별 레이블", "", hdr, sep]

        show_n = min(50, len(households))
        for i in range(show_n):
            hh = households[i]
            lbls = []
            for algo in ALL_ALGOS:
                lbl = label_table.get(res, {}).get(algo, np.full(len(households), -9))
                v = lbl[i] if i < len(lbl) else -9
                lbls.append(f"K{v}" if v >= 0 else "noise")
            lines.append(f"| {hh} | " + " | ".join(lbls) + " |")

        if len(households) > 50:
            lines.append(f"| ... ({len(households)}세대 중 50세대 표시) | " +
                         " | ".join(["..." for _ in ALL_ALGOS]) + " |")
        lines.append("")

    lines += [
        "---",
        f"*생성: {now} by `core/data_09_cluster_label.py`*",
    ]

    LABEL_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {LABEL_MD}")


# =========================================================================== #
#  데이터셋 분리 저장 (Task 2-2)
# =========================================================================== #

def save_clustered_datasets(df_raw: pd.DataFrame, households: list[str],
                             labels: np.ndarray, algo: str,
                             resolution: str) -> None:
    """
    labels 배열에 따라 세대 컬럼을 분리하여
    ~/data/KIER_4_Clustered/seasonal_naive/{algo}/KIER_USAGE_ELEC_INST_K{n}_{res}.csv
    로 저장.

    노이즈(-1) 세대는 별도 파일(Knoise_{res}.csv)로 저장.
    """
    algo_dir = OUT_CLUS / algo
    algo_dir.mkdir(parents=True, exist_ok=True)

    uniq_labels = sorted(set(labels))
    for cl_id in uniq_labels:
        hh_in = [h for h, lbl in zip(households, labels) if lbl == cl_id]
        if not hh_in:
            continue

        # 선택된 세대만 추출 (존재하는 컬럼만)
        valid_hh = [h for h in hh_in if h in df_raw.columns]
        if not valid_hh:
            continue

        cl_df = df_raw[valid_hh].copy()

        fname = (f"KIER_USAGE_ELEC_INST_Knoise_{resolution}.csv"
                 if cl_id == -1
                 else f"KIER_USAGE_ELEC_INST_K{cl_id}_{resolution}.csv")
        out_path = algo_dir / fname
        cl_df.to_csv(out_path)
        print(f"    [{algo}/{resolution}] K{cl_id}: {len(valid_hh)}세대 → {out_path.name}")


# =========================================================================== #
#  메인
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="5가지 군집화 알고리즘 세대 레이블 부여 + 데이터 분리")
    parser.add_argument(
        "--resolutions", nargs="+",
        choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
        metavar="RES", help="시간 해상도 (기본: 전체)")
    parser.add_argument(
        "--algos", nargs="+",
        choices=ALL_ALGOS, default=ALL_ALGOS,
        metavar="ALG", help=f"알고리즘 (기본: {ALL_ALGOS})")
    args = parser.parse_args()

    import time
    t0 = time.time()

    # 최적 K 로드
    if OPT_JSON.exists():
        optimal_k_map = json.loads(OPT_JSON.read_text(encoding="utf-8"))
        print(f"최적 K 로드: {OPT_JSON.name}")
        for res, info in optimal_k_map.items():
            print(f"  {res}: K={info['optimal_k']}")
    else:
        print(f"⚠ optimal_k.json 없음 — 기본 K=3 사용")
        print(f"  먼저 `python core/data_08_optimal_k.py` 실행을 권장합니다.")
        optimal_k_map = {res: {"optimal_k": 3} for res in DATASETS}

    label_table = {}   # {res: {algo: labels}}
    sil_table   = {}   # {res: {algo: sil_score}}
    households  = None

    for res in args.resolutions:
        csv_path = DATA_DIR / DATASETS[res]
        print(f"\n{'='*60}")
        print(f"[{res}]")

        df = pd.read_csv(csv_path, parse_dates=["METER_DATE"],
                         index_col="METER_DATE")

        if households is None:
            households = list(df.columns)

        k = optimal_k_map.get(res, {}).get("optimal_k", 3)
        print(f"  로드: {df.shape}  |  최적 K={k}")

        X = build_feature_matrix(df, res)
        print(f"  특징 행렬: {X.shape}")

        label_table[res] = {}
        sil_table[res]   = {}

        for algo in args.algos:
            print(f"\n  [{ALGO_LABEL[algo]}]")
            fn  = ALGO_FN[algo]

            # K가 필요한 알고리즘만 k 전달, MeanShift/DBSCAN은 k_hint로 전달
            if algo in ("kmeans", "l1l2", "gmm"):
                labels = fn(X, k)
            else:
                labels = fn(X, k)   # meanshift/dbscan도 k_hint 인수 받음

            n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
            noise  = (labels == -1).sum()
            sil    = safe_sil(X, labels)
            sil_s  = f"{sil:.4f}" if not np.isnan(sil) else "NaN"

            print(f"  군집 수: {n_cl}  노이즈: {noise}세대  Sil: {sil_s}")

            label_table[res][algo] = labels
            sil_table[res][algo]   = sil

            # 데이터셋 분리 저장
            save_clustered_datasets(df, households, labels, algo, res)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"전체 소요: {elapsed:.1f}s")

    # label_cluster.md 저장
    print("\n레이블 MD 저장 중...")
    write_label_md(label_table, sil_table, households or [],
                   args.resolutions, elapsed)
    print(f"분리 데이터: {OUT_CLUS}")


if __name__ == "__main__":
    main()
