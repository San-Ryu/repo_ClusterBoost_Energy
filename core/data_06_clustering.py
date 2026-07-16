"""
data_06_clustering.py
K-Means 클러스터링 공통 모듈 — Stage 3 (단일·다회차 분석, 평가지표, 시각화)

History
  2024-04-03  Created
  2024-07-24  Dunn Index 추가
  2024-08-05  코드 개선 및 Error Fix
  2026-03-30  Refactored - import 정리, 반복 패턴 간략화
  2026-05-02  Renamed from data_clustering.py → data_06_clustering.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    homogeneity_score, completeness_score, v_measure_score,
    adjusted_rand_score, silhouette_score, rand_score,
    calinski_harabasz_score, davies_bouldin_score,
)
from core.ref_cluster_eval import get_dunn_index

_COLORS = ["blue", "green", "orange", "cyan", "red",
           "black", "yellow", "peru", "purple", "slategray"]


# =========================================================================== #
#  유틸: K 범위 정규화
# =========================================================================== #

def _normalize_k_range(k_min: int, k_max: int) -> range:
    lo, hi = (int(k_max), int(k_min)) if k_min > k_max else (int(k_min), int(k_max))
    return range(lo, hi + 1)


# =========================================================================== #
#  단일회차 클러스터링
# =========================================================================== #

def get_calinski_harabasz_index(X: np.ndarray, labels: np.ndarray) -> float:
    """CHI(Calinski–Harabasz Index)를 수동 계산."""
    unique = np.unique(labels)
    K, n = len(unique), X.shape[0]
    c = X.mean(axis=0)
    num, denom = 0.0, 0.0
    for cl in unique:
        sub = X[labels == cl]
        c_k = sub.mean(axis=0)
        num += sub.shape[0] * np.sum((c_k - c) ** 2)
        denom += np.sum((sub - c_k) ** 2)
    return (num / (K - 1)) / (denom / (n - K))


def get_cluster_sizes(km, X: np.ndarray) -> list[int]:
    """KMeans 모델 적합 후 각 클러스터 크기 반환."""
    km.fit(X)
    return [int(np.sum(km.labels_ == i)) for i in range(km.n_clusters)]


def clustering_visualization(interval: str, km, X: np.ndarray) -> None:
    """2D 클러스터 산점도 시각화."""
    labels = km.labels_
    for k in range(km.n_clusters):
        data = X[labels == k]
        plt.scatter(data[:, 0], data[:, 1], c=_COLORS[k % len(_COLORS)],
                    alpha=0.8, label=f"cluster {k}")
        plt.scatter(km.cluster_centers_[k, 0], km.cluster_centers_[k, 1],
                    c="red", marker="x")
    plt.title(f"Clustering by {interval}")
    plt.legend(fontsize=12, loc="upper right")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.show()


def get_clustering_score(km, X: np.ndarray, y) -> list[float]:
    """
    클러스터링 평가지표 9종 계산·출력.
    [Silhouette, CHI, Dunn, DBI, Homogeneity, Completeness, V-measure, RI, ARI]
    """
    labels = km.labels_
    scores = [
        silhouette_score(X, labels, sample_size=1000),
        calinski_harabasz_score(X, labels),
        get_dunn_index(X, labels),
        davies_bouldin_score(X, labels),
        homogeneity_score(y, labels),
        completeness_score(y, labels),
        v_measure_score(y, labels),
        rand_score(y, labels),
        adjusted_rand_score(y, labels),
    ]
    names = ["Silhouette", "CHI", "Dunn", "DBI",
             "Homogeneity", "Completeness", "V-measure", "Rand-Index", "Adj Rand-Index"]
    for name, val in zip(names, scores):
        print(f"{name}: {val:.4f}")
    return scores


# =========================================================================== #
#  다회차 클러스터링
# =========================================================================== #

def _plot_metric_by_k(K: range, values: list, ylabel: str, title: str,
                      opt_X: int | None = None) -> None:
    """K별 지표 그래프 공통 시각화."""
    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    ax.plot(list(K), values, marker=".", markersize=5, zorder=2)
    ax.set_xticks(list(K))
    if opt_X is not None and (opt_X - min(K)) < len(values):
        plt.scatter(opt_X, values[opt_X - min(K)],
                    color="red", marker="^", label="Point", zorder=9999)
    plt.ylabel(ylabel)
    plt.xlabel("K")
    plt.title(title)
    plt.show()


def clustering_elbow_method(interval: str, k_min: int, k_max: int,
                            X: np.ndarray, opt_X: int | None = None):
    """
    Elbow Method: K별 Inertia 계산·시각화.
    Returns: (list_inertia, list_inertia_deriv)
    """
    K = _normalize_k_range(k_min, k_max)
    inertias = [KMeans(n_clusters=k).fit(X).inertia_ for k in K]
    derivs = [abs(inertias[i] - inertias[i - 1]) for i in range(1, len(inertias))]

    _plot_metric_by_k(K, inertias, "Inertia",
                      f"Inertia by number of clusters (Interval : {interval})", opt_X)
    return inertias, derivs


def clustering_CHI_method(interval: str, k_min: int, k_max: int,
                          X: np.ndarray, opt_X: int | None = None):
    """K별 CHI(Calinski–Harabasz Index) 계산·시각화."""
    K = _normalize_k_range(k_min, k_max)
    chis = []
    for k in K:
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=1).fit(X)
        chis.append(get_calinski_harabasz_index(X, km.predict(X)))

    _plot_metric_by_k(K, chis, "CHI",
                      f"Calinski-Harabasz Index by number of clusters (Interval : {interval})",
                      opt_X)
    return chis


def clustering_silhouette_method(interval: str, k_min: int, k_max: int,
                                 X: np.ndarray, opt_X: int | None = None):
    """
    K별 Silhouette Score 계산·시각화.
    Returns: (list_silhouette, list_cluster_sizes_by_K)
    """
    K = _normalize_k_range(k_min, k_max)
    sils, sizes_by_k = [], []

    for k in K:
        km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=1).fit(X)
        sils.append(silhouette_score(X, km.labels_, sample_size=1000))
        sizes_by_k.append([int(np.sum(km.labels_ == i)) for i in range(k)])

    _plot_metric_by_k(K, sils, "Silhouette Score",
                      f"Silhouette Score by number of clusters (Interval : {interval})",
                      opt_X)
    return sils, sizes_by_k


def clustering_get_cnt_by_loop(K: int, n_loop: int, X: np.ndarray):
    """K 고정, n_loop 회 반복 군집화 → 각 회차별 클러스터 크기 기록."""
    results = []
    for _ in range(n_loop):
        km = KMeans(n_clusters=K, init="k-means++", max_iter=300, n_init=1)
        results.append(get_cluster_sizes(km, X))
    return results
