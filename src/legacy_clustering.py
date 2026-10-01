"""Legacy-compatible clustering helpers.

The functions here mirror the previous project's KMeans-oriented clustering
analysis while operating on the current project's preprocessed wide CSV files.
They do not modify or depend on the previous repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    rand_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
PCA_MAX_COMPONENTS = 30


@dataclass(frozen=True)
class ClusterMetrics:
    k: int
    inertia: float
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    dunn: float
    cluster_sizes: tuple[int, ...]


def house_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "METER_DATE"]


def load_wide_usage(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["METER_DATE"]).sort_values("METER_DATE")


def build_feature_matrix(df: pd.DataFrame, resolution: str, pca_max: int = PCA_MAX_COMPONENTS) -> pd.DataFrame:
    """Build the household x feature matrix used by the legacy clustering flow."""
    work = df.copy()
    work["METER_DATE"] = pd.to_datetime(work["METER_DATE"])
    work = work.set_index("METER_DATE")
    cols = house_columns(work.reset_index())

    if resolution.upper() == "10MIN":
        slot = work.index.hour * 6 + work.index.minute // 10
        x = work[cols].groupby(slot).mean().T
    elif resolution.upper() == "1H":
        x = work[cols].groupby(work.index.hour).mean().T
    else:
        x = work[cols].T

    values = np.nan_to_num(x.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = StandardScaler().fit_transform(values)
    n_comp = min(pca_max, values.shape[1], values.shape[0] - 1)
    if n_comp > 1 and values.shape[1] > pca_max:
        values = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit_transform(values)
    return pd.DataFrame(values, index=x.index)


def dunn_index(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    max_intra = 0.0
    min_inter = float("inf")
    for lab in unique:
        cluster = x[labels == lab]
        if len(cluster) > 1:
            dist = np.sqrt(((cluster[:, None, :] - cluster[None, :, :]) ** 2).sum(axis=2))
            max_intra = max(max_intra, float(dist.max()))
    for i, lab_i in enumerate(unique):
        xi = x[labels == lab_i]
        for lab_j in unique[i + 1 :]:
            xj = x[labels == lab_j]
            dist = np.sqrt(((xi[:, None, :] - xj[None, :, :]) ** 2).sum(axis=2))
            min_inter = min(min_inter, float(dist.min()))
    if max_intra == 0.0:
        return float("inf")
    return min_inter / max_intra


def evaluate_kmeans_range(
    features: pd.DataFrame,
    k_values: Iterable[int],
    n_init: int = 1,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluate KMeans using the legacy metrics: inertia, silhouette, CHI, DBI, Dunn."""
    rows: list[dict[str, object]] = []
    x = features.to_numpy(dtype=float)
    for k in k_values:
        km = KMeans(n_clusters=int(k), init="k-means++", max_iter=300, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(x)
        sizes = tuple(int((labels == lab).sum()) for lab in range(int(k)))
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(x, labels, sample_size=min(1000, len(labels)))),
                "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
                "davies_bouldin": float(davies_bouldin_score(x, labels)),
                "dunn": float(dunn_index(x, labels)),
                "cluster_sizes": "|".join(str(v) for v in sizes),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans_labels(
    features: pd.DataFrame,
    k: int,
    n_init: int = 1,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(features.to_numpy(dtype=float))
    return pd.DataFrame({"household": features.index.to_list(), "cluster": labels})


def supervised_cluster_scores(features: pd.DataFrame, labels: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return the legacy supervised cluster score group when true labels exist."""
    x = features.to_numpy(dtype=float)
    return {
        "silhouette": float(silhouette_score(x, labels, sample_size=min(1000, len(labels)))),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "dunn": float(dunn_index(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "homogeneity": float(homogeneity_score(y, labels)),
        "completeness": float(completeness_score(y, labels)),
        "v_measure": float(v_measure_score(y, labels)),
        "rand": float(rand_score(y, labels)),
        "adjusted_rand": float(adjusted_rand_score(y, labels)),
    }


def load_cluster_groups(cluster_dir: Path, resolution: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(cluster_dir.glob(f"*_{resolution}.csv")):
        stem = path.stem
        group_id = stem.split("_")[-2] if "_" in stem else stem
        cols = [c for c in pd.read_csv(path, nrows=0).columns if c != "METER_DATE"]
        groups[group_id] = cols
    if not groups:
        raise FileNotFoundError(f"No cluster CSV files found in {cluster_dir} for resolution={resolution}")
    return groups
