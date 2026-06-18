"""Reusable clustering helpers for SPORES matrices."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd

from .features import numeric_features
from .spores import parse_spore_name


def scale_matrix(matrix: pd.DataFrame):
    """Scale numeric feature columns with scikit-learn's StandardScaler."""

    from sklearn.preprocessing import StandardScaler

    features = numeric_features(matrix)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler, features.columns


def pca_projection(matrix: pd.DataFrame, n_components: int = 2, scale: bool = True):
    """Return a PCA projection DataFrame plus fitted PCA and scaler objects."""

    from sklearn.decomposition import PCA

    if scale:
        values, scaler, _ = scale_matrix(matrix)
    else:
        values = numeric_features(matrix).to_numpy()
        scaler = None

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(values)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(projected, index=matrix.index, columns=columns), pca, scaler


def kmeans_labels(
    matrix: pd.DataFrame,
    n_clusters: int,
    pca_components: int | None = None,
    random_state: int = 42,
) -> pd.Series:
    """Cluster rows with K-means and return labels indexed like the matrix."""

    from sklearn.cluster import KMeans

    values, _, _ = scale_matrix(matrix)
    if pca_components is not None:
        from sklearn.decomposition import PCA

        values = PCA(n_components=pca_components).fit_transform(values)

    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(values)
    return pd.Series(labels, index=matrix.index, name="cluster")


def dbscan_labels(matrix: pd.DataFrame, eps: float = 0.8, min_samples: int = 12, pca_components: int | None = None) -> pd.Series:
    """Cluster rows with DBSCAN and return labels indexed like the matrix."""

    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    values, _, _ = scale_matrix(matrix)
    if pca_components is not None:
        values = PCA(n_components=pca_components).fit_transform(values)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(values)
    return pd.Series(labels, index=matrix.index, name="cluster")


def hierarchical_labels(
    matrix: pd.DataFrame,
    method: str = "ward",
    distance_threshold: float | None = None,
    n_clusters: int | None = None,
    pca_components: int | None = None,
) -> tuple[pd.Series, object]:
    """Cluster rows with scipy hierarchical linkage."""

    from scipy.cluster.hierarchy import fcluster, linkage

    if distance_threshold is None and n_clusters is None:
        raise ValueError("Set either distance_threshold or n_clusters")

    values, _, _ = scale_matrix(matrix)
    if pca_components is not None:
        from sklearn.decomposition import PCA

        values = PCA(n_components=pca_components).fit_transform(values)
    linkage_matrix = linkage(values, method=method)
    if distance_threshold is not None:
        labels = fcluster(linkage_matrix, distance_threshold, criterion="distance")
    else:
        labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    return pd.Series(labels, index=matrix.index, name="cluster"), linkage_matrix


def add_labels(matrix: pd.DataFrame, labels: pd.Series, label_column: str = "cluster") -> pd.DataFrame:
    """Return a copy of the matrix with a cluster-label column."""

    labelled = matrix.copy()
    labelled[label_column] = labels.reindex(matrix.index)
    return labelled


def kmeans_elbow(matrix: pd.DataFrame, k_values: Sequence[int] = range(1, 11), pca_components: int | None = None) -> pd.DataFrame:
    """Compute K-means inertia across candidate cluster counts."""

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    values, _, _ = scale_matrix(matrix)
    if pca_components is not None:
        values = PCA(n_components=pca_components).fit_transform(values)

    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(values)
        rows.append({"k": k, "inertia": model.inertia_})
    return pd.DataFrame(rows)


def cluster_family_counts(labels: pd.Series) -> pd.DataFrame:
    """Count filename families inside each cluster."""

    families = []
    for name in labels.index:
        families.append(parse_spore_name(f"{name}.nc")["family"])
    frame = pd.DataFrame({"cluster": labels.values, "family": families}, index=labels.index)
    return frame.groupby(["cluster", "family"]).size().unstack(fill_value=0)


def cluster_summary(labelled_matrix: pd.DataFrame, label_column: str = "cluster") -> pd.DataFrame:
    """Return per-cluster feature means."""

    features = numeric_features(labelled_matrix, exclude=(label_column,))
    return features.groupby(labelled_matrix[label_column]).mean()


def representative_members(labelled_matrix: pd.DataFrame, label_column: str = "cluster") -> pd.Series:
    """Return the row closest to the numeric centroid of each cluster."""

    features = numeric_features(labelled_matrix, exclude=(label_column,))
    representatives: dict[object, str] = {}
    for cluster, group in labelled_matrix.groupby(label_column):
        group_features = features.loc[group.index]
        centroid = group_features.mean(axis=0)
        distances = np.linalg.norm(group_features.to_numpy() - centroid.to_numpy(), axis=1)
        representatives[cluster] = str(group_features.index[int(np.argmin(distances))])
    return pd.Series(representatives, name="representative_spore")
