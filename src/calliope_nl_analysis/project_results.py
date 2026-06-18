"""Helpers for reproducing the selected project-result workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .clustering import (
    add_labels,
    cluster_family_counts,
    dbscan_labels,
    hierarchical_labels,
    kmeans_elbow,
    kmeans_labels,
    pca_projection,
    representative_members,
    scale_matrix,
)
from .features import capacity_matrix, operation_cost_timeseries_matrix

CAPACITY_TECHS = (
    "battery",
    "hydrogen_power_storage",
    "bioenergy",
    "ccgt",
    "coal",
    "nuclear",
    "solar_pv",
    "wind_offshore",
    "wind_onshore",
)


@dataclass(frozen=True)
class ClusteringSettings:
    """Clustering parameters used by a reproducible result workflow."""

    pca_components: int
    kmeans_clusters: int
    dbscan_eps: float
    dbscan_min_samples: int
    hierarchical_cuts: Mapping[str, float]
    final_method: str = "ward"


CAPACITY_SETTINGS = ClusteringSettings(
    pca_components=6,
    kmeans_clusters=4,
    dbscan_eps=0.8,
    dbscan_min_samples=12,
    hierarchical_cuts={"ward": 16.0, "complete": 5.2, "single": 3.1},
)

TIMESERIES_SETTINGS = ClusteringSettings(
    pca_components=2,
    kmeans_clusters=4,
    dbscan_eps=10.0,
    dbscan_min_samples=4,
    hierarchical_cuts={"ward": 100.0, "complete": 35.0, "single": 14.0},
)


def project_capacity_matrix(spore_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Build the 9-feature national capacity matrix used for project results."""

    matrix = capacity_matrix(
        spore_paths,
        carrier="power",
        exclude_transmission=True,
        drop_techs=(),
    )
    missing = [tech for tech in CAPACITY_TECHS if tech not in matrix.columns]
    if missing:
        raise KeyError(f"Capacity matrix is missing expected project technologies: {missing}")
    return matrix.loc[:, list(CAPACITY_TECHS)]


def project_timeseries_matrix(spore_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Build the timestep operation-cost matrix used for project results."""

    return operation_cost_timeseries_matrix(
        spore_paths,
        variable="cost_operation_variable",
        normalise_by_hours=3.0,
    )


def fit_project_suite(matrix: pd.DataFrame, settings: ClusteringSettings) -> dict[str, pd.Series]:
    """Fit the algorithm suite used by the project-result workflow."""

    labels: dict[str, pd.Series] = {
        "K-means": kmeans_labels(
            matrix,
            n_clusters=settings.kmeans_clusters,
            pca_components=settings.pca_components,
        ),
    }
    for method in ("ward", "complete", "single"):
        cut = settings.hierarchical_cuts[method]
        labels[f"Hier. ({method.title()})"], _ = hierarchical_labels(
            matrix,
            method=method,
            distance_threshold=cut,
            pca_components=settings.pca_components,
        )
    labels["DBSCAN"] = dbscan_labels(
        matrix,
        eps=settings.dbscan_eps,
        min_samples=settings.dbscan_min_samples,
        pca_components=settings.pca_components,
    )
    return labels


def final_ward_labels(matrix: pd.DataFrame, settings: ClusteringSettings) -> pd.Series:
    """Return Ward labels with the configured cut distance for a feature matrix."""

    labels, _ = hierarchical_labels(
        matrix,
        method="ward",
        distance_threshold=settings.hierarchical_cuts["ward"],
        pca_components=settings.pca_components,
    )
    return labels


def validation_metrics(
    matrix: pd.DataFrame,
    labels_by_method: Mapping[str, pd.Series],
    pca_components: int | None = None,
) -> pd.DataFrame:
    """Compute internal clustering indices for each label set."""

    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    values, _, _ = scale_matrix(matrix)
    if pca_components is not None:
        from sklearn.decomposition import PCA

        values = PCA(n_components=pca_components).fit_transform(values)
    rows = []
    for method, labels in labels_by_method.items():
        labels_array = labels.to_numpy()
        valid = labels_array != -1
        clean_labels = labels_array[valid]
        clean_values = values[valid]
        n_clusters = len(set(clean_labels))
        row = {
            "method": method,
            "clusters": n_clusters,
            "noise_points": int((labels_array == -1).sum()),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }
        if n_clusters >= 2 and len(clean_labels) > n_clusters:
            row["silhouette"] = silhouette_score(clean_values, clean_labels)
            row["davies_bouldin"] = davies_bouldin_score(clean_values, clean_labels)
            row["calinski_harabasz"] = calinski_harabasz_score(clean_values, clean_labels)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def elbow_curve(matrix: pd.DataFrame, settings: ClusteringSettings, k_values=range(1, 11)) -> pd.DataFrame:
    """Return the K-means elbow curve using the configured PCA settings."""

    return kmeans_elbow(matrix, k_values=k_values, pca_components=settings.pca_components)


def nearest_neighbor_distances(
    matrix: pd.DataFrame,
    settings: ClusteringSettings,
) -> pd.DataFrame:
    """Return sorted kth-nearest-neighbor distances for DBSCAN tuning."""

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    values, _, _ = scale_matrix(matrix)
    values = PCA(n_components=settings.pca_components).fit_transform(values)
    neighbors = NearestNeighbors(n_neighbors=settings.dbscan_min_samples)
    distances, _ = neighbors.fit(values).kneighbors(values)
    kth_distances = np.sort(distances[:, -1])
    return pd.DataFrame(
        {
            "point_order": np.arange(1, len(kth_distances) + 1),
            "distance": kth_distances,
        }
    )


def hierarchical_linkages(matrix: pd.DataFrame, settings: ClusteringSettings) -> dict[str, object]:
    """Return scipy linkage matrices for configured hierarchical methods."""

    from scipy.cluster.hierarchy import linkage
    from sklearn.decomposition import PCA

    values, _, _ = scale_matrix(matrix)
    values = PCA(n_components=settings.pca_components).fit_transform(values)
    return {
        method: linkage(values, method=method)
        for method in ("single", "complete", "ward")
    }


def parameter_table(settings: ClusteringSettings) -> pd.DataFrame:
    """Return a compact table of clustering parameters."""

    rows = [
        {
            "method": "K-means",
            "pca_components": settings.pca_components,
            "clusters": settings.kmeans_clusters,
            "epsilon": None,
            "min_samples": None,
            "cut_distance": None,
        },
    ]
    for method in ("ward", "complete", "single"):
        cut = settings.hierarchical_cuts[method]
        rows.append(
            {
                "method": f"Hier. ({method.title()})",
                "pca_components": settings.pca_components,
                "clusters": None,
                "epsilon": None,
                "min_samples": None,
                "cut_distance": cut,
            }
        )
    rows.append(
        {
            "method": "DBSCAN",
            "pca_components": settings.pca_components,
            "clusters": None,
            "epsilon": settings.dbscan_eps,
            "min_samples": settings.dbscan_min_samples,
            "cut_distance": None,
        }
    )
    return pd.DataFrame(rows).set_index("method")


def labelled_ward_matrix(matrix: pd.DataFrame, settings: ClusteringSettings) -> pd.DataFrame:
    """Return a matrix with Ward labels."""

    return add_labels(matrix, final_ward_labels(matrix, settings), label_column="cluster")


def ward_result_tables(matrix: pd.DataFrame, settings: ClusteringSettings) -> dict[str, pd.DataFrame | pd.Series]:
    """Return core tables for the final Ward interpretation."""

    labelled = labelled_ward_matrix(matrix, settings)
    labels = labelled["cluster"]
    return {
        "labelled_matrix": labelled,
        "family_counts": cluster_family_counts(labels),
        "centroids": matrix.groupby(labels).mean(),
        "representatives": representative_members(labelled, label_column="cluster"),
    }


def timeseries_statistics(matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarise each SPORE time series with descriptive statistics."""

    return pd.DataFrame(
        {
            "min": matrix.min(axis=1),
            "max": matrix.max(axis=1),
            "mean": matrix.mean(axis=1),
            "variance": matrix.var(axis=1),
            "range": matrix.max(axis=1) - matrix.min(axis=1),
        }
    )


def centroid_and_representative_statistics(
    matrix: pd.DataFrame,
    labels: pd.Series,
    representatives: pd.Series,
) -> pd.DataFrame:
    """Compare Ward centroids with their nearest real SPORES."""

    centroids = matrix.groupby(labels).mean()
    centroid_stats = timeseries_statistics(centroids)
    representative_stats = timeseries_statistics(matrix.loc[representatives.values])

    rows = []
    for cluster, representative in representatives.items():
        rows.append({"cluster": cluster, "configuration": f"Cluster {cluster} (centroid)", **centroid_stats.loc[cluster].to_dict()})
        rows.append({"cluster": cluster, "configuration": representative, **representative_stats.loc[representative].to_dict()})
    return pd.DataFrame(rows).set_index(["cluster", "configuration"])


def pca_scores_and_loadings(matrix: pd.DataFrame, n_components: int = 2):
    """Return PCA scores, fitted PCA, scaler, and loading matrix."""

    scores, pca, scaler = pca_projection(matrix, n_components=n_components, scale=True)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=matrix.columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return scores, loadings, pca, scaler


def percentile_selection(
    scores: pd.DataFrame,
    pc1_quantile: float = 0.2,
    pc2_quantile: float = 0.9,
) -> pd.DataFrame:
    """Select low-PC1 and high-PC2 candidates for decision-maker review."""

    pc1_cutoff = scores["PC1"].quantile(pc1_quantile)
    pc2_cutoff = scores["PC2"].quantile(pc2_quantile)
    selected = scores[(scores["PC1"] <= pc1_cutoff) & (scores["PC2"] >= pc2_cutoff)].copy()
    selected["pc1_cutoff"] = pc1_cutoff
    selected["pc2_cutoff"] = pc2_cutoff
    return selected


def pareto_candidates(
    capacity_matrix_: pd.DataFrame,
    labels: pd.Series,
    technologies: Sequence[str] = ("ccgt", "coal", "bioenergy"),
    quantile: float = 0.25,
) -> pd.DataFrame:
    """Return low-dispatchable-capacity candidates for decision-maker review."""

    missing = [tech for tech in technologies if tech not in capacity_matrix_.columns]
    if missing:
        raise KeyError(f"Missing Pareto technologies: {missing}")

    frame = capacity_matrix_.loc[:, list(technologies)].copy()
    frame["cluster"] = labels.reindex(frame.index)
    frame["dispatchable_total"] = frame.loc[:, list(technologies)].sum(axis=1)
    cutoff = frame["dispatchable_total"].quantile(quantile)
    frame["selected_low_dispatchable"] = frame["dispatchable_total"] <= cutoff
    return frame.sort_values(["selected_low_dispatchable", "dispatchable_total"], ascending=[False, True])


def export_tables(tables: Mapping[str, pd.DataFrame | pd.Series], output_dir: str | Path) -> list[Path]:
    """Write a set of result tables as CSV files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in tables.items():
        path = output / f"{name}.csv"
        if isinstance(table, pd.Series):
            table.to_frame().to_csv(path)
        else:
            table.to_csv(path)
        written.append(path)
    return written
