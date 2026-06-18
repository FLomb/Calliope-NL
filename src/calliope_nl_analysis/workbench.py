"""Reusable analysis-workbench presets for SPORES clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

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


@dataclass(frozen=True)
class AnalysisPreset:
    """A reproducible clustering setup for one decision variable."""

    key: str
    label: str
    variable: str
    matrix_kind: str
    pca_components: int
    kmeans_clusters: int
    dbscan_eps: float
    dbscan_min_samples: int
    hierarchical_cuts: Mapping[str, float]
    notes: str


PRESETS = {
    "flow_cap_default": AnalysisPreset(
        key="flow_cap_default",
        label="Capacity clustering defaults",
        variable="flow_cap",
        matrix_kind="capacity",
        pca_components=6,
        kmeans_clusters=4,
        dbscan_eps=0.8,
        dbscan_min_samples=12,
        hierarchical_cuts={"ward": 16.0, "complete": 5.2, "single": 3.1},
        notes=(
            "Default capacity workflow: national technology capacities, "
            "transmission removed, and lost/load import/export demand/curtailment columns dropped."
        ),
    ),
    "cost_operation_variable_default": AnalysisPreset(
        key="cost_operation_variable_default",
        label="Operation-cost time-series clustering defaults",
        variable="cost_operation_variable",
        matrix_kind="timeseries",
        pca_components=2,
        kmeans_clusters=4,
        dbscan_eps=10.0,
        dbscan_min_samples=4,
        hierarchical_cuts={"ward": 100.0, "complete": 35.0, "single": 14.0},
        notes=(
            "Default time-series workflow: national operation cost per timestep, "
            "summed over nodes/techs and divided by the 3-hour timestep resolution."
        ),
    ),
}


def get_preset(key: str) -> AnalysisPreset:
    """Return one of the named workbench presets."""

    try:
        return PRESETS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown preset {key!r}. Choose one of: {sorted(PRESETS)}") from exc


def preset_table() -> pd.DataFrame:
    """Return a compact overview of available presets."""

    rows = []
    for preset in PRESETS.values():
        rows.append(
            {
                "preset": preset.key,
                "variable": preset.variable,
                "pca_components": preset.pca_components,
                "kmeans_clusters": preset.kmeans_clusters,
                "dbscan_eps": preset.dbscan_eps,
                "dbscan_min_samples": preset.dbscan_min_samples,
                "ward_cut": preset.hierarchical_cuts["ward"],
                "complete_cut": preset.hierarchical_cuts["complete"],
                "single_cut": preset.hierarchical_cuts["single"],
                "notes": preset.notes,
            }
        )
    return pd.DataFrame(rows).set_index("preset")


def build_matrix(spore_paths: Iterable[str | Path], preset: AnalysisPreset) -> pd.DataFrame:
    """Build the feature matrix for a preset."""

    if preset.matrix_kind == "capacity":
        return capacity_matrix(
            spore_paths,
            variable=preset.variable,
            carrier="power",
            exclude_transmission=True,
            drop_techs=("lost_load", "import_power", "export_power", "demand_power", "curtailment"),
        )
    if preset.matrix_kind == "timeseries":
        return operation_cost_timeseries_matrix(
            spore_paths,
            variable=preset.variable,
            normalise_by_hours=3.0,
        )
    raise ValueError(f"Unsupported matrix kind: {preset.matrix_kind}")


def fit_all_methods(matrix: pd.DataFrame, preset: AnalysisPreset) -> dict[str, pd.Series]:
    """Fit all clustering methods configured for a preset."""

    labels: dict[str, pd.Series] = {
        "K-means": kmeans_labels(
            matrix,
            n_clusters=preset.kmeans_clusters,
            pca_components=preset.pca_components,
        )
    }
    for method in ("ward", "complete", "single"):
        labels[f"Hier. ({method.title()})"], _ = hierarchical_labels(
            matrix,
            method=method,
            distance_threshold=preset.hierarchical_cuts[method],
            pca_components=preset.pca_components,
        )
    labels["DBSCAN"] = dbscan_labels(
        matrix,
        eps=preset.dbscan_eps,
        min_samples=preset.dbscan_min_samples,
        pca_components=preset.pca_components,
    )
    return labels


def method_parameter_table(preset: AnalysisPreset) -> pd.DataFrame:
    """Return method parameters for a preset."""

    rows = [
        {
            "method": "K-means",
            "pca_components": preset.pca_components,
            "clusters": preset.kmeans_clusters,
            "epsilon": None,
            "min_samples": None,
            "cut_distance": None,
        }
    ]
    for method in ("ward", "complete", "single"):
        rows.append(
            {
                "method": f"Hier. ({method.title()})",
                "pca_components": preset.pca_components,
                "clusters": None,
                "epsilon": None,
                "min_samples": None,
                "cut_distance": preset.hierarchical_cuts[method],
            }
        )
    rows.append(
        {
            "method": "DBSCAN",
            "pca_components": preset.pca_components,
            "clusters": None,
            "epsilon": preset.dbscan_eps,
            "min_samples": preset.dbscan_min_samples,
            "cut_distance": None,
        }
    )
    return pd.DataFrame(rows).set_index("method")


def validation_metrics(matrix: pd.DataFrame, labels_by_method: Mapping[str, pd.Series], pca_components: int) -> pd.DataFrame:
    """Compute internal validation metrics on the PCA-reduced space."""

    from sklearn.decomposition import PCA
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    values, _, _ = scale_matrix(matrix)
    values = PCA(n_components=pca_components).fit_transform(values)

    rows = []
    for method, labels in labels_by_method.items():
        labels_array = labels.to_numpy()
        valid = labels_array != -1
        clean_values = values[valid]
        clean_labels = labels_array[valid]
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


def nearest_neighbor_distances(matrix: pd.DataFrame, preset: AnalysisPreset) -> pd.DataFrame:
    """Return sorted kth-nearest-neighbor distances for DBSCAN tuning."""

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    values, _, _ = scale_matrix(matrix)
    values = PCA(n_components=preset.pca_components).fit_transform(values)
    distances, _ = NearestNeighbors(n_neighbors=preset.dbscan_min_samples).fit(values).kneighbors(values)
    kth_distances = np.sort(distances[:, -1])
    return pd.DataFrame({"point_order": np.arange(1, len(kth_distances) + 1), "distance": kth_distances})


def hierarchical_linkages(matrix: pd.DataFrame, preset: AnalysisPreset) -> dict[str, object]:
    """Return linkage matrices for each hierarchical method in a preset."""

    from scipy.cluster.hierarchy import linkage
    from sklearn.decomposition import PCA

    values, _, _ = scale_matrix(matrix)
    values = PCA(n_components=preset.pca_components).fit_transform(values)
    return {method: linkage(values, method=method) for method in ("single", "complete", "ward")}


def elbow_curve(matrix: pd.DataFrame, preset: AnalysisPreset, k_values=range(1, 11)) -> pd.DataFrame:
    """Return the K-means elbow curve for a preset."""

    return kmeans_elbow(matrix, k_values=k_values, pca_components=preset.pca_components)


def selected_method_tables(matrix: pd.DataFrame, labels: pd.Series) -> dict[str, pd.DataFrame | pd.Series]:
    """Return common interpretation tables for a selected clustering output."""

    labelled = add_labels(matrix, labels, label_column="cluster")
    return {
        "labelled_matrix": labelled,
        "family_counts": cluster_family_counts(labels),
        "cluster_summary": matrix.groupby(labels).mean(),
        "representatives": representative_members(labelled, label_column="cluster"),
    }


def pca_scores(matrix: pd.DataFrame, n_components: int = 2):
    """Return a 2D PCA score table and fitted PCA object."""

    scores, pca, scaler = pca_projection(matrix, n_components=n_components, scale=True)
    return scores, pca, scaler


def export_workbench_tables(tables: Mapping[str, pd.DataFrame | pd.Series], output_dir: str | Path) -> list[Path]:
    """Write workbench tables to CSV."""

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
