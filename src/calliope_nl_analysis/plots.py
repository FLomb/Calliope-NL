"""Small plotting helpers for clean SPORES analysis notebooks."""

from __future__ import annotations

from typing import Sequence


def _cluster_sort_key(value):
    try:
        return (float(value) == -1, float(value))
    except (TypeError, ValueError):
        return (str(value) == "-1", str(value))


def _cluster_label(value) -> str:
    return "Noise (-1)" if str(value) == "-1" else f"Cluster {value}"


def _cluster_color_map(values):
    import matplotlib.pyplot as plt

    ordered = sorted(values, key=_cluster_sort_key)
    palette = plt.get_cmap("tab10")
    colors = {}
    next_color = 0
    for value in ordered:
        if str(value) == "-1":
            colors[value] = "#7f7f7f"
        else:
            colors[value] = palette(next_color % 10)
            next_color += 1
    return colors


def _as_datetime_index(index):
    import pandas as pd

    converted = pd.to_datetime(index, errors="coerce")
    if converted.notna().all():
        return converted
    return index


def _display_technology(value) -> str:
    labels = {
        "battery": "Battery",
        "bioenergy": "Bioenergy",
        "ccgt": "CCGT",
        "coal": "Coal",
        "hydrogen_power_storage": "H2 storage",
        "nuclear": "Nuclear",
        "solar_pv": "Solar PV",
        "wind_offshore": "Offshore wind",
        "wind_onshore": "Onshore wind",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def plot_validation_metrics(metrics, title: str = "Internal validation indices"):
    """Plot Silhouette, Davies-Bouldin, and Calinski-Harabasz as compact bars."""

    import matplotlib.pyplot as plt

    columns = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, column in zip(axes, columns):
        values = metrics[column].sort_values(ascending=(column == "davies_bouldin"))
        values.plot(kind="bar", ax=axis, color="#4c78a8", width=0.72)
        axis.set_title(column.replace("_", " ").title())
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.3)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_elbow_curve(elbow, title: str = "K-means elbow curve"):
    """Plot K-means inertia by number of clusters."""

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(elbow["k"], elbow["inertia"], marker="o")
    axis.set_title(title)
    axis.set_xlabel("Number of clusters")
    axis.set_ylabel("Inertia")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axis


def plot_nearest_neighbor_curve(distances, eps: float | None = None, title: str = "Nearest-neighbor distances"):
    """Plot sorted kth-nearest-neighbor distances for DBSCAN tuning."""

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(distances["point_order"], distances["distance"])
    if eps is not None:
        axis.axhline(eps, color="crimson", linestyle="--", linewidth=1.2, label=f"eps = {eps:g}")
        axis.legend()
    axis.set_title(title)
    axis.set_xlabel("Sorted SPORES")
    axis.set_ylabel("kth-neighbor distance")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axis


def plot_dendrograms(linkages, cut_distances=None, title: str = "Hierarchical dendrograms"):
    """Plot dendrograms for a dictionary of scipy linkage matrices."""

    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    methods = list(linkages)
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    cut_distances = cut_distances or {}
    for axis, method in zip(axes, methods):
        dendrogram(linkages[method], no_labels=True, ax=axis)
        if method in cut_distances:
            axis.axhline(cut_distances[method], color="crimson", linestyle="--", linewidth=1.2)
        axis.set_title(method.title())
        axis.set_xlabel("SPORES")
        axis.set_ylabel("Distance")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_family_heatmap(counts, title: str = "SPORES allocation in clusters"):
    """Plot a cluster-by-family allocation heatmap."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axis = plt.subplots(figsize=(max(11, 0.62 * len(counts.columns)), max(4, 0.48 * len(counts.index))))
    sns.heatmap(
        counts,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.4,
        linecolor="white",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("SPORES family")
    axis.set_ylabel("Cluster")
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig, axis


def plot_capacity_centroids(
    centroids,
    title: str = "Ward capacity centroids",
    style: str = "heatmap",
    normalize: bool = True,
    annotate_values: bool = True,
):
    """Plot cluster centroids for the capacity feature matrix.

    The default heatmap encodes each technology relative to its own maximum,
    which keeps small but meaningful technologies visible next to solar/wind.
    Set ``style="bar"`` for the original grouped-bar view.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    ordered = centroids.loc[sorted(centroids.index, key=_cluster_sort_key)]
    if style == "bar":
        color_map = _cluster_color_map(ordered.index)
        colors = [color_map[value] for value in ordered.index]
        axis = ordered.T.plot(kind="bar", figsize=(12, 5.5), color=colors, width=0.78)
        axis.set_title(title)
        axis.set_xlabel("Technology")
        axis.set_ylabel("Installed capacity")
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=35)
        axis.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        axis.figure.tight_layout()
        return axis.figure, axis

    heatmap_values = ordered.copy()
    if normalize:
        column_max = heatmap_values.max(axis=0).replace(0, 1)
        heatmap_values = heatmap_values / column_max

    annotations = None
    if annotate_values:
        annotations = ordered.round(0).astype(int).astype(str)
        annotations = annotations.mask(ordered.abs() < 0.5, "")

    fig, axis = plt.subplots(figsize=(12.5, max(3.8, 0.52 * len(ordered.index) + 1.7)))
    x_tick_labels = [_display_technology(column) for column in heatmap_values.columns]
    sns.heatmap(
        heatmap_values,
        annot=annotations,
        fmt="",
        cmap="YlGnBu",
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "Share of technology maximum" if normalize else "Installed capacity"},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Technology")
    axis.set_ylabel("Cluster")
    axis.set_xticklabels(x_tick_labels, rotation=30, ha="right")
    axis.set_yticklabels([_cluster_label(value) for value in ordered.index], rotation=0)
    fig.tight_layout()
    return fig, axis


def plot_timeseries_centroids(
    centroids,
    title: str = "Ward operation-cost centroids",
    smooth_window: int | None = 56,
    show_raw: bool = False,
    include_noise: bool = False,
):
    """Plot cluster centroids for time-series operation costs.

    The default smoothing window is 56 timesteps, i.e. roughly one week for
    3-hour Calliope timesteps. Set ``show_raw=True`` for faint raw traces.
    """

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plot_frame = centroids.T.copy()
    plot_frame.index = _as_datetime_index(plot_frame.index)
    if hasattr(plot_frame.index, "is_monotonic_increasing") and not plot_frame.index.is_monotonic_increasing:
        plot_frame = plot_frame.sort_index()

    hidden_noise = False
    if not include_noise:
        noise_columns = [column for column in plot_frame.columns if str(column) == "-1"]
        hidden_noise = bool(noise_columns)
        plot_frame = plot_frame.drop(columns=noise_columns, errors="ignore")

    if smooth_window and smooth_window > 1:
        smooth_frame = plot_frame.rolling(window=smooth_window, min_periods=max(2, smooth_window // 4), center=True).mean()
        smooth_frame = smooth_frame.bfill().ffill()
    else:
        smooth_frame = plot_frame

    ordered_columns = sorted(smooth_frame.columns, key=_cluster_sort_key)
    color_map = _cluster_color_map(ordered_columns)

    fig, axis = plt.subplots(figsize=(13.5, 5.2))
    if show_raw:
        for column in ordered_columns:
            axis.plot(
                plot_frame.index,
                plot_frame[column],
                color=color_map[column],
                linewidth=0.45,
                alpha=0.12,
            )

    for column in ordered_columns:
        is_noise = str(column) == "-1"
        axis.plot(
            smooth_frame.index,
            smooth_frame[column],
            label=_cluster_label(column),
            color=color_map[column],
            linewidth=1.4 if is_noise else 2.2,
            linestyle="--" if is_noise else "-",
            alpha=0.75 if is_noise else 0.95,
        )

    axis.set_title(title)
    axis.set_xlabel("Timestep")
    ylabel = "Variable operation cost"
    if smooth_window and smooth_window > 1:
        ylabel += f" ({smooth_window}-step rolling mean)"
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    if hasattr(smooth_frame.index, "to_pydatetime"):
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axis.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    if hidden_noise:
        axis.text(
            0.995,
            0.02,
            "DBSCAN noise cluster (-1) hidden",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )
    axis.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis


def plot_pca_scores(scores, labels, title: str = "PCA scores", percentiles: Sequence[float] = (0.2, 0.8, 0.9)):
    """Plot the first two PCA score dimensions with cluster labels."""

    import matplotlib.pyplot as plt

    plot_frame = scores.copy()
    plot_frame["cluster"] = labels.reindex(scores.index)
    ordered_clusters = sorted(plot_frame["cluster"].dropna().unique(), key=_cluster_sort_key)
    color_map = _cluster_color_map(ordered_clusters)
    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    for cluster in ordered_clusters:
        group = plot_frame[plot_frame["cluster"] == cluster]
        is_noise = str(cluster) == "-1"
        axis.scatter(
            group["PC1"],
            group["PC2"],
            label=_cluster_label(cluster),
            s=35 if is_noise else 58,
            color=color_map[cluster],
            alpha=0.35 if is_noise else 0.82,
            edgecolor="none" if is_noise else "white",
            linewidth=0.7,
        )

    for quantile in percentiles:
        axis.axvline(scores["PC1"].quantile(quantile), color="grey", linewidth=0.8, linestyle="--", alpha=0.45)
        axis.axhline(scores["PC2"].quantile(quantile), color="grey", linewidth=0.8, linestyle="--", alpha=0.45)

    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.grid(alpha=0.3)
    axis.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig, axis


def plot_pca_loadings(loadings, columns: Sequence[str] = ("PC1", "PC2"), title: str = "PCA loadings"):
    """Plot selected PCA loading columns."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.8 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns):
        loadings[column].plot(ax=axis)
        axis.set_title(column)
        axis.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_pareto_3d(frame, technologies: Sequence[str] = ("ccgt", "coal", "bioenergy"), title: str = "Dispatchable-capacity candidates"):
    """Plot a 3D scatter for CCGT, coal, and bioenergy capacities."""

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    selected = frame["selected_low_dispatchable"]
    axis.scatter(
        frame.loc[~selected, technologies[0]],
        frame.loc[~selected, technologies[1]],
        frame.loc[~selected, technologies[2]],
        alpha=0.35,
        label="Other SPORES",
    )
    axis.scatter(
        frame.loc[selected, technologies[0]],
        frame.loc[selected, technologies[1]],
        frame.loc[selected, technologies[2]],
        alpha=0.9,
        label="Low dispatchable candidates",
    )
    axis.set_xlabel(technologies[0])
    axis.set_ylabel(technologies[1])
    axis.set_zlabel(technologies[2])
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    return fig, axis
