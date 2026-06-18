"""Feature matrix builders for SPORES result clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def capacity_matrix(
    spore_paths: Iterable[str | Path],
    variable: str = "flow_cap",
    carrier: str | None = "power",
    exclude_transmission: bool = True,
    drop_techs: Sequence[str] = ("lost_load",),
    fill_value: float = 0.0,
):
    """Return a SPORES-by-technology capacity matrix from NetCDF files."""

    import pandas as pd
    import xarray as xr

    rows = []
    for path in spore_paths:
        path = Path(path)
        with xr.open_dataset(path) as dataset:
            if variable not in dataset:
                raise KeyError(f"{variable!r} not found in {path}")
            data = dataset[variable]
            if carrier is not None and "carriers" in data.coords:
                data = data.sel(carriers=carrier)
            frame = data.to_dataframe(name=variable).reset_index()

        if "techs" not in frame:
            raise KeyError(f"{variable!r} in {path} does not expose a 'techs' coordinate")

        if exclude_transmission:
            frame = frame[~frame["techs"].astype(str).str.contains("_to_", regex=False)]
        frame = frame.dropna(subset=[variable])

        series = frame.groupby("techs", observed=True)[variable].sum()
        if drop_techs:
            series = series.drop(labels=list(drop_techs), errors="ignore")
        series.name = path.stem
        rows.append(series)

    matrix = pd.concat(rows, axis=1).T
    return matrix.fillna(fill_value)


def operation_cost_timeseries_matrix(
    spore_paths: Iterable[str | Path],
    variable: str = "cost_operation_variable",
    normalise_by_hours: float | None = 3.0,
    fill_value: float = 0.0,
):
    """Return a SPORES-by-timestep operation-cost matrix from NetCDF files."""

    import pandas as pd
    import xarray as xr

    rows = []
    for path in spore_paths:
        path = Path(path)
        with xr.open_dataset(path) as dataset:
            if variable not in dataset:
                raise KeyError(f"{variable!r} not found in {path}")
            frame = dataset[variable].to_dataframe(name=variable).reset_index()

        if "timesteps" not in frame:
            raise KeyError(f"{variable!r} in {path} does not expose a 'timesteps' coordinate")

        frame = frame.dropna(subset=[variable])
        values = frame.groupby("timesteps", observed=True)[variable].sum()
        if normalise_by_hours is not None:
            values = values / normalise_by_hours
        values.name = path.stem
        rows.append(values)

    matrix = pd.concat(rows, axis=1).T
    return matrix.fillna(fill_value)


def numeric_features(frame, exclude: Sequence[str] = ("cluster", "Cluster", "Cluster_DBSCAN", "Cluster_Hierarchical")):
    """Return only numeric feature columns, excluding common label columns."""

    return frame.drop(columns=list(exclude), errors="ignore").select_dtypes(include="number")
