import calliope
import xarray as xr
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = 'iframe'  # or 'iframe' or 'plotly_mimetype'
pio.templates.default = "plotly"
import os
os.environ["MAPBOX_ACCESS_TOKEN"] = ""
import re
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#%%

def plot_dispatch(model_path, plot_export_path='', colors={}):

    try:
        model = calliope.read_netcdf(model_path)
        df_electricity = (
            (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
            .sel(carriers="power")
            .sum("nodes")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow in/out (MW)")
            .reset_index()
        )
    except:
        model = xr.read_netcdf(model_path)
        df_electricity = (
                (model.flow_out.fillna(0) - model.flow_in.fillna(0))
                .sel(carriers="power")
                .sum("nodes")
                .to_series()
                .where(lambda x: x != 0)
                .dropna()
                .to_frame("Flow in/out (MW)")
                .reset_index()
            )
    time_resolution = model.inputs.timestep_resolution.to_series().max()


    df_electricity_demand = df_electricity[df_electricity.techs == "demand_power"]
    df_electricity_demand.loc[:,"Flow in/out (MW)"] = df_electricity_demand["Flow in/out (MW)"]/time_resolution
    df_electricity_other = df_electricity[df_electricity.techs != "demand_power"]
    df_electricity_other.loc[:,"Flow in/out (MW)"] = df_electricity_other["Flow in/out (MW)"]/time_resolution

    fig = px.bar(
        df_electricity_other,
        x="timesteps",
        y="Flow in/out (MW)",
        color="techs",
        color_discrete_map=colors,
    )
    fig.add_scatter(
        x=df_electricity_demand.timesteps,
        y=-1 * df_electricity_demand["Flow in/out (MW)"],
        marker_color="black",
        name="demand",
    )
    fig.show()
    if plot_export_path == '':
        pass
    else:
        fig.write_html(plot_export_path)

def plot_capacity(model_path, exclude_transmission=False, plot_export_path='', colors={}):

    try:
        model = calliope.read_netcdf(model_path)
        df_capacity = (
            model.results.flow_cap.where(model.results.techs != "demand_power")
            .sel(carriers="power")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow capacity (MW)")
            .reset_index()
        )
    except:
        model = xr.open_dataset(model_path, engine="netcdf4")
        df_capacity = (
            model.flow_cap.where(model.techs != "demand_power")
            .sel(carriers="power")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow capacity (MW)")
            .reset_index()
        )



    if exclude_transmission == True:
        df_capacity = df_capacity[~df_capacity['techs'].str.contains('_to_')]
    else:
        pass

    fig = px.bar(
        df_capacity,
        x="nodes",
        y="Flow capacity (MW)",
        color="techs",
        color_discrete_map=colors,
    )
    fig.show()
    if plot_export_path == '':
        pass
    else:
        fig.write_html(plot_export_path)


#%%
def plot_network(model_path, plot_export_path=''):
    
    model = calliope.read_netcdf(model_path)

    coordinates = model.inputs[["latitude", "longitude"]].to_dataframe()

    grid_capacity = (
        model.results.flow_cap.where(model.inputs.base_tech == "transmission")
        .sel(carriers="power")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Flow capacity (MW)")
        .reset_index()
    ).set_index("techs")

    grid_capacity["from"] = [re.split("_to_",x)[0] for x in grid_capacity.index]
    grid_capacity["to"] = [re.split("_to_",x)[-1] for x in grid_capacity.index]

    # Create map figure
    fig = go.Figure()

    max_grid_cap = grid_capacity["Flow capacity (MW)"].max()
    magnifier = 6
    n_connections = len(grid_capacity.index)

    # Plot lines
    for conn in grid_capacity.index:
        source = grid_capacity["from"].loc[(grid_capacity.index==conn) & (grid_capacity["nodes"]==grid_capacity["from"])]
        target = grid_capacity["to"].loc[(grid_capacity.index==conn) & (grid_capacity["nodes"]==grid_capacity["from"])] 

        line_cap = grid_capacity["Flow capacity (MW)"].loc[
            (grid_capacity.index==conn) & (grid_capacity["nodes"]==grid_capacity["from"])
            ].values[0]
        
        fig.add_trace(go.Scattermapbox(
            mode = "lines+text",
            lon = [coordinates["longitude"].loc[source].values[0], coordinates["longitude"].loc[target].values[0]],
            lat = [coordinates["latitude"].loc[source].values[0], coordinates["latitude"].loc[target].values[0]],
            line = dict(width = (line_cap / max_grid_cap * magnifier),
                        color = "crimson"),
            visible=True,
            text = "{}".format(round(line_cap)),
            hoverinfo='text',
            # legendgroup="line_caps",  # this can be any string, not just "group"
            # name="Line capacity",
            showlegend=False
        ))

    # Add markers for points
    for coord in coordinates.index:
        fig.add_trace(go.Scattermapbox(
            mode = "markers+text",
            lon = [coordinates["longitude"].loc[coord]],
            lat = [coordinates["latitude"].loc[coord]],
            marker = dict(size = 10, color = "grey"),
            text = coord,
            textposition = "top right",
            name = coord,
            showlegend=False
        ))

    fig.update_layout(
        mapbox = dict(
            style = "carto-positron",
            zoom = 6
        ),
        mapbox_center_lat=coordinates.latitude.mean(),
        mapbox_center_lon=coordinates.longitude.mean(),
        margin = dict(l=0, r=0, t=0, b=0),
    )

    fig.show()
    if plot_export_path == '':
        pass
    else:
        fig.write_html(plot_export_path)
