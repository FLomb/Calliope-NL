import calliope
import pandas as pd
import plotly.express as px

def plot_dispatch(model_path, plot_export_path=''):

    model = calliope.read_netcdf(model_path)
    colors = model.inputs.color.to_series().to_dict()

    df_electricity = (
        (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
        .sel(carriers="power")
        .sum("nodes")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Flow in/out (MWh)")
        .reset_index()
    )
    df_electricity_demand = df_electricity[df_electricity.techs == "demand_power"]
    df_electricity_other = df_electricity[df_electricity.techs != "demand_power"]

    fig = px.bar(
        df_electricity_other,
        x="timesteps",
        y="Flow in/out (MWh)",
        color="techs",
        color_discrete_map=colors,
    )
    fig.add_scatter(
        x=df_electricity_demand.timesteps,
        y=-1 * df_electricity_demand["Flow in/out (MWh)"],
        marker_color="black",
        name="demand",
    )
    fig.show()
    if plot_export_path == '':
        pass
    else:
        fig.write_html(plot_export_path)

def plot_capacity(model_path, plot_export_path=''):

    model = calliope.read_netcdf(model_path)
    colors = model.inputs.color.to_series().to_dict()

    df_capacity = (
        model.results.flow_cap.where(model.results.techs != "demand_power")
        .sel(carriers="power")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Flow capacity (MW)")
        .reset_index()
    )

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

def plot_load_duration_curve(dataframe, plot_export_path='', x_label='count', y_label='value'):
    
    df = dataframe    
    column = df.columns[0]
    df = df.sort_values(by=column, ascending=False)
    df.index = [i for i in range(1, len(df.index) + 1)]
    
    fig = px.line(
        df,
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    fig.show()
    if plot_export_path == '':
        pass
    else:
        fig.write_html(plot_export_path)

