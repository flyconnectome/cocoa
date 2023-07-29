import math

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import dash_bootstrap_components as dbc

from plotly import subplots
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

from ._dendrogram import _create_dendrogram


def interactive_clustermap(dists, labels, meta=None, cmap="rocket", **kwargs):
    """ """
    app = Dash(__name__)

    Z = sch.linkage(ssd.squareform(dists.values), method="ward", optimal_ordering=True)
    dists = dists.iloc[sch.leaves_list(Z), sch.leaves_list(Z)]

    cm = _interactive_clustermap(dists, labels, cmap=cmap, **kwargs)
    graph_cm = dcc.Graph(id="clustermap", figure=cm)

    if meta is None:
        meta = pd.DataFrame()
        meta["id"] = dists.index
    elif not isinstance(meta, pd.DataFrame):
        raise TypeError("Meta data must be DataFrame")
    elif "id" not in meta.columns:
        raise ValueError("Meta data tabke must have an `id` column")

    meta = meta.set_index('id').loc[dists.index.values].reset_index(drop=False)

    table = dash_table.DataTable(
        data=meta.to_dict("records"),
        style_data={"height": "auto", "whiteSpace": "normal"},
        columns=[{"id": c, "name": c} for c in meta.columns],
        # fill_width=True,
        id="data_table",
        page_size=10,
    )

    app.layout = html.Div(
        [
            html.Div(graph_cm, style={"display": "inline-block", "width": "49%"}),
            html.Div(table, style={"display": "inline-block", "width": "49%"}),
        ],
        className="g-0",
        style={"display": "flex"},
    )

    @app.callback(
        [Output("data_table", "data")],
        [Input("clustermap", "relayoutData")],  # this triggers the event
        [State("clustermap", "figure")],
    )
    def zoom_event(relayout_data, *figures):
        fig = figures[0]
        range = fig["layout"]["xaxis11"]["range"]
        range_fixed = np.array(range) // 10
        range_fixed = [math.floor(range_fixed[0]), math.ceil(range_fixed[1])]
        # print('Before', fig['layout']['xaxis11']["range"])
        fig["layout"]["xaxis11"]["range"] = range_fixed
        # print('After', fig['layout']['xaxis11']["range"])

        range_fixed[1] += 1
        range_fixed = np.array(range_fixed) * -1

        if range_fixed[0] != 0:
            data = meta.iloc[range_fixed[1]: range_fixed[0]].to_dict("records")
        else:
            data = meta.iloc[range_fixed[1]: ].to_dict("records")
        return (data, )

    return app


def _heatmap(dists, cmap):
    """Generate heatmaps."""
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)

    if callable(cmap):
        color_map = []
        for i in [0.0, 0.5, 1.0]:
            c = (np.array(cmap(i)) * 255).round()
            color_map.append([i, f"rgb({c[0]},{c[1]},{c[2]})"])
    elif isinstance(cmap, list):
        color_map = cmap
    else:
        raise TypeError(f"Unexpected cmap of type {type(cmap)}")

    hm = go.Heatmap(
        x=dists.index,
        y=dists.columns,
        z=dists.values,
        colorscale=color_map,
        showlegend=False,
        # TODO: This should be based on the text width of the labels, or
        # at least passable by the user, so they can adjust it
        # colorbar={"xpad": 100},
    )

    fig = go.Figure(data=hm)

    for ax in ["xaxis", "yaxis"]:
        fig["layout"][ax].update(
            fixedrange=True,
        )

    return fig


def _interactive_clustermap(
    dists, labels, linkage_method="ward", symbols=None, cmap="rocket", marks=None
):
    assert isinstance(dists, pd.DataFrame)

    Z = sch.linkage(
        ssd.squareform(dists.values), method=linkage_method, optimal_ordering=True
    )

    dend_col = _create_dendrogram(
        None, linkage=Z, orientation="bottom", color_threshold=1.5
    ).data
    dend_row = _create_dendrogram(
        None, linkage=Z, orientation="right", color_threshold=1.5
    ).data

    dists = dists.iloc[sch.leaves_list(Z), sch.leaves_list(Z)]

    ids = dists.index.values
    # ids_reordered = ids[sch.leaves_list(Z)]

    labels_dict = dict(zip(ids, labels))
    # labels_reordered = np.array([labels_dict.get(i, i) for i in ids_reordered])

    fig = subplots.make_subplots(
        rows=5,
        cols=5,
        specs=[
            [{}, {}, {"colspan": 2}, None, {}],
            [{}, {}, {"colspan": 2}, None, {}],
            [
                {"rowspan": 2},
                {"rowspan": 2},
                {"colspan": 2, "rowspan": 2},
                None,
                {"rowspan": 2},
            ],
            [None, None, None, None, None],
            [{}, {}, {"colspan": 2}, None, {}],
        ],
        vertical_spacing=0,
        horizontal_spacing=0,
        print_grid=False,
    )
    fig["layout"].update(hovermode="closest")

    tickvals_col = []
    tickvals_row = []

    # for column dendrogram, leaves are at bottom (y=0)
    for i in range(len(dend_col)):
        xs = dend_col[i]["x"]
        ys = dend_col[i]["y"]

        # during serialization (e.g., in a dcc.Store, the NaN
        # values become None and the arrays get turned into lists;
        # they must be converted back
        if isinstance(xs, list):
            xs = np.array(xs, dtype=np.float)
            dend_col[i].update(x=xs)
        if isinstance(ys, list):
            ys = np.array(ys, dtype=np.float)
            dend_col[i].update(y=ys)
        tickvals_col += [
            xs.flatten()[j]
            for j in range(len(xs.flatten()))
            if ys.flatten()[j] == 0.0 and xs.flatten()[j] % 10 == 5
        ]
    tickvals_col = list(set(tickvals_col))

    # for row dendrogram, leaves are at right(x=0, since we
    # horizontally flipped it)
    for i in range(len(dend_row)):
        xs = dend_row[i]["x"]
        ys = dend_row[i]["y"]

        if isinstance(xs, list):
            xs = np.array(xs, dtype=np.float)
            dend_row[i].update(x=xs)
        if isinstance(ys, list):
            ys = np.array(ys, dtype=np.float)
            dend_row[i].update(y=ys)

        tickvals_row += [
            ys.flatten()[j]
            for j in range(len(ys.flatten()))
            if xs.flatten()[j] == 0.0 and ys.flatten()[j] % 10 == 5
        ]
    tickvals_row = list(set(tickvals_row))

    # sort so they are in the right order (lowest to highest)
    tickvals_col.sort()
    tickvals_row.sort()

    # update axis settings for dendrograms and heatmap
    for i in [1, 3, 5, 6, 7, 9, 10, 11]:
        for s in ["x", "y"]:
            a = f"{s}axis{i}"
            fig["layout"][a].update(
                type="linear",
                showline=False,
                showgrid=False,
                zeroline=False,
                mirror=False,
                fixedrange=False,
                showticklabels=False,
            )

    # this dictionary relates curve numbers (accessible from the
    # hoverData/clickData props) to cluster numbers
    cluster_curve_numbers = {}
    for i in range(len(dend_col)):
        cdt = dend_col[i]
        cdt["name"] = "Col Cluster %d" % i
        # cdt["line"] = dict(width=self._line_width[1])
        cdt["hoverinfo"] = "y+name"
        cluster_curve_numbers[len(fig.data)] = ["col", i]
        fig.append_trace(cdt, 1, 3)

    # row dendrogram (displays on left side)
    for i in range(len(dend_row)):
        rdt = dend_row[i]
        rdt["name"] = "Row Cluster %d" % i
        # rdt["line"] = dict(width=self._line_width[0])
        rdt["hoverinfo"] = "x+name"
        cluster_curve_numbers[len(fig.data)] = ["row", i]
        fig.append_trace(rdt, 3, 1)

    col_dendro_traces_y = [r["y"] for r in dend_col]
    # arbitrary extrema if col_dendro_traces_y is empty
    col_dendro_traces_min_y = 0
    col_dendro_traces_max_y = 1
    if len(col_dendro_traces_y):
        col_dendro_traces_min_y = np.concatenate(col_dendro_traces_y).min()
        col_dendro_traces_max_y = np.concatenate(col_dendro_traces_y).max()

    # ensure that everything is aligned properly
    # with the heatmap
    yaxis9 = fig["layout"]["yaxis9"]
    yaxis9.update(scaleanchor="y11")
    xaxis3 = fig["layout"]["xaxis3"]
    xaxis3.update(scaleanchor="x11")

    if len(tickvals_col) == 0:
        tickvals_col = [10 * i + 5 for i in range(dists.shape[1])]

    # add in all of the labels
    fig["layout"]["xaxis11"].update(
        tickmode="array",
        tickvals=tickvals_col,
        ticktext=dists.columns,
        # tickfont=self._tick_font,
        showticklabels=True,
        side="bottom",
        showline=False,
        range=[min(tickvals_col) - 5, max(tickvals_col) + 5]
        # workaround for autoscale issues above; otherwise
        # the graph cuts off and must be scaled manually
    )

    if len(tickvals_row) == 0:
        tickvals_row = [10 * i + 5 for i in range(dists.shape[1])]

    fig["layout"]["yaxis11"].update(
        tickmode="array",
        tickvals=tickvals_row,
        ticktext=dists.index,
        # tickfont=self._tick_font,
        showticklabels=True,
        side="right",
        showline=False,
    )

    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)

    if callable(cmap):
        color_map = []
        for i in [0.0, 0.5, 1.0]:
            c = (np.array(cmap(i)) * 255).round()
            color_map.append([i, f"rgb({c[0]},{c[1]},{c[2]})"])
    elif isinstance(cmap, list):
        color_map = cmap
    else:
        raise TypeError(f"Unexpected cmap of type {type(cmap)}")

    heatmap = go.Heatmap(
        x=tickvals_col,
        y=tickvals_row,
        z=dists.values,
        colorscale=color_map,
        # TODO: This should be based on the text width of the labels, or
        # at least passable by the user, so they can adjust it
        # colorbar={"xpad": 100},
    )

    fig.append_trace(heatmap, 3, 3)

    # it seems the range must be set after heatmap is appended to the
    # traces, otherwise the range gets overwritten
    fig["layout"]["yaxis9"].update(
        range=[min(tickvals_row), max(tickvals_row)],
    )

    # hide all legends
    fig["layout"].update(
        showlegend=False,
    )

    # apply the display ratio
    row_ratio = 0
    col_ratio = 0

    # the argument can be either in list form or float form
    # first is ratio for row; second is ratio for column
    _display_ratio = [0.1, 0.1]
    if _display_ratio[0] != 0:
        row_ratio = (
            0 if len(dend_row) == 0 else 0.95 / float(1 + int(1 / _display_ratio[0]))
        )
    if _display_ratio[1] != 0:
        col_ratio = (
            0 if len(dend_col) == 0 else 0.95 / float(1 + int(1 / _display_ratio[1]))
        )

    # the row/column labels take up 0.05 of the graph, and the rest
    # is taken up by the heatmap and dendrogram for each dimension

    # row: dendrogram, heatmap, row labels (left-to-right)
    # column: dendrogram, column labels, heatmap (top-to-bottom)

    row_colors_heatmap = col_colors_heatmap = None
    row_colors_ratio = 0.01 if row_colors_heatmap is not None else 0
    col_colors_ratio = 0.01 if col_colors_heatmap is not None else 0

    # width adjustment for row dendrogram
    fig["layout"]["xaxis1"].update(domain=[0, 0.95])
    fig["layout"]["xaxis3"].update(
        domain=[row_ratio + row_colors_ratio, 0.95], anchor="y9"
    )
    fig["layout"]["xaxis5"].update(domain=[0, 0.95])
    fig["layout"]["xaxis7"].update(
        domain=[row_ratio + row_colors_ratio, 0.95], anchor="y9"
    )
    fig["layout"]["xaxis9"].update(domain=[0, row_ratio])
    fig["layout"]["xaxis10"].update(domain=[row_ratio, row_ratio + row_colors_ratio])
    fig["layout"]["xaxis11"].update(domain=[row_ratio + row_colors_ratio, 0.95])

    # height adjustment for column dendrogram
    fig["layout"]["yaxis1"].update(domain=[1 - col_ratio, 1])
    fig["layout"]["yaxis3"].update(
        domain=[1 - col_ratio, 1],
        range=[col_dendro_traces_min_y, col_dendro_traces_max_y],
    )
    fig["layout"]["yaxis5"].update(
        domain=[1 - col_ratio - col_colors_ratio, 1 - col_ratio]
    )

    fig["layout"]["yaxis6"].update(
        domain=[1 - col_ratio - col_colors_ratio, 1 - col_ratio]
    )
    fig["layout"]["yaxis7"].update(
        domain=[1 - col_ratio - col_colors_ratio, 1 - col_ratio]
    )

    fig["layout"]["yaxis9"].update(domain=[0, 1 - col_ratio - col_colors_ratio])

    fig["layout"]["yaxis10"].update(domain=[0, 1 - col_ratio - col_colors_ratio])
    fig["layout"]["yaxis11"].update(domain=[0, 1 - col_ratio - col_colors_ratio])

    fig["layout"]["legend"] = dict(x=0.7, y=0.7)

    # axis settings for subplots that will display group labels
    for a in ["xaxis12", "yaxis12", "xaxis15", "yaxis15"]:
        fig["layout"][a].update(
            type="linear",
            showline=False,
            showgrid=False,
            zeroline=False,
            mirror=False,
            fixedrange=False,
            showticklabels=False,
        )

    # group labels for row dendrogram
    fig["layout"]["yaxis12"].update(
        domain=[0, 0.95 - col_ratio], scaleanchor="y11", scaleratio=1
    )
    if len(tickvals_row) > 0:
        fig["layout"]["yaxis12"].update(range=[min(tickvals_row), max(tickvals_row)])
    # padding between group label line and dendrogram
    fig["layout"]["xaxis12"].update(domain=[0.95, 1], range=[-5, 1])

    # group labels for column dendrogram
    fig["layout"]["xaxis15"].update(
        domain=[row_ratio, 0.95], scaleanchor="x11", scaleratio=1
    )
    if len(tickvals_col) > 0:
        fig["layout"]["xaxis15"].update(range=[min(tickvals_col), max(tickvals_col)])
    fig["layout"]["yaxis15"].update(
        domain=[0.95 - col_ratio, 1 - col_ratio], range=[-0.5, 0.5]
    )

    # set background colors
    _paper_bg_color = _plot_bg_color = "rgba(0,0,0,0)"
    fig["layout"].update(paper_bgcolor=_paper_bg_color, plot_bgcolor=_plot_bg_color)

    # finally add height and width
    _height = 500
    _width = 500
    fig["layout"].update(height=_height, width=_width)

    return fig
