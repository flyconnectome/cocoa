import uuid
import copy
import navis

import matplotlib.colors as mcl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import scipy as scp
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from plotly.figure_factory._dendrogram import _Dendrogram
from dash import Dash
from fafbseg.flywire import neuroglancer


# Red (1) -> Green (5)
CONF_COLORS = {1: "#fc3a1c", 2: "#f5b642", 3: "#00ab39"}

# Some hard-coded X positions for the left and the right columns in the
# tanglegram
X_LEFT = 0
X_RIGHT = 5

COLORS = {"None": mcl.to_hex("w")}


def interactive_dendrogram(
    dists,
    labels,
    linkage_method="ward",
    symbols=None,
    marks=None,
):
    """Generate Dash app for exploring and editing labels on dendrograms.

    Parameters
    ----------

    """
    app = Dash(__name__)

    labels = np.asarray(labels).astype(str)
    if symbols is not None:
        symbols = np.asarray(symbols).astype(str)

    if dists.shape[0] > 1:
        Z = sch.linkage(
            ssd.squareform(dists.values), method=linkage_method, optimal_ordering=False
        )
    else:
        Z = None
    ids = dists.index.values

    labels_dict = dict(zip(ids, labels))

    dend = create_dendrogram(
        Z, ids=ids, clusters=labels_dict, marks=marks, symbols=symbols
    )

    graph = dcc.Graph(id="graph", figure=dend)
    available_ct = np.unique(labels[~pd.isnull(labels)])
    available_ct = [c for l in available_ct for c in l.split(",")]
    available_ct = np.unique(available_ct)
    available_ct = [c for c in available_ct if c and c != "None"]
    available_ct = [
        "None",
        "Combine selected",
        "Reset selected",
        "New random",
    ] + available_ct

    dd = dcc.Dropdown(available_ct, id="dd", placeholder="Change celltype")

    empty = html.Div(id="empty")
    empty1 = html.Div(id="empty1")
    empty2 = html.Div(id="empty2")

    button_show = html.Button("Open URL", id="button_show")
    button_show_sel = html.Button("Open selected", id="button_show_selected")

    app.layout = html.Div(
        children=[
            button_show,
            button_show_sel,
            html.Div([graph, dd]),
            empty,
            empty1,
            empty2,
        ]
    )

    @app.callback(
        Output("graph", "figure"),
        Output("dd", "value"),
        Input("dd", "value"),
        State("graph", "selectedData"),
        State("graph", "figure"),
    )
    def update_labels(ct, selected, fig):
        selected = extract_scatter(selected, fig)

        if not ct:
            raise PreventUpdate
        if not selected:
            raise PreventUpdate
        elif not selected.get("points", None):
            raise PreventUpdate

        if ct == "New random":
            ct = str(uuid.uuid4()).split("-")[0]

        # Change labels
        if ct == "Combine selected":
            fig = combine_labels(selected, fig)
        elif ct == "Reset selected":
            fig = reset_labels(selected, fig, labels_dict)
        else:
            fig = change_labels(ct, selected, fig)

        return fig, ""

    def change_labels(ct, selected, fig):
        # Change labels
        to_change = np.array([p["pointIndex"] for p in selected["points"]])

        trace = [t for t in fig["data"] if t.get("name") == "leafs"][0]

        for p in to_change:
            trace["text"][p] = ct

        trace["marker"]["color"] = [get_color(t) for t in trace["text"]]

        return fig

    def combine_labels(selected, fig):
        # Change labels
        to_change = np.array([p["pointIndex"] for p in selected["points"]])
        trace = [t for t in fig["data"] if t.get("name") == "leafs"][0]

        cell_types = [p["text"] for p in selected["points"]]
        cell_types = [s for t in cell_types for s in t.split(",")]
        cell_types = np.unique(cell_types)
        cell_types = cell_types[cell_types != "None"]

        for p in to_change:
            trace["text"][p] = ",".join(cell_types)

        trace["marker"]["color"] = [get_color(t) for t in trace["text"]]

        return fig

    def reset_labels(selected, fig, labels_dict):
        # Reset
        to_change = np.array([p["pointIndex"] for p in selected["points"]])
        ids = np.array([p["customdata"] for p in selected["points"]])
        trace = [t for t in fig["data"] if t.get("name") == "leafs"][0]

        for p, i in zip(to_change, ids):
            trace["text"][p] = labels_dict.get(i, labels_dict.get(int(i), "NA"))

        trace["marker"]["color"] = [get_color(t) for t in trace["text"]]

        return fig

    @app.callback(
        Output("empty", "children"),
        Input("button_show", "n_clicks"),
        State("graph", "figure"),
    )
    def open_url(n_clicks, fig):
        if n_clicks is None:
            raise PreventUpdate

        trace = [t for t in fig["data"] if t.get("name") == "leafs"][0]
        ids = np.array(trace["customdata"])
        colors = np.array(trace["marker"]["color"])
        groups = np.array(trace["text"])

        is_fw = is_flywire_id(ids)
        seg_ids, body_ids = ids[is_fw], ids[~is_fw]
        seg_colors, body_colors = colors[is_fw], colors[~is_fw]
        seg_groups, body_groups = groups[is_fw], [f"hb{g}" for g in groups[~is_fw]]

        encode_url(
            root_ids=seg_ids,
            seg_colors=seg_colors,
            seg_groups=seg_groups,
            body_ids=body_ids,
            body_groups=body_groups,
            body_colors=body_colors,
            open_browser=True,
        )

        return []

    @app.callback(
        Output("empty1", "children"),
        Input("button_show_selected", "n_clicks"),
        State("graph", "selectedData"),
        State("graph", "figure"),
    )
    def open_selected(n_clicks, sel, fig):
        if n_clicks is None:
            raise PreventUpdate

        sel = extract_scatter(sel, fig)

        if sel:
            ids = np.array([p["customdata"] for p in sel["points"]])
            colors = np.array([p["marker.color"] for p in sel["points"]])
            groups = np.array([p["text"] for p in sel["points"]])
            is_fw = is_flywire_id(ids)
            seg_ids, body_ids = ids[is_fw], ids[~is_fw]
            seg_colors, body_colors = colors[is_fw], colors[~is_fw]
            seg_groups, body_groups = groups[is_fw], [f"hb{g}" for g in groups[~is_fw]]
        else:
            seg_ids, body_ids = None
            seg_colors, body_colors = None
            seg_groups, body_groups = None

        encode_url(
            root_ids=seg_ids,
            seg_colors=seg_colors,
            seg_groups=seg_groups,
            body_ids=body_ids,
            body_groups=body_groups,
            body_colors=body_colors,
            open_browser=True,
        )

        return []

    return app


def extract_scatter(selected, fig):
    if not selected:
        return
    # Get the curve number corresponding to the leafs scatterplot in this figure
    curve_no = [i for i, t in enumerate(fig["data"]) if t.get("name") == "leafs"][0]

    # Remove all points that don't belong to the leafs scatter plot
    if "points" in selected:
        selected["points"] = [
            p for p in selected["points"] if p.get("curveNumber") == curve_no
        ]

    return selected


def create_dendrogram(Z, ids, clusters=None, marks=None, symbols=None, fig=None):
    """Generate dendrogram for left/right scores.

    Parameters
    ----------
    Z :             DataFrame
                    Linkage.
    ids :           iterable
                    Labels in same order as original distance matrix.
    clusters :      dict, optional
                    Dict with cluster IDs for each neuron. Used for the
                    scatterplot.
    marks :          iterable of bools
                    True/False in same order as original distance matrix.
                    Where mark==True will add a asterisk at the leaf

    Returns
    -------
    fig :           plotly Figure

    """
    # Get the order of labels
    if Z is not None:
        order = sch.leaves_list(Z)
    else:
        order = np.array([0])

    # Map to y positions
    ids = np.asarray(ids)
    ids_ordered = ids[order]

    # Generate scatter plot
    if clusters is not None:
        # Check if our clusters already have a color
        colors = [
            get_color(clusters.get(i, clusters.get(str(i), None))) for i in ids_ordered
        ]
    else:
        colors = "rgba(0.5, 0.5, 0.5, 1)"

    # Generate hovertext
    hovertexts = []
    clusters = clusters if clusters else {}
    for i in ids_ordered:
        cl = clusters.get(i, None)
        if cl is not None:
            hovertexts.append(f"{cl}")
        else:
            hovertexts.append(f"{i}")

    if not fig:
        fig = go.Figure()

    if symbols is None:
        symbols = "square"
    elif not isinstance(symbols, str):
        assert len(symbols) == len(ids)
        from plotly.validators.scatter.marker import SymbolValidator

        raw_symbols = SymbolValidator().values[2::3]
        if not all([s in raw_symbols for s in symbols]):
            simple_symbols = [s for s in raw_symbols if "-" not in s]
            unique = np.unique(symbols)
            symbol_mapping = dict(zip(unique, simple_symbols))

            symbols_dict = dict(zip(ids, [symbol_mapping[s] for s in symbols]))
            symbols = [symbols_dict[i] for i in ids_ordered]

    # Create the figure and scatter plot for the leafs
    scatter = go.Scatter(
        y=np.full(len(ids), fill_value=0),
        x=np.arange(len(ids)) * 10 + 5,
        mode="markers+text",
        name="leafs",
        hoverinfo="text",
        text=hovertexts,
        textposition="bottom center",
        textfont=dict(size=10, color="crimson"),
        customdata=ids_ordered.astype(str),
        marker=dict(size=20, color=colors, symbol=symbols),
    )

    if marks is not None:
        marks = marks[order]
        marks_scatter = go.Scatter(
            y=np.full(len(ids), fill_value=0)[~pd.isnull(marks)],
            x=(np.arange(len(ids)) * 10 + 5)[~pd.isnull(marks)],
            mode="markers",
            name="marks",
            hoverinfo="text",
            text=marks[~pd.isnull(marks)],
            marker=dict(size=5, color="rgb(1,1,1)"),
        )

    # Create dendrograms
    if Z is not None:
        dend = _create_dendrogram(
            None, linkage=Z, orientation="bottom", color_threshold=1.5
        )

        # Move dendograms to other axis and add to figure
        for i in range(len(dend["data"])):
            dend["data"][i]["xaxis"] = "x"

        for data in dend["data"]:
            fig.add_trace(data)
    fig.add_trace(scatter)

    if marks is not None:
        fig.add_trace(marks_scatter)

    fig.layout.xaxis.visible = True
    fig.layout.yaxis.visible = True
    fig.layout.yaxis.fixedrange = False
    fig.layout.xaxis.fixedrange = False
    # fig.layout.height = 700
    # fig.layout.width = 600
    fig.layout.paper_bgcolor = "rgba(1,1,1,0)"
    fig.layout.plot_bgcolor = "rgba(1,1,1,0)"
    fig.layout.showlegend = False
    fig.layout.dragmode = "select"

    # Edit xaxes
    fig.update_layout(
        xaxis={  #'domain': [0, .5],
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        },
        yaxis={
            "showgrid": False,
        },
    )

    return fig


def _create_dendrogram(
    X,
    orientation="bottom",
    labels=None,
    colorscale=None,
    linkage=None,
    linkagefun=lambda x: sch.linkage(x, "complete"),
    hovertext=None,
    color_threshold=None,
    mark_threshold=False,
    above_threshold_color="#d3d3d3",
):
    """
    Function that returns a dendrogram Plotly figure object. This is a thin
    wrapper around scipy.cluster.hierarchy.dendrogram.
    See also https://dash.plot.ly/dash-bio/clustergram.

    :param (ndarray) X: (N, N) Matrix of pairwise distances
    :param (str) orientation: 'top', 'right', 'bottom', or 'left'
    :param (list) labels: List of axis category labels(observation labels)
    :param (list) colorscale: Optional colorscale for the dendrogram tree.
                              Requires 8 colors to be specified, the 7th of
                              which is ignored.  With scipy>=1.5.0, the 2nd, 3rd
                              and 6th are used twice as often as the others.
                              Given a shorter list, the missing values are
                              replaced with defaults and with a longer list the
                              extra values are ignored.
    :param (function) distfun: Function to compute the pairwise distance from
                               the observations
    :param (function) linkagefun: Function to compute the linkage matrix from
                               the pairwise distances
    :param (list[list]) hovertext: List of hovertext for constituent traces of dendrogram
                               clusters
    :param (double) color_threshold: Value at which the separation of clusters will be made
    Example 1: Simple bottom oriented dendrogram
    >>> from plotly.figure_factory import create_dendrogram
    >>> import numpy as np
    >>> X = np.random.rand(10,10)
    >>> fig = create_dendrogram(X)
    >>> fig.show()
    Example 2: Dendrogram to put on the left of the heatmap

    >>> from plotly.figure_factory import create_dendrogram
    >>> import numpy as np
    >>> X = np.random.rand(5,5)
    >>> names = ['Jack', 'Oxana', 'John', 'Chelsea', 'Mark']
    >>> dendro = create_dendrogram(X, orientation='right', labels=names)
    >>> dendro.update_layout({'width':700, 'height':500}) # doctest: +SKIP
    >>> dendro.show()
    Example 3: Dendrogram with Pandas

    >>> from plotly.figure_factory import create_dendrogram
    >>> import numpy as np
    >>> import pandas as pd
    >>> Index= ['A','B','C','D','E','F','G','H','I','J']
    >>> df = pd.DataFrame(abs(np.random.randn(10, 10)), index=Index)
    >>> fig = create_dendrogram(df, labels=Index)
    >>> fig.show()
    """
    if not isinstance(X, type(None)):
        s = np.asarray(X).shape
        if len(s) != 2:
            raise ValueError("X should be 2-dimensional array.")
    elif isinstance(linkage, type(None)):
        raise ValueError("Must provide either `X` or `linkage`")

    dendrogram = Dendrogram(
        X,
        orientation,
        labels,
        colorscale,
        linkagefun=linkagefun,
        hovertext=hovertext,
        color_threshold=color_threshold,
        above_threshold_color=above_threshold_color,
        linkage=linkage,
    )

    fig = go.Figure(data=dendrogram.data, layout=dendrogram.layout)

    if color_threshold and mark_threshold:
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=0,
            y0=color_threshold,
            x1=s[0] * 10,
            y1=color_threshold,
            line=dict(color="white", width=2, dash="dot"),
        )

    return fig


class Dendrogram(_Dendrogram):
    """Plotly Dendrogram that works with distances instead of observations.

    Also accepts a linkage as input.
    """

    def __init__(
        self,
        X,
        orientation="bottom",
        labels=None,
        colorscale=None,
        width=np.inf,
        height=np.inf,
        xaxis="xaxis",
        yaxis="yaxis",
        distfun=None,
        linkagefun=lambda x: sch.linkage(x, "complete"),
        hovertext=None,
        color_threshold=None,
        **kwargs,
    ):
        # Set kwargs before super
        for k, v in kwargs.items():
            setattr(self, k, v)

        super().__init__(
            X,
            orientation=orientation,
            labels=labels,
            colorscale=colorscale,
            width=width,
            height=height,
            xaxis=xaxis,
            yaxis=yaxis,
            distfun=distfun,
            linkagefun=linkagefun,
            hovertext=hovertext,
            color_threshold=color_threshold,
        )

    def get_dendrogram_traces(
        self, d, colorscale, __distfun, linkagefun, hovertext, color_threshold
    ):
        """
        Calculates all the elements needed for plotting a dendrogram.
        :param (ndarray) X: Matrix of observations as array of arrays
        :param (list) colorscale: Color scale for dendrogram tree clusters
        :param (IGNORED) distfun: Function to compute the pairwise distance
                                   from the observations
        :param (function) linkagefun: Function to compute the linkage matrix
                                      from the pairwise distances
        :param (list) hovertext: List of hovertext for constituent traces of dendrogram
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for dendrogram tree
            (b) icoord: All X points of the dendrogram tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the dendrogram tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) P['leaves']: left-to-right traversal of the leaves
        """
        Z = getattr(self, "linkage", None)
        if isinstance(Z, type(None)):
            if not d.ndim == 1:
                d = ssd.squareform(d, checks=False)

            Z = linkagefun(d)
        P = sch.dendrogram(
            Z,
            orientation=self.orientation,
            labels=self.labels,
            no_plot=True,
            color_threshold=color_threshold,
            above_threshold_color=self.above_threshold_color,
        )

        icoord = scp.array(P["icoord"])
        dcoord = scp.array(P["dcoord"])
        ordered_labels = scp.array(P["ivl"])
        color_list = scp.array(P["color_list"])
        colors = self.get_color_dict(colorscale)

        trace_list = []

        for i in range(len(icoord)):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree
            if self.orientation in ["top", "bottom"]:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ["top", "bottom"]:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=colors.get(color_key, color_key)),
                text=hovertext_label,
                hoverinfo="text",
            )

            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ""

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        return trace_list, icoord, dcoord, ordered_labels, P["leaves"]


def get_color(label):
    """Get color for given label. Generate new color if not present."""
    if label not in COLORS:
        COLORS[label] = mcl.to_hex(sns.color_palette("tab20", len(COLORS) + 1)[-1])

    return COLORS[label]


def is_flywire_id(x):
    """Check if ID(s) are FlyWire or hemibrain IDs"""
    x = np.asarray(x).astype(int)

    return x > 1e10


HB_MESH_LAYER = {
    "type": "segmentation",
    "mesh": "precomputed://https://spine.itanna.io/files/data/hemibrain2flywire/precomputed/neuronmeshes/mesh/",
    "colorSeed": 3429908875,
    "segments": [],
    "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
    "name": "hemibrain_meshes",
}

HB_MESH_LAYER_MIRR = {
    "type": "segmentation",
    "mesh": "precomputed://https://spine.itanna.io/files/data/hemibrain2flywire_mirror/precomputed/neuronmeshes/mesh/",
    "colorSeed": 3429908875,
    "segments": [],
    "skeletonRendering": {"mode2d": "lines_and_points", "mode3d": "lines"},
    "name": "hemibrain_meshes",
}


def encode_url(
    root_ids=None,
    body_ids=None,
    body_colors=None,
    body_groups=None,
    hb_mirrored=False,
    **kwargs,
):
    """Encode data as FlyWire neuroglancer scene.

    Parameters
    ----------
    root_ids :      int | list of int, optional
                    FlyWire root IDs to have selected.
    body_ids :      int | list of int, optional
                    Hemibrain body IDs to have selected.
    body_colors :   str | tuple | list | dict, optional
                    Single color (name or RGB tuple), or list or dictionary
                    mapping colors to ``segments``. Can also be a numpy array
                    of labels which will be automatically turned into colors.
    **kwargs
                    Keyword arguments are passed through to `flywire.encode_url`.

    Returns
    -------
    url :           str

    """
    if body_ids is not None:
        if isinstance(body_ids, (int, str)):
            body_ids = [int(body_ids)]

        scene = neuroglancer.decode_url(neuroglancer.encode_url(short=False, neuroglancer='basic'), format='full')

        scene["layout"] = "3d"
        scene["navigation"] = {
            "pose": {
                "position": {
                    "voxelSize": [4, 4, 40],
                    "voxelCoordinates": [
                        133186.171875,
                        59157.77734375,
                        4065.686279296875,
                    ],
                }
            },
            "zoomFactor": 2.8,
        }

        scene["layers"].append(copy.deepcopy(HB_MESH_LAYER))
        scene["layers"][-1]["segments"] = [str(i) for i in body_ids]
        seg_layer_ix = len(scene["layers"]) - 1

        if hb_mirrored:
            scene["layers"][-1]["mesh"] = scene["layers"][-1]["mesh"].replace(
                "hemibrain2flywire", "hemibrain2flywire_mirror"
            )

        # See if we need to assign colors
        if body_colors is not None:
            if isinstance(body_colors, list):
                body_colors = np.array(body_colors)

            if isinstance(body_colors, str):
                body_colors = {s: body_colors for s in body_ids}
            elif isinstance(body_colors, tuple) and len(body_colors) == 3:
                body_colors = {s: body_colors for s in body_ids}
            elif (
                isinstance(body_colors, (np.ndarray, pd.Series))
                and body_colors.ndim == 1
            ):
                if len(body_colors) != len(body_ids):
                    raise ValueError(
                        f"Got {len(body_colors)} colors for {len(body_ids)} segments."
                    )

                uni_ = np.unique(body_colors)
                if len(uni_) > 20:
                    # Note the +1 to avoid starting and ending on the same color
                    pal = sns.color_palette("hls", len(uni_) + 1)
                    # Shuffle to avoid having two neighbouring clusters with
                    # similar colours
                    rng = np.random.default_rng(1985)
                    rng.shuffle(pal)
                elif len(uni_) > 10:
                    pal = sns.color_palette("tab20", len(uni_))
                else:
                    pal = sns.color_palette("tab10", len(uni_))
                _colors = dict(zip(uni_, pal))
                body_colors = {s: _colors[l] for s, l in zip(body_ids, body_colors)}
            elif not isinstance(body_colors, dict):
                if not navis.utils.is_iterable(body_colors):
                    raise TypeError(
                        f'`body_colors` must be dict or iterable, got "{type(body_colors)}"'
                    )
                if len(body_colors) < len(body_ids):
                    raise ValueError(
                        f"Got {len(body_colors)} colors for {len(body_ids)} segments."
                    )

                # Turn into dictionary
                body_colors = dict(zip(body_ids, body_colors))

            # Turn colors into hex codes
            # Also make sure keys are int (not np.int64)
            # Not sure but this might cause issue on Windows systems
            # But JSON doesn't like np.int64... so we're screwed
            body_colors = {str(s): mcl.to_hex(c) for s, c in body_colors.items()}

            # Assign colors
            scene["layers"][-1]["segmentColors"] = body_colors

        if body_groups is not None:
            if not isinstance(body_groups, dict):
                if not navis.utils.is_iterable(body_groups):
                    raise TypeError(
                        f'`body_groups` must be dict or iterable, got "{type(body_groups)}"'
                    )
                if len(body_groups) != len(body_ids):
                    raise ValueError(
                        f"Got {len(body_groups)} groups for {len(body_ids)} body IDs."
                    )

                body_groups = np.asarray(body_groups)

                if body_groups.dtype != object:
                    body_groups = [f"group_{i}" for i in body_groups]

                # Turn into dictionary
                body_groups = dict(zip(body_ids, body_groups))

            # Check if dict is {id: group} or {group: [id1, id2, id3]}
            is_list = [
                isinstance(v, (list, tuple, set, np.ndarray))
                for v in body_groups.values()
            ]
            if not any(is_list):
                groups = {}
                for s, g in body_groups.items():
                    if not isinstance(g, str):
                        raise TypeError(
                            f"Expected body groups to be strings, got {type(g)}"
                        )
                    groups[g] = groups.get(g, []) + [s]
            elif all(is_list):
                groups = body_groups
            else:
                raise ValueError(
                    "`body_groups` appears to be a mix of {id: group} "
                    "and {group: [id1, id2, id3]}."
                )

            for g in groups:
                scene["layers"].append(copy.deepcopy(scene["layers"][seg_layer_ix]))
                scene["layers"][-1]["name"] = f"{g}"
                scene["layers"][-1]["segments"] = [str(s) for s in groups[g]]
                scene["layers"][-1]["visible"] = False
                if body_colors is not None:
                    scene["layers"][-1]["segmentColors"] = body_colors

    else:
        scene = None

    return neuroglancer.encode_url(
        segments=root_ids,
        scene=scene,
        neuroglancer='basic',
        short=False,
        **kwargs,
    )