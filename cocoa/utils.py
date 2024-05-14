import six
import functools

import numpy as np
import pandas as pd
import networkx as nx

from collections.abc import Iterable


def printv(*args, verbose=True, **kwargs):
    """Thin wrapper around print function."""
    if verbose:
        print(*args, **kwargs)


def check_frame(x, required_cols=None, dtypes=None):
    """Check dataframe."""
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f'Expected pandas.DataFrame, got "{type(x)}"')

    if not isinstance(required_cols, type(None)):
        if isinstance(required_cols, str):
            required_cols = [required_cols]
        for c in required_cols:
            if c not in x.columns:
                raise ValueError(f"DataFrame has to have a {c} column")

    if not isinstance(dtypes, type(None)):
        for c, types in dtypes.items():
            if not isinstance(types, (tuple, list)):
                types = (types,)

            if c not in x.columns:
                raise ValueError(f"DataFrame has to have a {c} column")
            if x[c].dtype not in types:
                raise ValueError(
                    f'Column {c} is expected to be of type "{types}" '
                    f"got {x[c].dtype}"
                )


def make_iterable(x, force_type=None) -> np.ndarray:
    """Force input into a numpy array.

    For dicts, keys will be turned into array.
    """
    if not isinstance(x, Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, (dict, set)):
        x = list(x)

    return np.asarray(x, dtype=force_type)


def req_compile(func):
    """Check if we need to compile connectivity."""

    @functools.wraps(func)
    def inner(*args, **kwargs):
        if not hasattr(args[0], "dists_"):
            args[0].compile()
        return func(*args, **kwargs)

    return inner


def collapse_neuron_nodes(G):
    """Collapse nodes representing neurons into groups with the same connectivity.

    Parameters
    ----------
    G : networkx Graph
        Graph with nodes representing neurons and labels. Neuron nodes must
        have a 'type' attribute.

    Returns
    -------
    G_grp : nx.DiGraph
        Graph with neuron nodes collapsed into groups.

    """
    # Turn into edge list
    edges = nx.to_pandas_edgelist(G)
    edges["weight"] = edges.weight.fillna(1).astype(int)
    # Get edges from neurons to labels
    types = nx.get_node_attributes(G, "type")
    node_edges = edges[
        edges.source.isin([k for k, v in types.items() if v == "neuron"])
    ].copy()

    # If we have neurons from different datasets, we need to collapse them separately
    datasets = nx.get_node_attributes(G, "dataset")
    group2dataset = {}
    if datasets:
        to_collapse = []
        for ds in set(datasets.values()):
            node_edges_ds = node_edges[
                node_edges.source.isin([k for k, v in datasets.items() if v == ds])
            ]
            adj = (
                node_edges_ds.pivot(index="source", columns="target", values="weight")
                .fillna(0)
                .astype(bool)
            )
            this_to_collapse = (
                adj.groupby(list(adj))
                .apply(lambda x: tuple(x.index), include_groups=False)
                .values.tolist()
            )
            group2dataset.update(
                {
                    f"group_{i+ len(to_collapse)}": ds
                    for i in range(len(this_to_collapse))
                }
            )
            to_collapse.extend(this_to_collapse)
    else:
        # Pivot
        adj = (
            node_edges.pivot(index="source", columns="target", values="weight")
            .fillna(0)
            .astype(bool)
        )
        # Collapse - this is now a tuple of (id1, id2, ...) for each group
        to_collapse = (
            adj.groupby(list(adj))
            .apply(lambda x: tuple(x.index), include_groups=False)
            .values
        )

    # We could contract nodes in G now but that's painfully slow
    groups = {n: f"group_{i}" for i, group in enumerate(to_collapse) for n in group}
    edges["source_new"] = edges.source.map(lambda x: groups.get(x, x))

    # Group edges
    edges_grp = edges.groupby(["source_new", "target"], as_index=False).weight.sum()

    # Make collapsed graph
    G_grp = nx.from_pandas_edgelist(edges_grp, source="source_new", edge_attr=True)

    # Set a bunch of node attributes
    types = {
        f"group_{i}": "neuron" for i, group in enumerate(to_collapse) for n in group
    }
    nx.set_node_attributes(G_grp, types, "type")
    sizes = {
        f"group_{i}": len(group) for i, group in enumerate(to_collapse) for n in group
    }
    nx.set_node_attributes(G_grp, sizes, "size")
    ids = {
        f"group_{i}": ",".join([str(n) for n in group])
        for i, group in enumerate(to_collapse)
    }
    nx.set_node_attributes(G_grp, ids, "ids")
    if group2dataset:
        nx.set_node_attributes(G_grp, group2dataset, "dataset")

    return G_grp
