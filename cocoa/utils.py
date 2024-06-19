import six

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

    if 'weight' not in edges.columns:
        edges['weight'] = 1
    else:
        edges["weight"] = edges.weight.fillna(1).astype(int)

    # If graph is undirected, we have to sort the edges such that
    # neuron nodes are always the source
    types = nx.get_node_attributes(G, "type")
    if not nx.is_directed(G):
        target_is_neuron = edges.target.isin(types)
        edges.loc[target_is_neuron, "source"], edges.loc[target_is_neuron, "target"] = (
            edges.loc[target_is_neuron, "target"],
            edges.loc[target_is_neuron, "source"],
        )

    # Get the edges from neurons onto labels
    node_edges = edges[
        edges.source.isin([k for k, v in types.items() if v == "neuron"])
    ].copy()

    # If we have neurons from different datasets, we need to collapse them separately
    datasets = nx.get_node_attributes(G, "dataset")
    group2dataset = {}
    if datasets:
        to_collapse = []
        for ds in set(datasets.values()):
            # Get the nodes and their targets for this dataset
            node_edges_ds = node_edges[
                node_edges.source.isin([k for k, v in datasets.items() if v == ds])
            ]
            # Group such that we get a list of nodes that have the same targets
            grp = node_edges_ds.groupby("source").target.apply(tuple)
            this_to_collapse = (
                grp.groupby(grp).apply(lambda x: tuple(x.index)).values.tolist()
            )
            # Assign "group_i" labels to for each of the groups
            group2dataset.update(
                {
                    f"group_{i + len(to_collapse)}": ds
                    for i in range(len(this_to_collapse))
                }
            )
            to_collapse.extend(this_to_collapse)
    else:
        grp = node_edges.groupby("source").target.apply(tuple)
        to_collapse = grp.groupby(grp).apply(lambda x: tuple(x.index)).values

    # We could contract nodes in G now but that's painfully slow
    # Instead we will work on the edge list again
    groups = {n: f"group_{i}" for i, group in enumerate(to_collapse) for n in group}
    edges["source_new"] = edges.source.map(lambda x: groups.get(x, x))

    # Group edges
    edges_grp = edges.groupby(["source_new", "target"], as_index=False).weight.sum()

    # Make collapsed graph
    G_grp = nx.from_pandas_edgelist(
        edges_grp, source="source_new", edge_attr=True, create_using=nx.DiGraph
    )

    # Set a bunch of node attributes
    types = {
        f"group_{i}": "neuron" for i, group in enumerate(to_collapse) for n in group
    }
    nx.set_node_attributes(G_grp, types, "type")
    sizes = {
        f"group_{i}": len(group) for i, group in enumerate(to_collapse) for n in group
    }
    nx.set_node_attributes(G_grp, sizes, "size")

    # Copy over the dataset-related attributes (e.g. "FlyWire_in" or "FlyWire": True)
    if datasets:
        for ds in set(datasets.values()):
            nx.set_node_attributes(
                G_grp, nx.get_node_attributes(G, f"{ds}_in"), f"{ds}_in"
            )
            nx.set_node_attributes(G_grp, nx.get_node_attributes(G, f"{ds}"), f"{ds}")

    ids = {
        f"group_{i}": ",".join([str(n) for n in group])
        for i, group in enumerate(to_collapse)
    }
    nx.set_node_attributes(G_grp, ids, "ids")
    if group2dataset:
        nx.set_node_attributes(G_grp, group2dataset, "dataset")

    return G_grp
