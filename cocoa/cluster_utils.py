import networkx as nx
import tanglegram as tg
import numpy as np

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


__all__ = ['extract_homogeneous_clusters']


def is_good(v, n_unique_ds):
    """Check if composition of labels.

    `v` is a len(3) vector with counts per label (i.e. datatset): e.g. [1, 2, 1]
    """
    if isinstance(v, dict):
        v = list(v.values())
    if len(v) < n_unique_ds:  # if not all datasets present
        return False
    mn = min(v)
    mx = max(v)
    if ((mx - mn) > 3) and ((mx / mn) >= 2):
        return False
    return True


def extract_homogeneous_clusters(dists, labels, eval_func=is_good, method="ward"):
    """Make clusters that contains representatives of each unique label.

    Parameters
    ----------
    dists :     pd.DataFrame
                Distances to cluster.
    labels :    np.ndarray
                Labels for each row in `dists`.
    eval_func : callable
                Must accept a vector of label counts (e.g. `[1, 1, 2]`) and
                return True if that composition is acceptable and False if it
                isn't.
    method :    str
                Method to use for generating the linkage.

    Returns
    -------
    cl :        np.ndarray

    """
    if dists.values[0, 0] >= 0.999:
        dists = 1 - dists

    # Make linkage
    Z = linkage(squareform(dists), method=method)

    # Turn linkage into graph
    G = tg.utils.linkage_to_graph(Z, labels=labels)

    # Add origin as node attribute
    n_unique_ds = len(np.unique(labels))

    # From the root of the tree, walk down to each leaf and then propagate the
    # label counts back up
    n = len(Z) + 1  # This is the number of leafs
    root = len(G) - 1  # This is the index of the root
    ds_counts = {}
    paths = nx.shortest_path(G, source=root)  # Get all paths from a node to the root
    for i in range(n):
        ds = G.nodes[i]["label"]  # Get label of this leaf
        for p in paths.get(i, [])[:-1]:  # Walk up towards the root
            ds_counts[p] = ds_counts.get(p, {})  # Add counts
            ds_counts[p][ds] = ds_counts[p].get(ds, 0) + 1

    # For each leaf
    keep = []
    for i in range(n):
        this_p = paths[i]
        # Walk up the path towards root
        for k in range(0, len(this_p)):
            # If this hinge/node hasn't been visited, stop
            if this_p[k] not in ds_counts:
                break
            # Get the dataset counts for this hinge
            v = ds_counts[this_p[k]]

            # Check if this hinge still satisfies our requirements
            if not eval_func(v, n_unique_ds):
                break

        # Keep only the nodes that comply
        keep += this_p[k - 1 :]

    # Make a subgraph
    SG = G.subgraph(list(set(keep)))

    # Get connected components
    CC = list(nx.connected_components(SG.to_undirected()))

    clusters = np.zeros(n, dtype=int)
    for i, nodes in enumerate(CC):
        for no in nodes:
            if no < n:
                clusters[no] = i

    return clusters
