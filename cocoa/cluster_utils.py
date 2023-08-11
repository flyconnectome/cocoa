import networkx as nx
import tanglegram as tg
import numpy as np

from functools import partial
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage


__all__ = ["extract_homogeneous_clusters"]


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


def extract_homogeneous_clusters(
    dists,
    labels,
    eval_func=is_good,
    max_dist=None,
    min_dist=None,
    min_dist_diff=None,
    link_method="ward",
    verbose=False,
):
    """Make clusters that contains representatives of each unique label.

    Parameters
    ----------
    dists :         pd.DataFrame
                    Distances from which to find clusters.
    labels :        np.ndarray
                    Labels for each row in `dists`.
    eval_func :     callable
                    Must accept two positional arguments:
                     1. A numpy array of label counts (e.g. `[1, 1, 2]`)
                     2. An integer describing how many unique labels we expect
                    Must return True if cluster composition is acceptable and
                    False if it isn't.
    min/max_dist :  float, optional
                    Use this to set a range of between-cluster distances at which
                    we are allowed to make clusters. For example:
                     - ``min_dist=.1`` means that we will not split further if
                       a cluster is already more similar than .1
                     - ``max_dist=1`` means that we will keep splitting clusters
                       that are more dissimilar than 1 even if they don't fullfil
                       the `eval_func`
    min_dist_diff : float, optional
                    Consider two homogenous clusters that are adjacent two each
                    other in the dendrogram: if the difference in distance
                    between the two clusters and their supercluster is smaller
                    than `min_dist_diff` they will be merged. Or in other words:
                    we will merge if the three horizontal lines in the dendrograms
                    are closer together than `min_dist_diff`.
    link_method :   str
                    Method to use for generating the linkage.

    Returns
    -------
    cl :        np.ndarray

    """
    if dists.values[0, 0] >= 0.999:
        dists = 1 - dists

    # Make linkage
    Z = linkage(squareform(dists, checks=False), method=link_method)

    # Turn linkage into graph
    G = tg.utils.linkage_to_graph(Z, labels=labels)

    # Add origin as node attribute
    n_unique_ds = len(np.unique(labels))

    # Prepare eval function
    def _eval_func(x):
        return eval_func(x, n_unique_ds)

    # Find clusters recursively
    clusters = {}
    label_dict = nx.get_node_attributes(G, "label")
    _ = _find_clusters_rec(
        G,
        clusters=clusters,
        eval_func=_eval_func,
        label_dict=label_dict,
        max_dist=max_dist,
        min_dist=min_dist,
        verbose=verbose,
    )

    # Keep only clusters labels for the leaf nodes
    clusters = {k: v for k, v in clusters.items() if k in label_dict}

    # Clusters are currently labels based at which hinge they were created
    # We have to renumber them
    reind = {c: i for i, c in enumerate(np.unique(list(clusters.values())))}
    clusters = {k: reind[v] for k, v in clusters.items()}

    cl = np.array([clusters[i] for i in np.arange(len(dists))])

    if min_dist_diff:
        cl = _merge_similar_clusters(cl=cl, G=G, Z=G, dist_thresh=min_dist_diff,
                                     verbose=verbose)

    return cl


def _find_clusters_rec(
    G, clusters, eval_func, label_dict, max_dist=None, min_dist=None, verbose=False
):
    """Recursively find clusters."""
    if G.is_directed:
        G = G.to_undirected()

    # The root node should always be the last in the graph
    root = max(G.nodes)

    try:
        dist = G.nodes[root]["distance"]  # the distance between the two prior clusters
    except KeyError:
        # If this is a leaf-node it won't have a "distance" property
        dist = 0

    # Remove the root in this graph
    G2 = G.copy()
    G2.remove_node(root)

    # Split into the two connected components
    CC = list(nx.connected_components(G2))
    # Count the number of labels (i.e. datasets) present in each subgraph
    counts = [_count_labels(c, label_dict=label_dict) for c in CC]
    # Evaluate the counts
    is_good = [eval_func(c) for c in counts]

    # Check if we should stop here
    stop = False
    # If we are below the minimum distance we have to stop
    if min_dist and (dist <= min_dist):
        stop = True
    # If the two clusters are bad...
    elif not all(is_good):
        # ... and the distance between the two clusters below is not too big
        # we can stop
        if max_dist and (dist <= max_dist):
            stop = True
        elif not max_dist:
            stop = True

    if not stop:
        for c in CC:
            _find_clusters_rec(
                G.subgraph(c),
                clusters=clusters,
                eval_func=eval_func,
                label_dict=label_dict,
                max_dist=max_dist,
                min_dist=min_dist,
                verbose=verbose,
            )
    else:
        if verbose:
            print(
                f"Found cluster of {sum([c.sum() for c in counts])} at distance {dist} ({root})"
            )
        clusters.update({n: root for n in G.nodes})

    return


def _count_labels(cluster, label_dict):
    """Takes a list of node IDs and counts labels among those."""
    cluster = list(cluster) if isinstance(cluster, set) else cluster
    cluster = np.asarray(cluster)
    cluster = cluster[np.isin(cluster, list(label_dict))]
    _, cnt = np.unique([label_dict[n] for n in cluster], return_counts=True)
    return cnt


def _merge_similar_clusters(cl, G, Z, dist_thresh, verbose=False):
    """Merge similar clusters.

    Parameters
    ----------
    cl :        np.ndarray
                Clusters membership that is to be checked.
    Z :         np.ndarray
                Linkage.
    G :         nx.DiGraph
                Graph representing the linkage.
    dist_thresh : float
                Distance under which to merge clusters.

    Returns
    -------
    cl :        np.ndarray
                Fixed cluster membership.

    """
    ix = np.arange(len(cl))
    to_merge = []
    for c1 in np.unique(cl):
        # Get the connected subgraph for this cluster
        p = nx.shortest_path(G.to_undirected(), source=ix[cl == c1][0], target=None)
        sg = [p[i] for i in ix[cl == c1][1:]]
        sg = np.unique([i for l in sg for i in l])  # flatten

        # The root for this cluster
        root = max(sg)

        dist_c1 = G.nodes[root].get("distance", 0)

        # The cluster one above this one
        try:
            top = next(G.predecessors(root))
        except StopIteration:
            continue
        except BaseException:
            raise

        # Distance between our original cluster and the closest
        dist_top = G.nodes[top].get("distance", np.inf)

        # Distance for the neighbouring cluster
        other = [s for s in G.successors(top) if s not in sg][0]
        dist_c2 = G.nodes[other].get("distance", 0)

        # If merging this and the next cluster are very similar
        th = 0.1
        if (dist_top - dist_c1) <= th and (dist_top - dist_c2) < th:
            # Get the index of the other cluster
            sg_other = list(nx.dfs_postorder_nodes(G, other))
            c2 = cl[[i for i in sg_other if i in ix]][0]

            if verbose:
                print(f"Merging {c1} and {c2} (top={dist_top}; left={dist_c1}, right={dist_c2}")

            to_merge.append([c1, c2])

    # Deduplicate
    to_merge = list(set([tuple(sorted(p)) for p in to_merge]))

    cl2 = cl.copy()
    for p in to_merge:
        cl2[cl2 == p[1]] = p[0]

    return cl2