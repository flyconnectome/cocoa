import heapq
import re
import warnings

import networkx as nx
import tanglegram as tg
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.distance import squareform, cdist
from tqdm.auto import tqdm
from multiprocessing import Pool, shared_memory
from multiprocessing.managers import SharedMemoryManager
from scipy.sparse import coo_array, coo_matrix, csr_matrix, issparse
from scipy.sparse import load_npz as sparse_load_npz
from scipy.sparse.csgraph import connected_components

from .composition import ClusterContext, EqualBalance, _wants_context
from .distance import calculate_distance
from .utils import printv

__all__ = [
    "extract_homogeneous_clusters",
    "extract_homogeneous_clusters_knn",
    "extract_homogeneous_clusters_graph",
    "knn_tree",
    "graph_tree",
    "read_knn_graph",
    "tree_heights",
    "tree_to_linkage",
]

# Above this many features we reduce dimensionality before running Ward - the
# connectivity-constrained agglomeration needs a dense float64 array and the
# connectivity vectors are typically both very wide and very sparse
SVD_THRESHOLD = 256
SVD_COMPONENTS = 200

# Refuse to materialise a dense array larger than this (in GB) - see `n_components`
DENSE_LIMIT_GB = 8.0

# Floor for graph edge weights - see `_to_similarity`
_SIM_EPS = 1e-9


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
    dists_or_linkage,
    labels,
    eval_func=EqualBalance(),
    max_dist=None,
    min_dist=None,
    min_dist_diff=None,
    link_method="ward",
    verbose=False,
):
    """Make clusters that contains representatives of each unique label.

    Parameters
    ----------
    dists_or_linkage : pd.DataFrame | np.ndarray
                    Either a square distance (or similarity) matrix, or a
                    precomputed linkage of shape (N, 4) as produced by
                    `scipy.cluster.hierarchy.linkage`. Which one it is is
                    detected from the shape; `link_method` is ignored if a
                    linkage is given.
    labels :        np.ndarray
                    Labels for each observation, in the same order as the rows
                    of the distance matrix (or the leaves of the linkage).
    eval_func :     CompositionCriterion | callable
                    Decides whether a candidate cluster has an acceptable mix of
                    labels (i.e. datasets). Defaults to `EqualBalance`, which
                    expects every dataset to be present and to contribute roughly
                    equally - the same rule the older `is_good` function applies.
                    If your datasets differ in size, e.g. one covers both
                    hemispheres and another only one, use `ProportionalBalance`
                    instead: it compares each dataset against its share of the
                    pool rather than against the other datasets.

                    A `CompositionCriterion` is called as ``(counts, ctx)``, where
                    `counts` includes zeros for absent datasets and `ctx` is a
                    `ClusterContext`. For backwards compatibility a plain function
                    is instead called as ``(counts_without_zeros, n_datasets)``.
                    Either way it returns True if the composition is acceptable.
    min/max_dist :  float, optional
                    Use this to set a range of between-cluster distances at which
                    we are allowed to make clusters. For example:
                     - ``min_dist=.1`` means that we will not split further if
                       a cluster is already more similar than .1
                     - ``max_dist=1`` means that we will keep splitting clusters
                       that are more dissimilar than 1 even if they don't fullfil
                       the `eval_func`
    min_dist_diff : float, optional
                    Consider two homogenous clusters that are adjacent to each
                    other in the dendrogram: if the difference in distance
                    between the two clusters and their supercluster is smaller
                    than `min_dist_diff` they will be merged. Or in other words:
                    we will merge if the three horizontal lines in the dendrograms
                    are closer together than `min_dist_diff`.
    link_method :   str
                    Method to use for generating the linkage. Ignored if
                    `dists_or_linkage` is already a linkage.

    Returns
    -------
    cl :            np.ndarray
                    Cluster membership for each observation, in the same order as
                    the rows of the distance matrix (or the leaves of the linkage).

    """
    if _is_linkage(dists_or_linkage):
        Z = np.asarray(dists_or_linkage)
    else:
        dists = dists_or_linkage
        if isinstance(dists, pd.DataFrame):
            dists = dists.values
        dists = np.asarray(dists)
        if dists.ndim != 2 or dists.shape[0] != dists.shape[1]:
            raise ValueError(
                "Expected either a square distance matrix or an (N, 4) linkage, "
                f"got array of shape {dists.shape}."
            )
        # Distance matrices have a zero diagonal, similarity matrices a diagonal
        # of one - use that to detect the latter and convert it
        if dists[0, 0] >= 0.999:
            dists = 1 - dists
        Z = sch.linkage(squareform(dists, checks=False), method=link_method)

    # The linkage has one row per merge, i.e. one less than the observations
    n_obs = Z.shape[0] + 1

    # Describe the dendrogram as flat arrays - this is a lot cheaper to walk
    # than a networkx graph and lets us count labels in a single sweep
    return _extract_from_tree(
        _dendrogram_arrays(Z, n_obs) + (n_obs,),
        labels=labels,
        eval_func=eval_func,
        max_dist=max_dist,
        min_dist=min_dist,
        min_dist_diff=min_dist_diff,
        verbose=verbose,
    )


def _enforce_monotonic(node_dist, n_obs, verbose=False):
    """Make merge heights non-decreasing towards the root.

    Neither a neighbourhood-constrained Ward nor Paris guarantees monotone
    heights, but `min_dist`, `max_dist` and `min_dist_diff` all assume it.
    Children always have lower IDs than their parent, so a running maximum over
    the hinges in ID order is enough.
    """
    n_inv = int((np.diff(node_dist[n_obs:]) < 0).sum())
    if n_inv:
        printv(
            f"Enforcing monotonicity on {n_inv:,} of {len(node_dist) - n_obs:,} merges.",
            verbose=verbose,
        )
        node_dist[n_obs:] = np.maximum.accumulate(node_dist[n_obs:])
    return node_dist


def _extract_from_tree(
    tree,
    labels,
    eval_func=EqualBalance(),
    max_dist=None,
    min_dist=None,
    min_dist_diff=None,
    verbose=False,
):
    """Run the homogeneity walk over a dendrogram given as flat arrays.

    This is the tree-agnostic core shared by all the `extract_*_clusters`
    entrypoints: it does not care whether the dendrogram came from a scipy
    linkage or from a KNN-constrained agglomeration.

    Parameters
    ----------
    tree :          tuple
                    `(children, parent, node_dist, n_obs)`, as returned by
                    `_dendrogram_arrays`, `knn_tree` or `graph_tree`.
    labels :        np.ndarray
                    Labels for each observation, in the same order as the leafs.
    eval_func/max_dist/min_dist/min_dist_diff/verbose
                    See `extract_homogeneous_clusters`.

    Returns
    -------
    cl :            np.ndarray
                    Cluster membership for each observation, in leaf order.

    """
    children, parent, node_dist, n_obs = tree

    if len(labels) != n_obs:
        raise ValueError(f"Got {len(labels)} labels for {n_obs} observations.")

    # Encode the labels as integer codes. `np.unique` sorts, so counting into
    # these codes gives us per-cluster counts in the same (sorted) order that
    # `np.unique(..., return_counts=True)` would.
    uniq, codes = np.unique(labels, return_inverse=True)
    n_unique_ds = len(uniq)

    # Number of observations per label for every node in the dendrogram
    counts = _subtree_label_counts(children, codes, n_obs, n_unique_ds)

    # Resolve the calling convention once, up front, so that the walk itself
    # never has to branch and the legacy path stays exactly as it was
    if _wants_context(eval_func):
        # The root holds the composition of the whole pool being clustered -
        # that's what we compare each candidate cluster against
        pool_counts = counts[-1]
        proportions = pool_counts / pool_counts.sum()
        pool_size = int(pool_counts.sum())

        def _eval_child(node, child):
            c = counts[child]
            return eval_func(
                c,
                ClusterContext(
                    counts=c,
                    pool_counts=pool_counts,
                    proportions=proportions,
                    dataset_labels=uniq,
                    pool_size=pool_size,
                    node=int(child),
                    dist=float(node_dist[child]),
                    parent_counts=counts[node],
                ),
            )

    else:

        def _eval_child(node, child):
            c = counts[child]
            return eval_func(c[c > 0], n_unique_ds)

    # Find clusters
    clusters = {}
    _find_clusters(
        root=len(children) - 1,
        clusters=clusters,
        eval_child=_eval_child,
        children=children,
        node_dist=node_dist,
        counts=counts,
        n_obs=n_obs,
        max_dist=max_dist,
        min_dist=min_dist,
        verbose=verbose,
    )

    # Clusters are currently labels based at which hinge they were created
    # We have to renumber them
    reind = {c: i for i, c in enumerate(np.unique(list(clusters.values())))}
    clusters = {k: reind[v] for k, v in clusters.items()}

    # At this point singletons might not be assigned a cluster - we need
    # to account for that and give them a unique cluster
    n_cl = len(reind)
    for i in range(n_obs):
        if i not in clusters:
            clusters[i] = n_cl
            n_cl += 1

    cl = np.array([clusters[i] for i in range(n_obs)])

    if min_dist_diff:
        cl = _merge_similar_clusters(
            cl=cl,
            children=children,
            parent=parent,
            node_dist=node_dist,
            n_obs=n_obs,
            dist_thresh=min_dist_diff,
            verbose=verbose,
        )

    return cl


def extract_homogeneous_clusters_knn(
    X,
    labels,
    eval_func=EqualBalance(),
    max_dist=None,
    min_dist=None,
    min_dist_diff=None,
    n_neighbors=15,
    metric="cosine",
    n_components="auto",
    knn=None,
    random_state=0,
    monotonic=True,
    verbose=False,
):
    """Make clusters that contain representatives of each unique label.

    This is a scalable variant of `extract_homogeneous_clusters` that works
    directly off a feature vector and/or a KNN graph. It applies the exact same
    top-down walk, but builds the dendrogram with a neighbourhood-constrained
    Ward agglomeration instead of a full linkage, so nothing `O(N^2)` is ever
    allocated. Use it when the pairwise distance matrix would be prohibitive -
    as a rule of thumb, from a few tens of thousands of observations upwards.

    Parameters
    ----------
    X :             pd.DataFrame | np.ndarray | scipy.sparse matrix
                    Feature vector of shape (n_observations, n_features), e.g.
                    a `Clustering.vect_`. Sparse input is kept sparse for as
                    long as possible. Required even when `knn` is given: the
                    graph only constrains *which* merges are considered, the
                    merge heights always come from the features.
    labels :        np.ndarray
                    Labels for each observation, in the same order as the rows
                    of `X`.
    eval_func :     CompositionCriterion | callable
                    Decides whether a candidate cluster has an acceptable mix of
                    labels (i.e. datasets). See `extract_homogeneous_clusters`.
    min/max_dist :  float, optional
    min_dist_diff : float, optional
                    See `extract_homogeneous_clusters`. Note that these are
                    **not on the same scale** as for the distance-matrix version:
                    there, thresholds are cosine distances handed to Ward, here
                    they are Euclidean Ward heights in the (normalized and
                    possibly reduced) feature space. Expect to re-tune them; the
                    percentiles of the `node_dist` that `knn_tree` returns are a
                    good starting point.
    n_neighbors :   int
                    Number of neighbours for the KNN graph. Larger values give a
                    better-connected graph and a tree closer to unconstrained
                    Ward, at higher cost. Ignored if `knn` is provided.
    metric :        "cosine" | "euclidean"
                    Metric for the KNN graph. `"cosine"` L2-normalizes the rows
                    before the agglomeration, which makes Euclidean Ward
                    equivalent to spherical Ward (for unit vectors
                    ``||a - b||^2 == 2 * cosine_distance``).
    n_components :  "auto" | int | None
                    Dimensionality to reduce to (via truncated SVD) before the
                    agglomeration, which needs a dense array. `"auto"` reduces
                    to `SVD_COMPONENTS` only if there are more than
                    `SVD_THRESHOLD` features; `None` disables reduction.
    knn :           optional
                    Precomputed neighbourhood, which skips the KNN search. See
                    `knn_tree` for the accepted formats.
    random_state :  int, optional
                    Seed for the approximate KNN search and the SVD. Pinned by
                    default because approximate neighbours are otherwise not
                    reproducible.
    monotonic :     bool
                    Whether to enforce monotonically increasing merge heights;
                    see `knn_tree`.
    verbose :       bool

    Returns
    -------
    cl :            np.ndarray
                    Cluster membership for each observation, in the same order
                    as the rows of `X`.

    See Also
    --------
    extract_homogeneous_clusters
                    The original, which takes a distance matrix or a linkage.
    knn_tree        Builds just the dendrogram, if you want to reuse it.

    """
    return _extract_from_tree(
        knn_tree(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            n_components=n_components,
            knn=knn,
            random_state=random_state,
            monotonic=monotonic,
            verbose=verbose,
        ),
        labels=labels,
        eval_func=eval_func,
        max_dist=max_dist,
        min_dist=min_dist,
        min_dist_diff=min_dist_diff,
        verbose=verbose,
    )


def extract_homogeneous_clusters_graph(
    graph,
    labels,
    *,
    similarity,
    eval_func=EqualBalance(),
    max_dist=None,
    min_dist=None,
    min_dist_diff=None,
    weights="degree",
    monotonic=True,
    verbose=False,
):
    """Make clusters that contain representatives of each unique label.

    Works off a **weighted KNN graph alone** - no feature vector and no full
    distance matrix. Use it when the pairwise distances were computed elsewhere
    and all you have locally is the neighbourhood graph.

    The tree is built with Paris (Bonald et al. 2018), a hierarchical clustering
    algorithm designed for graphs, and then cut by the same top-down walk the
    other `extract_homogeneous_clusters*` functions use. Paris is what makes this
    work: the linkages that can be derived from a distance graph on its own are
    single linkage - which chains into one giant cluster as soon as there are
    outlier observations - and average/complete linkage, which scikit-learn only
    offers against a dense (N, N) matrix. On synthetic data with planted cell
    types plus outliers, Paris scored 0.88 adjusted Rand index where both
    single-linkage variants scored 0.00.

    Parameters
    ----------
    graph :         see `graph_tree`
                    The weighted neighbourhood. A path to a `.parquet` /
                    `.npz` artefact works too, see `read_knn_graph`.
    labels :        np.ndarray
                    Labels for each observation. Note that the graph's node
                    indices are **positional** - `labels[i]` must describe
                    node `i`. `len(labels)` also defines the number of
                    observations, since an edge list cannot reveal isolated
                    trailing nodes.
    similarity :    None | "one_minus" | "max_minus" | "inverse" | "gaussian" | callable
                    Required, keyword-only: how to turn edge *distances* into the
                    similarities Paris expects, or `None` if the graph's weights
                    already are similarities. There is deliberately no default -
                    the right choice depends on the metric your distances were
                    computed with, and guessing it silently would be worse than
                    asking. See `graph_tree`.
    eval_func :     CompositionCriterion | callable
                    Decides whether a candidate cluster has an acceptable mix of
                    labels (i.e. datasets). See `extract_homogeneous_clusters`.
    min/max_dist :  float, optional
    min_dist_diff : float, optional
                    See `extract_homogeneous_clusters`. These work, but Paris
                    heights are node-pair-sampling values on a scale unrelated
                    to your distances, so **thresholds from the other functions
                    are meaningless here**. Use `tree_heights` to calibrate them.
    weights :       "degree" | "uniform"
                    Paris' node weighting. N.B. there is no `link_method` - Paris
                    has no choice of linkage.
    monotonic :     bool
    verbose :       bool

    Returns
    -------
    cl :            np.ndarray
                    Cluster membership for each observation, in node order.

    See Also
    --------
    extract_homogeneous_clusters
                    The original, which takes a distance matrix or a linkage.
    extract_homogeneous_clusters_knn
                    Feature vector (+ optional KNN graph); keeps Ward.
    graph_tree      Builds just the dendrogram, if you want to inspect it.
    tree_heights    Calibrate `min_dist` / `max_dist` / `min_dist_diff`.

    """
    n_obs = len(labels)
    if isinstance(graph, (str, Path)):
        graph = read_knn_graph(graph, verbose=verbose)
    dists = _knn_distance_graph(graph, n_obs, verbose=verbose)

    def cut(sub_dists, sub_labels):
        return _extract_from_tree(
            _paris_tree(
                sub_dists,
                similarity=similarity,
                weights=weights,
                monotonic=monotonic,
                verbose=verbose,
            ),
            labels=sub_labels,
            eval_func=eval_func,
            max_dist=max_dist,
            min_dist=min_dist,
            min_dist_diff=min_dist_diff,
            verbose=verbose,
        )

    # Cluster each connected component on its own. Paris joins components at
    # infinite height, and those bridges would otherwise dominate the top of the
    # tree: a single isolated node makes the root's composition unacceptable and
    # stops the walk immediately, collapsing everything into one cluster. A
    # bridge between unrelated components carries no similarity information
    # anyway, so there is nothing to lose by cutting them apart first.
    n_comp, comp = connected_components(dists, directed=False)
    if n_comp == 1:
        return cut(dists, labels)

    printv(
        f"Graph has {n_comp:,} connected components - clustering each separately.",
        verbose=verbose,
    )
    # Group the nodes by component in one pass. Scanning `comp == c` inside the
    # loop instead would be O(n_comp * n_obs), which bites hard on the very
    # graphs that need this branch - thousands of isolated nodes are routine.
    order = np.argsort(comp, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(comp, minlength=n_comp))]

    cl = np.zeros(n_obs, dtype=np.int64)
    n_cl = 0
    for c in range(n_comp):
        members = order[bounds[c] : bounds[c + 1]]
        if len(members) < 2:
            # Nothing to split - an isolated node is its own cluster
            cl[members] = n_cl
            n_cl += 1
            continue
        sub = cut(dists[members][:, members], labels[members])
        cl[members] = sub + n_cl
        n_cl += int(sub.max()) + 1

    return cl


def graph_tree(
    graph,
    similarity,
    n_obs=None,
    weights="degree",
    monotonic=True,
    verbose=False,
):
    """Build a dendrogram from a weighted KNN graph alone, using Paris.

    Paris (Bonald et al. 2018, `sknetwork.hierarchy.Paris`) agglomerates nodes by
    a node-pair-sampling distance evaluated only along existing edges, so it
    never needs coordinates or a full distance matrix. Its output is a plain
    scipy linkage, which `_dendrogram_arrays` consumes unchanged.

    Parameters
    ----------
    graph :         str | pathlib.Path | scipy.sparse matrix | pd.DataFrame | tuple
                    The weighted neighbourhood. Accepted formats:
                     - a path to a `.parquet` or `.npz` artefact, see
                       `read_knn_graph`
                     - a square, sparse (N, N) matrix of distances
                     - a `(indices, distances)` tuple of (N, k) arrays
                     - a DataFrame with `nn_idx_{i}` / `nn_dist_{i}` columns
                     - a weighted edge list: an (M, 3) array, a list of
                       `(source, target, distance)` tuples, or a DataFrame with
                       `source`+`target`+one more column (also `pre`+`post`,
                       `from`+`to`)
                    Unlike `knn_tree`'s `knn` argument the weights are
                    *required* here - without a feature vector they are the only
                    signal there is. Edges are treated as directed and
                    symmetrised by taking the larger weight.

                    Observations with fewer than `k` neighbours can be padded the
                    way BigClust writes them - index `-1` and an infinite
                    distance - and those placeholder edges are dropped.
    similarity :    None | "one_minus" | "max_minus" | "inverse" | "gaussian" | callable
                    How to turn edge distances `d` into similarities. Pass `None`
                    if the graph's weights already *are* similarities (higher =
                    more alike, as Paris expects) and should be used unchanged;
                    otherwise pick the transform that matches the metric your
                    distances were computed with:
                     - ``"one_minus"``: ``1 - d``. Recovers the exact cosine (or
                       correlation) similarity, so this is the one to use for
                       cocoa's default metric. Requires ``d <= 1``.
                     - ``"max_minus"``: ``d.max() - d``. Works for any metric and
                       scored best in testing, but is data-dependent, so results
                       are not comparable between subsets of the same dataset.
                     - ``"inverse"``: ``1 / d``. Scale-free.
                     - ``"gaussian"``: ``exp(-d / median(d))``. Scale-free and
                       robust; a good default for Euclidean distances.
                     - a callable mapping the (M, ) array of edge distances to
                       an array of positive similarities.
    n_obs :         int, optional
                    Number of observations. Required for the edge list and
                    `(indices, distances)` formats if the graph might contain
                    isolated trailing nodes; inferred otherwise.
    weights :       "degree" | "uniform"
                    Passed through to `Paris`.
    monotonic :     bool
                    Paris is monotone on a connected graph but not once the graph
                    splits into components, which it bridges at arbitrary
                    heights. Since `min_dist`, `max_dist` and `min_dist_diff` all
                    assume monotone heights, take a running maximum by default.
    verbose :       bool

    Returns
    -------
    children :  (n_nodes, 2) array of node IDs; `-1` for leafs.
    parent :    (n_nodes, ) array of node IDs; `-1` for the root.
    node_dist : (n_nodes, ) array of hinge distances; `0` for leafs.
    n_obs :     int

    """
    if isinstance(graph, (str, Path)):
        graph = read_knn_graph(graph, verbose=verbose)

    if n_obs is None:
        n_obs = _infer_n_obs(graph)
    if n_obs < 2:
        raise ValueError(f"Need at least 2 observations to build a tree, got {n_obs}.")

    dists = _knn_distance_graph(graph, n_obs, verbose=verbose)

    n_connected = connected_components(dists, directed=False)[0]
    if n_connected > 1:
        warnings.warn(
            f"The graph has {n_connected} connected components; Paris bridges "
            "them with infinite-height merges, which this function pins just "
            "above the real ones. The merges near the root are therefore "
            "arbitrary. `extract_homogeneous_clusters_graph` avoids the problem "
            "by clustering each component separately."
        )

    return _paris_tree(
        dists,
        similarity=similarity,
        weights=weights,
        monotonic=monotonic,
        verbose=verbose,
    )


def _paris_tree(dists, similarity, weights="degree", monotonic=True, verbose=False):
    """Run Paris on an already-parsed sparse distance graph.

    Split out of `graph_tree` so that callers which have already normalised the
    graph - notably the per-component loop in
    `extract_homogeneous_clusters_graph` - do not pay for `_knn_distance_graph`
    (a dedup sort plus a symmetrisation) all over again on each subgraph.
    """
    try:
        from sknetwork.hierarchy import Paris
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Clustering a graph on its own requires scikit-network. Install it "
            "with `pip install scikit-network` (or `pip install cocoa[graph]`)."
        ) from None

    n_obs = dists.shape[0]
    sim = _to_similarity(dists, similarity, verbose=verbose)

    printv(
        f"Clustering a graph of {n_obs:,} nodes and {sim.nnz // 2:,} edges (Paris).",
        verbose=verbose,
    )
    # Paris returns a plain scipy linkage, so the usual description applies
    children, parent, node_dist = _dendrogram_arrays(
        Paris(weights=weights).fit_predict(sim), n_obs
    )

    # Paris bridges disconnected components with infinite-height merges. Left in,
    # they make the distance gates and `tree_heights` meaningless and turn
    # `np.diff` into NaN, so pin them just above the tallest real merge.
    bridges = ~np.isfinite(node_dist)
    if bridges.any():
        real = node_dist[~bridges]
        node_dist[bridges] = (real.max() if len(real) else 1.0) * 1.5

    if monotonic:
        node_dist = _enforce_monotonic(node_dist, n_obs, verbose=verbose)

    return children, parent, node_dist, n_obs


def _infer_n_obs(graph):
    """Best guess at the number of observations described by `graph`."""
    if issparse(graph) and graph.shape[0] == graph.shape[1]:
        return graph.shape[0]
    if isinstance(graph, pd.DataFrame) and _sorted_nn_columns(graph, "idx"):
        return len(graph)
    if isinstance(graph, (tuple, list)) and len(graph) == 2:
        return len(np.asarray(graph[0]))
    raise ValueError(
        "Cannot infer the number of observations from an edge list - isolated "
        "trailing nodes would be missed. Pass `n_obs` explicitly (or use "
        "`extract_homogeneous_clusters_graph`, which takes it from `labels`)."
    )


# Similarity transforms, see `graph_tree`. Each maps edge distances to weights.
SIMILARITY_FUNCS = {
    "one_minus": lambda d: 1.0 - d,
    "max_minus": lambda d: d.max() - d,
    "inverse": lambda d: 1.0 / np.maximum(d, _SIM_EPS),
    "gaussian": lambda d: np.exp(-d / max(np.median(d), _SIM_EPS)),
}


def _to_similarity(graph, similarity, verbose=False):
    """Convert a sparse distance graph into the similarity graph Paris wants.

    `similarity=None` means the weights already *are* similarities and are used
    as they stand.
    """
    if similarity is None:
        func, name = None, "none"
    elif callable(similarity):
        func, name = similarity, getattr(similarity, "__name__", "callable")
    else:
        name = str(similarity)
        if name not in SIMILARITY_FUNCS:
            raise ValueError(
                f'Unknown `similarity` "{name}"; expected None, a callable, or '
                f"one of {sorted(SIMILARITY_FUNCS)}."
            )
        func = SIMILARITY_FUNCS[name]
        if name == "one_minus" and graph.data.max() > 1:
            raise ValueError(
                f'similarity="one_minus" needs distances <= 1 (yours reach '
                f"{graph.data.max():.3g}). It is meant for cosine and "
                'correlation distances; try "gaussian" or "max_minus" instead.'
            )

    if func is None:
        data = graph.data.astype(np.float64, copy=False)
    else:
        data = np.asarray(func(graph.data), dtype=np.float64)
        # Only a user-supplied callable can get the shape wrong
        if data.shape != graph.data.shape:
            raise ValueError(
                f"`similarity` must return one value per edge; got "
                f"{data.shape} for {graph.data.shape} edges."
            )
    if not np.isfinite(data).all():
        raise ValueError(f'similarity="{name}" produced non-finite weights.')

    # Share `indices`/`indptr` with the input rather than copying the matrix -
    # only the data changes, and this runs once per connected component
    sim = csr_matrix((data, graph.indices, graph.indptr), shape=graph.shape)

    # Similarities of exactly zero (e.g. `1 - d` at d == 1) would either be
    # dropped as non-edges or fed to Paris as zero-weight edges - keep them as
    # negligible-but-present instead, and only prune genuinely absent edges.
    if sim.data.min() <= 0:
        n_flat = int((sim.data <= 0).sum())
        printv(
            f"Clipping {n_flat:,} non-positive similarities to {_SIM_EPS:.1e}.",
            verbose=verbose,
        )
        np.maximum(sim.data, _SIM_EPS, out=sim.data)
    else:
        sim.eliminate_zeros()
    return sim


def read_knn_graph(path, verbose=False):
    """Load a weighted KNN graph from disk.

    Parameters
    ----------
    path :  str | pathlib.Path
            Either
             - a `.parquet` table with `nn_idx_{i}` / `nn_dist_{i}` columns, the
               layout `BigClustProject` writes for `distances_type="knn"`, or
             - a `.npz` holding a square, sparse distance matrix as written by
               `scipy.sparse.save_npz`.

            N.B. neither format records the index space: `nn_idx` values are
            **positional** row offsets, so the row order here must match the
            order of the `labels` you cluster with (for a BigClust project, the
            order of `meta.parquet`). Rows with fewer than `k` neighbours are
            expected to be padded with an index of `-1` and an infinite
            distance; those entries are dropped downstream.

    Returns
    -------
    graph : (indices, distances) tuple, or a square sparse matrix
            Ready to hand to `graph_tree` / `extract_homogeneous_clusters_graph`.

    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    if path.suffix == ".npz":
        graph = sparse_load_npz(path)
        if graph.shape[0] != graph.shape[1]:
            raise ValueError(
                f"Expected a square sparse distance matrix in {path.name}, got "
                f"shape {graph.shape}."
            )
        printv(
            f"Loaded a {graph.shape[0]:,} node graph with {graph.nnz:,} edges.",
            verbose=verbose,
        )
        return graph

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        idx_cols = _sorted_nn_columns(df, "idx")
        dist_cols = _sorted_nn_columns(df, "dist")
        if not idx_cols or len(idx_cols) != len(dist_cols):
            raise ValueError(
                f"{path.name} must have matching `nn_idx_{{i}}` and "
                f"`nn_dist_{{i}}` columns; found {len(idx_cols)} index and "
                f"{len(dist_cols)} distance columns."
            )
        printv(
            f"Loaded a {len(df):,} node graph with k={len(idx_cols)}.",
            verbose=verbose,
        )
        return df[idx_cols].to_numpy(), df[dist_cols].to_numpy(float)

    raise ValueError(
        f'Expected a ".parquet" or ".npz" file, got "{path.suffix}" ({path.name}).'
    )


def tree_heights(tree, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    """Summarise a tree's merge heights, to calibrate the distance thresholds.

    `min_dist` and `max_dist` are compared against merge heights, while
    `min_dist_diff` is compared against the *difference* between a cluster's
    height and its parent's - so both distributions are reported. Useful because
    the heights of a `graph_tree` (and, to a lesser extent, a `knn_tree`) are on
    a different scale from the cosine distances the thresholds were originally
    tuned against.

    Parameters
    ----------
    tree :          tuple
                    The 4-tuple returned by `graph_tree` or `knn_tree`.
    percentiles :   iterable of float

    Returns
    -------
    pd.DataFrame
                    Indexed by percentile, with a `height` column (for
                    `min_dist` / `max_dist`) and a `height_diff` column (for
                    `min_dist_diff`).

    """
    _, _, node_dist, n_obs = tree
    heights = np.asarray(node_dist)[n_obs:]
    percentiles = list(percentiles)
    return pd.DataFrame(
        {
            "height": np.percentile(heights, percentiles),
            "height_diff": np.percentile(np.abs(np.diff(heights)), percentiles),
        },
        index=pd.Index(percentiles, name="percentile"),
    )


def knn_tree(
    X,
    n_neighbors=15,
    metric="cosine",
    n_components="auto",
    knn=None,
    random_state=0,
    monotonic=True,
    verbose=False,
):
    """Build a dendrogram from a feature vector and/or a KNN graph.

    Runs a neighbourhood-constrained Ward agglomeration: distances are only ever
    evaluated between observations that are neighbours, so the cost is
    `O(N * n_neighbors)` rather than `O(N^2)`. The result is described the same
    way `_dendrogram_arrays` describes a scipy linkage and can be handed to
    `_extract_from_tree` (or converted with `tree_to_linkage`).

    Parameters
    ----------
    X :             pd.DataFrame | np.ndarray | scipy.sparse matrix
                    Feature vector of shape (n_observations, n_features).
                    Required even when `knn` is given: Ward's merge cost is a
                    function of cluster centroids and sizes, so a neighbourhood
                    graph - even one carrying distances - cannot produce the
                    tree on its own. See the note under `knn`.
    n_neighbors :   int
    metric :        "cosine" | "euclidean"
    n_components :  "auto" | int | None
    knn :           optional
                    Precomputed neighbourhood, which skips the KNN search.
                    Accepted formats:
                     - an edge list of `(source, target)` or
                       `(source, target, distance)` rows: an (M, 2) / (M, 3)
                       array, a list of tuples, or a DataFrame with
                       `source`+`target` (or `pre`+`post`, or `from`+`to`)
                       columns. Edges are treated as directed and symmetrised.
                     - a `(indices, distances)` tuple of (N, k) arrays, as
                       returned by `pynndescent` or `NearestNeighbors.kneighbors`
                     - an (N, k) integer array of neighbour indices
                     - a square, sparse (N, N) graph
                     - a DataFrame with `nn_idx_{i}` / `nn_dist_{i}` columns, the
                       layout used by `BigClustProject`

                    Placeholder edges for missing neighbours (index `-1`
                    and/or a non-finite distance) are dropped from any of these.

                    N.B. only the neighbourhood *structure* is used; any
                    distances are ignored. They cannot substitute for `X`:
                    Ward's merge cost depends on cluster centroids and sizes,
                    not just on distances between neighbours. The linkages that
                    *can* be computed from a distance graph alone are single
                    linkage - which collapses to one giant cluster as soon as
                    cluster sizes are uneven - and average/complete linkage,
                    which scikit-learn only offers against a dense (N, N)
                    matrix, i.e. the very thing this function exists to avoid.

                    An integer (N, 2) or (N, 3) array is read as neighbour
                    indices for k=2/3, not as an edge list; use a DataFrame with
                    named columns if you mean the latter.
    random_state :  int, optional
    monotonic :     bool
                    Ward heights are not guaranteed to increase monotonically
                    when the agglomeration is constrained to a graph. Since
                    `min_dist`, `max_dist` and `min_dist_diff` all assume they
                    do, we take a running maximum over the merges by default.
                    Set to False to keep the raw heights.
    verbose :       bool

    Returns
    -------
    children :  (n_nodes, 2) array of node IDs; `-1` for leafs.
    parent :    (n_nodes, ) array of node IDs; `-1` for the root.
    node_dist : (n_nodes, ) array of hinge distances; `0` for leafs.
    n_obs :     int

    """
    from sklearn.cluster import ward_tree

    metric = str(metric).lower()
    if metric not in ("cosine", "euclidean"):
        raise ValueError(
            f'`metric` must be "cosine" or "euclidean", got "{metric}". Ward '
            "requires a Euclidean geometry; cosine works because we L2-normalize."
        )

    if X is None:
        raise ValueError(
            "`X` is required: the neighbourhood graph only constrains which "
            "merges are considered, the merge heights come from the features."
        )

    X = _as_matrix(X)

    n_obs = X.shape[0]
    if n_obs < 3:
        raise ValueError(f"Need at least 3 observations to build a tree, got {n_obs}.")

    # Neighbourhood graph. We only pass the *structure* to Ward - the merge
    # heights are computed from the features - so any distances are discarded.
    if knn is not None:
        conn = _knn_connectivity(knn, n_obs, verbose=verbose)
    else:
        k = min(n_neighbors, n_obs - 1)
        if k < n_neighbors:
            printv(
                f"Only {n_obs} observations - reducing n_neighbors to {k}.",
                verbose=verbose,
            )
        conn = _knn_connectivity(
            _knn_indices(X, k, metric, random_state, verbose), n_obs, verbose=verbose
        )

    # Geometry for the agglomeration. Normalizing makes Euclidean Ward
    # equivalent to spherical Ward, and the reduction is what keeps the dense
    # array that Ward insists on within reach.
    Xw = X
    if metric == "cosine":
        from sklearn.preprocessing import normalize

        # `astype` already hands us a private buffer, so let `normalize` work in
        # place rather than allocate a second copy of a potentially huge matrix
        Xw = X.astype(np.float32, copy=False)
        Xw = normalize(Xw, copy=Xw is X)
    Xw = _reduce_dims(Xw, n_components, random_state, verbose)
    Xw = _densify(Xw)

    printv(
        f"Running Ward on a {Xw.shape[0]:,} by {Xw.shape[1]:,} matrix "
        f"constrained to {conn.nnz:,} neighbour pairs.",
        verbose=verbose,
    )
    # N.B. we deliberately ignore sklearn's `parent` output: it sets the root's
    # parent to itself, whereas the rest of this module expects `-1`
    ward_children, n_connected, _, _, dists = ward_tree(
        Xw, connectivity=conn, return_distance=True
    )

    if n_connected > 1:
        warnings.warn(
            f"The KNN graph has {n_connected} connected components; scikit-learn "
            "bridged them to produce a single tree, which makes the merges near "
            "the root somewhat arbitrary. Consider increasing `n_neighbors`."
        )

    # sklearn already numbers nodes the way scipy does - leafs `0` to `n_obs - 1`,
    # hinge `i` is node `n_obs + i` - and guarantees a hinge outranks its children,
    # so the linkage's own description works unchanged
    children, parent, node_dist = _tree_arrays(ward_children, dists, n_obs)

    if monotonic:
        node_dist = _enforce_monotonic(node_dist, n_obs, verbose=verbose)

    return children, parent, node_dist, n_obs


def _as_matrix(X):
    """Coerce `X` to a numpy array or a sparse matrix, keeping it sparse."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    if issparse(X):
        # Keep sparse arrays and matrices apart - pynndescent only handles the latter
        return csr_matrix(X)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2d feature vector, got shape {X.shape}.")
    return X


def _knn_indices(X, n_neighbors, metric, random_state, verbose):
    """Find the `n_neighbors` nearest neighbours of each row of `X`.

    Uses `pynndescent` if available - it is approximate but scales to millions of
    observations and handles sparse input natively - and falls back to an exact
    (and quadratic) scikit-learn search otherwise.
    """
    # N.B. don't cast to float here - both backends validate their input into a
    # C-ordered float32 buffer anyway, and pre-casting just makes them copy twice
    try:
        from pynndescent import NNDescent
    except ModuleNotFoundError:
        from sklearn.neighbors import NearestNeighbors

        warnings.warn(
            "`pynndescent` is not installed - falling back to an exact "
            "scikit-learn neighbour search, which is quadratic and will be slow "
            "beyond a few tens of thousands of observations. Install it with "
            "`pip install pynndescent`."
        )
        printv(f"Searching for {n_neighbors} nearest neighbours...", verbose=verbose)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric).fit(X)
        return nn.kneighbors(return_distance=False)

    printv(
        f"Searching for {n_neighbors} approximate nearest neighbours...",
        verbose=verbose,
    )
    # N.B. we ask for one extra neighbour because the first hit is the point
    # itself; the self-loops are dropped when we build the graph either way
    index = NNDescent(
        X,
        metric=metric,
        n_neighbors=n_neighbors + 1,
        random_state=random_state,
        verbose=verbose,
    )
    return index.neighbor_graph[0]


# Column name pairs accepted for an edge list, in order of preference
EDGE_LIST_COLUMNS = (("source", "target"), ("pre", "post"), ("from", "to"))


def _check_indices(idx, n_obs, what):
    """Validate that `idx` holds usable node indices."""
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"{what} must be integers, got {idx.dtype}.")
    if len(idx) and (idx.min() < 0 or idx.max() >= n_obs):
        raise ValueError(
            f"{what} must be in [0, {n_obs}), got [{idx.min()}, {idx.max()}]."
        )


def _drop_missing_edges(rows, cols, weights, verbose=False):
    """Drop the placeholder edges that stand in for missing neighbours.

    BigClust KNN tables pad observations with fewer than `k` neighbours using an
    index of -1 and an infinite distance. Any edge with a negative index or a
    non-finite weight is treated the same way and simply removed, so that a
    ragged neighbourhood behaves like the sparse graph it really is.
    """
    keep = (rows >= 0) & (cols >= 0)
    if weights is not None:
        keep &= np.isfinite(weights)

    n_drop = len(rows) - int(keep.sum())
    if not n_drop:
        return rows, cols, weights

    printv(
        f"Dropping {n_drop:,} of {len(rows):,} ({n_drop / len(rows):.1%}) edges "
        "marking missing neighbours.",
        verbose=verbose,
    )
    return rows[keep], cols[keep], None if weights is None else weights[keep]


def _sorted_nn_columns(df, kind):
    """The `nn_idx_{i}` / `nn_dist_{i}` columns of `df`, in index order."""
    return sorted(
        (c for c in df.columns if re.fullmatch(rf"nn_{kind}_\d+", str(c))),
        key=lambda c: int(str(c).rsplit("_", 1)[1]),
    )


def _as_edge_list(knn, n_obs):
    """Read `knn` as an edge list, or return None if it is some other format.

    Edge lists are `(source, target)` or `(source, target, distance)` per row.

    Returns
    -------
    rows, cols :    (M, ) int64 arrays
    weights :       (M, ) float array, or None if no distance column was given.

    """
    if isinstance(knn, pd.DataFrame):
        cols = {str(c).lower(): c for c in knn.columns}
        for a, b in EDGE_LIST_COLUMNS:
            if a in cols and b in cols:
                extra = [c for c in knn.columns if c not in (cols[a], cols[b])]
                weights = knn[extra[0]].to_numpy(float) if len(extra) == 1 else None
                return (
                    knn[cols[a]].to_numpy(),
                    knn[cols[b]].to_numpy(),
                    weights,
                )
        return None

    if issparse(knn):
        return None

    arr = np.asarray(knn)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        return None
    # An integer (n_obs, 2) or (n_obs, 3) array is ambiguous - it reads equally
    # well as an edge list or as neighbour indices for k=2/3. Keep the latter,
    # older meaning; anyone wanting the former can name their columns.
    if arr.shape[0] == n_obs and np.issubdtype(arr.dtype, np.integer):
        return None

    edges = arr[:, :2]
    if not np.issubdtype(edges.dtype, np.integer):
        rounded = np.rint(edges)
        if not np.array_equal(rounded, edges):
            raise ValueError(
                "The first two columns of an edge list must be whole-numbered "
                "node indices."
            )
        edges = rounded
    weights = arr[:, 2].astype(float) if arr.shape[1] == 3 else None
    return edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64), weights


def _parse_knn(knn, n_obs, verbose=False):
    """Read any of the accepted neighbourhood formats as a list of edges.

    The shared front end of `_knn_connectivity` (which throws the weights away)
    and `_knn_distance_graph` (which needs them). See `knn_tree` for the formats.

    Placeholder edges standing in for missing neighbours - a negative index
    and/or a non-finite distance, the way BigClust pads its KNN tables - are
    dropped from every format, see `_drop_missing_edges`.

    Returns
    -------
    rows, cols :    (M, ) int64 arrays of node indices.
    weights :       (M, ) float array of edge distances, or None if the format
                    carried none.

    """
    edges = _as_edge_list(knn, n_obs)
    if edges is not None:
        rows, cols, weights = _drop_missing_edges(*edges, verbose=verbose)
        _check_indices(rows, n_obs, "Edge list sources")
        _check_indices(cols, n_obs, "Edge list targets")
        return rows, cols, weights

    nn_dist = None
    if isinstance(knn, (tuple, list)) and len(knn) == 2:
        knn, nn_dist = np.asarray(knn[0]), np.asarray(knn[1])
    elif isinstance(knn, pd.DataFrame):
        # BigClust's layout, see `BigClustProject`. Matched here rather than
        # imported from `.bigclust` to keep this module below it in the stack.
        idx_cols = _sorted_nn_columns(knn, "idx")
        if not idx_cols:
            raise ValueError(
                "A `knn` DataFrame must have either `nn_idx_{i}` columns holding "
                "the neighbour indices, or a pair of edge list columns "
                f"({' / '.join('+'.join(p) for p in EDGE_LIST_COLUMNS)}); got "
                f"{list(knn.columns)[:5]}."
            )
        dist_cols = _sorted_nn_columns(knn, "dist")
        if len(dist_cols) == len(idx_cols):
            nn_dist = knn[dist_cols].to_numpy(float)
        knn = knn[idx_cols].to_numpy()

    if issparse(knn) and knn.shape[0] == knn.shape[1]:
        if knn.shape[0] != n_obs:
            raise ValueError(
                f"Graph describes {knn.shape[0]:,} observations but {n_obs:,} were "
                "expected (from `n_obs` or `len(labels)`). Note that node indices "
                "are positional, so the graph's row order must match `labels`."
            )
        # Drop explicit zeros so they aren't mistaken for edges
        knn = csr_matrix(knn)
        knn.eliminate_zeros()
        graph = knn.tocoo()
        return _drop_missing_edges(
            graph.row.astype(np.int64, copy=False),
            graph.col.astype(np.int64, copy=False),
            graph.data.astype(float, copy=False),
            verbose=verbose,
        )

    nn_idx = np.asarray(knn)
    if nn_idx.ndim != 2 or nn_idx.shape[0] != n_obs:
        raise ValueError(
            f"Expected neighbour indices of shape ({n_obs}, k), got {nn_idx.shape}."
        )
    # Check the dtype before we cast, then let `_drop_missing_edges` deal with
    # the -1 padding so only genuinely out-of-range indices raise below
    if not np.issubdtype(nn_idx.dtype, np.integer):
        raise ValueError(f"Neighbour indices must be integers, got {nn_idx.dtype}.")

    if nn_dist is not None and nn_dist.shape != nn_idx.shape:
        raise ValueError(
            f"Neighbour distances of shape {nn_dist.shape} do not match the "
            f"indices of shape {nn_idx.shape}."
        )

    rows, cols, weights = _drop_missing_edges(
        np.repeat(np.arange(n_obs, dtype=np.int64), nn_idx.shape[1]),
        nn_idx.ravel().astype(np.int64),
        None if nn_dist is None else nn_dist.ravel().astype(float, copy=False),
        verbose=verbose,
    )
    _check_indices(cols, n_obs, "Neighbour indices")
    return rows, cols, weights


def _knn_connectivity(knn, n_obs, verbose=False):
    """Turn a neighbourhood into a symmetric, binary connectivity graph.

    See `knn_tree` for the accepted formats. Only the structure survives - Ward
    computes its own distances from the features.
    """
    rows, cols, _ = _parse_knn(knn, n_obs, verbose=verbose)

    # Self-loops would make a node its own merge candidate. Filtering them on the
    # triplets is cheaper than `setdiag(False)` + `eliminate_zeros()` afterwards.
    keep = rows != cols
    conn = coo_matrix(
        (np.ones(int(keep.sum()), dtype=bool), (rows[keep], cols[keep])),
        shape=(n_obs, n_obs),
    ).tocsr()

    # KNN is not a symmetric relation, but the agglomeration needs it to be
    return conn.maximum(conn.T)


def _knn_distance_graph(knn, n_obs, verbose=False):
    """Turn a *weighted* neighbourhood into a symmetric sparse distance graph.

    Unlike `_knn_connectivity` this keeps the distances - they are the only
    signal available when there is no feature vector to fall back on. Edges
    marking missing neighbours (index -1 and/or an infinite distance) are
    dropped by `_parse_knn` before we get here.
    """
    rows, cols, weights = _parse_knn(knn, n_obs, verbose=verbose)
    if weights is None:
        raise ValueError(
            "This neighbourhood carries no distances, but clustering a graph on "
            "its own needs them. Pass e.g. a `(indices, distances)` pair, a "
            "weighted (M, 3) edge list, a `nn_idx_{i}`/`nn_dist_{i}` table or a "
            "square sparse distance matrix."
        )
    # N.B. a negative distance is a broken value rather than a missing-neighbour
    # marker, so unlike -1/inf it is an error rather than something to drop
    if len(weights) and weights.min() < 0:
        raise ValueError("Edge distances must not be negative.")
    if not len(rows):
        raise ValueError(
            "The neighbourhood has no usable edges - every one of them was "
            "dropped as a missing neighbour (index -1 / infinite distance)."
        )

    # Drop self-loops before they can become zero-distance edges
    keep = rows != cols
    rows, cols, weights = rows[keep], cols[keep], weights[keep]

    # Collapse duplicate edges. This matters: `coo_matrix.tocsr()` *sums*
    # duplicates, which would silently add distances together. For a metric the
    # duplicates agree anyway, so taking the maximum just picks that value while
    # staying consistent with how we symmetrise below.
    # A single combined key lets numpy use its integer radix sort, which is
    # cheaper than `lexsort`'s two stable argsorts over the same data.
    key = rows * np.int64(n_obs) + cols
    order = np.argsort(key, kind="stable")
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    rows, cols = rows[order], cols[order]
    if len(starts) < len(rows):
        weights = np.maximum.reduceat(weights[order], starts)
        rows, cols = rows[starts], cols[starts]
    else:
        weights = weights[order]

    graph = coo_matrix((weights, (rows, cols)), shape=(n_obs, n_obs)).tocsr()

    # KNN is not a symmetric relation. `maximum` is the right symmetriser: for a
    # metric d(i, j) == d(j, i), so where both directions exist they agree, and
    # where only one does its distance is carried over.
    return graph.maximum(graph.T)


def _reduce_dims(X, n_components, random_state, verbose):
    """Reduce `X` to `n_components` dimensions with a truncated SVD."""
    if n_components is None:
        return X

    n_feat = X.shape[1]
    max_comp = min(n_feat, X.shape[0]) - 1

    if n_components == "auto":
        if n_feat <= SVD_THRESHOLD:
            return X
        n_components = min(SVD_COMPONENTS, max_comp)
    else:
        n_components = min(int(n_components), max_comp)

    from sklearn.decomposition import TruncatedSVD

    printv(
        f"Reducing {n_feat:,} features to {n_components} components (SVD).",
        verbose=verbose,
    )
    return TruncatedSVD(
        n_components=n_components, random_state=random_state
    ).fit_transform(X)


def _densify(X):
    """Make `X` the dense array that Ward requires.

    N.B. we deliberately do *not* upcast to float64: Ward accumulates into its
    own `(2 * n_obs - 1, n_features)` float64 buffer and upcasts on assignment,
    so casting here would only add a full-size temporary. That buffer, not the
    input, is what the limit below is really about.
    """
    n_rows, n_feat = X.shape
    # Ward's own float64 accumulator dominates; a dense input is reused in place,
    # so only a sparse one adds the cost of being materialised
    n_bytes = (2 * n_rows - 1) * n_feat * 8
    if issparse(X):
        n_bytes += n_rows * n_feat * X.dtype.itemsize
    if n_bytes > DENSE_LIMIT_GB * 1e9:
        raise MemoryError(
            f"Agglomerating a {n_rows:,} by {n_feat:,} matrix would need about "
            f"{n_bytes / 1e9:.1f} GB (limit is {DENSE_LIMIT_GB} GB). Reduce "
            "dimensionality by passing e.g. `n_components=200`, or raise "
            "`cocoa.cluster_utils.DENSE_LIMIT_GB`."
        )
    return X.toarray() if issparse(X) else X


def tree_to_linkage(tree):
    """Convert a tree back into a scipy linkage.

    Useful for feeding a `knn_tree` or `graph_tree` into scipy's `dendrogram`,
    `cut_tree`, etc. Note that scipy assumes monotone heights, so build the tree
    with `monotonic=True` if you intend to plot it.

    Parameters
    ----------
    tree :  tuple
            `(children, parent, node_dist, n_obs)`, as returned by `knn_tree`
            or `graph_tree`.

    Returns
    -------
    Z :     (n_obs - 1, 4) np.ndarray

    """
    children, _, node_dist, n_obs = tree

    sizes = np.zeros(len(children), dtype=np.float64)
    sizes[:n_obs] = 1
    for i in range(n_obs, len(children)):
        left, right = children[i]
        sizes[i] = sizes[left] + sizes[right]

    return np.column_stack(
        [
            children[n_obs:, 0].astype(np.float64),
            children[n_obs:, 1].astype(np.float64),
            node_dist[n_obs:],
            sizes[n_obs:],
        ]
    )


def _is_linkage(x):
    """Check whether `x` is a linkage rather than a distance matrix."""
    if isinstance(x, pd.DataFrame):
        x = x.values
    x = np.asarray(x)

    # Linkages are (N, 4); distance matrices are square
    if x.ndim != 2 or x.shape[1] != 4:
        return False

    # A (4, 4) input is ambiguous - it could be a linkage for 5 observations or
    # a distance matrix for 4. Distance (and similarity) matrices are symmetric
    # with a constant diagonal, linkages effectively never are.
    if x.shape[0] == x.shape[1]:
        diag = np.diag(x)
        if np.allclose(x, x.T) and np.allclose(diag, diag[0]):
            return False

    return True


def _dendrogram_arrays(Z, n_obs):
    """Describe a linkage as flat arrays.

    Nodes `0` to `n_obs - 1` are the leafs, `n_obs` onwards the hinges. Because
    a hinge always has a higher ID than the two nodes it joins, the last node is
    the root and any subtree's root is its highest node.

    Returns
    -------
    children :  (n_nodes, 2) array of node IDs; `-1` for leafs.
    parent :    (n_nodes, ) array of node IDs; `-1` for the root.
    node_dist : (n_nodes, ) array of hinge distances; `0` for leafs.

    """
    return _tree_arrays(Z[:, :2].astype(np.int64), Z[:, 2], n_obs)


def _tree_arrays(pairs, heights, n_obs):
    """Describe a sequence of merges as flat arrays.

    The shared back end of `_dendrogram_arrays` and `knn_tree`: `pairs[i]` are
    the two nodes joined by hinge `n_obs + i` and `heights[i]` is the distance
    at which they were joined. Keeping the node-numbering invariant in one place
    means the two tree sources can never drift apart.

    Returns
    -------
    children, parent, node_dist
                See `_dendrogram_arrays`.

    """
    n_nodes = 2 * n_obs - 1

    children = np.full((n_nodes, 2), -1, dtype=np.int64)
    children[n_obs:] = pairs

    parent = np.full(n_nodes, -1, dtype=np.int64)
    hinges = np.arange(n_obs, n_nodes)
    parent[children[n_obs:, 0]] = hinges
    parent[children[n_obs:, 1]] = hinges

    node_dist = np.zeros(n_nodes, dtype=np.float64)
    node_dist[n_obs:] = heights

    return children, parent, node_dist


def _subtree_label_counts(children, codes, n_obs, n_unique):
    """Count labels below every node in the dendrogram.

    Hinges are processed in ascending order, which is guaranteed to be
    bottom-up, so each node is just the sum of its two children.
    """
    counts = np.zeros((len(children), n_unique), dtype=np.intp)
    counts[np.arange(n_obs), codes] = 1
    for i in range(n_obs, len(children)):
        left, right = children[i]
        counts[i] = counts[left] + counts[right]
    return counts


def _subtree_leafs(node, children, n_obs):
    """Collect the leafs below `node`."""
    if node < n_obs:
        return [int(node)]

    leafs, stack = [], [node]
    while stack:
        n = stack.pop()
        if n < n_obs:
            leafs.append(int(n))
        else:
            stack.extend(children[n])
    return leafs


def _find_clusters(
    root,
    clusters,
    eval_child,
    children,
    node_dist,
    counts,
    n_obs,
    max_dist=None,
    min_dist=None,
    verbose=False,
):
    """Find clusters by walking the dendrogram from `root` downwards.

    Uses an explicit stack rather than recursion: dendrograms are easily deep
    enough to exhaust Python's recursion limit.

    `eval_child(node, child)` judges whether one subtree has an acceptable
    composition; it already knows which calling convention the criterion wants.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        dist = node_dist[node]
        is_leaf = node < n_obs

        # A leaf can't be split and hence has nothing to evaluate
        kids = [] if is_leaf else list(children[node])
        # Count the number of labels (i.e. datasets) present in each subtree
        kid_counts = [counts[k] for k in kids]
        # Evaluate the composition of each subtree
        is_good = [eval_child(node, k) for k in kids]

        # Check if we should stop here
        stop = False
        # If we are below the minimum distance we have to stop
        if min_dist and (dist <= min_dist):
            stop = True
        # If one or both of the clusters are bad...
        elif not all(is_good):
            # ... and the distance between the two clusters below is not too big
            # we can stop
            if max_dist and (dist <= max_dist):
                stop = True
            elif not max_dist:
                stop = True

        if not stop:
            if not is_leaf:
                # Push right first so that the left branch is visited first
                stack.extend(children[node][::-1])
        else:
            if verbose:
                print(
                    f"Found cluster of {sum([c.sum() for c in kid_counts])} at distance {dist} ({node})"
                )
            node = int(node)
            for leaf in _subtree_leafs(node, children, n_obs):
                clusters[leaf] = node


def _lca(nodes, parent):
    """Lowest common ancestor of `nodes`.

    A hinge always has a higher ID than the nodes it joins, so repeatedly
    lifting the lowest node to its parent converges on their common ancestor.
    """
    pending = {int(n) for n in nodes}
    heap = sorted(pending)
    heapq.heapify(heap)
    while len(pending) > 1:
        node = heapq.heappop(heap)
        pending.discard(node)
        # The root is never popped while others remain (it has the highest ID),
        # so we can't walk off the top here
        up = int(parent[node])
        if up not in pending:
            pending.add(up)
            heapq.heappush(heap, up)
    return pending.pop()


def _leftmost_leaf(node, children, n_obs):
    """First leaf below `node`, descending via the first child throughout."""
    while node >= n_obs:
        node = children[node][0]
    return int(node)


def _merge_similar_clusters(
    cl, children, parent, node_dist, n_obs, dist_thresh, verbose=False
):
    """Merge similar clusters.

    Parameters
    ----------
    cl :        np.ndarray
                Clusters membership that is to be checked.
    children/parent/node_dist/n_obs
                The dendrogram, see `_dendrogram_arrays`.
    dist_thresh : float
                Distance under which to merge clusters.

    Returns
    -------
    cl :        np.ndarray
                Fixed cluster membership.

    """
    # Group the leafs by cluster in one pass instead of masking `cl` per cluster
    order = np.argsort(cl, kind="stable")
    breaks = np.flatnonzero(np.r_[True, np.diff(cl[order]) != 0, True])

    to_merge = []
    for i in range(len(breaks) - 1):
        members = order[breaks[i] : breaks[i + 1]]
        c1 = cl[members[0]]

        # Single-leaf clusters span no part of the dendrogram - skip them
        if len(members) < 2:
            continue

        # The root for this cluster is the lowest common ancestor of its leafs
        root = _lca(members, parent)

        dist_c1 = node_dist[root]

        # The cluster one above this one
        top = parent[root]
        if top < 0:  # `root` is the root of the whole dendrogram
            continue

        # Distance between our original cluster and the closest
        dist_top = node_dist[top]

        # Distance for the neighbouring cluster, i.e. the sibling of `root`
        left, right = children[top]
        other = right if left == root else left
        dist_c2 = node_dist[other]

        # If merging this and the next cluster are very similar
        if (dist_top - dist_c1) <= dist_thresh and (dist_top - dist_c2) < dist_thresh:
            # Get the index of the other cluster
            c2 = cl[_leftmost_leaf(other, children, n_obs)]

            if verbose:
                print(
                    f"Merging {c1} and {c2} (top={dist_top}; left={dist_c1}, right={dist_c2}"
                )

            to_merge.append([c1, c2])

    # Deduplicate
    to_merge = list(set([tuple(sorted(p)) for p in to_merge]))

    cl2 = cl.copy()
    for p in to_merge:
        cl2[cl2 == p[1]] = p[0]

    return cl2


def bootstrap_prob(
    data,
    method="ward",
    metric="cosine",
    n_boot=1000,
    r=(0.5, 1.4, 0.1),
    leaf_prob=False,
    seed=None,
    parallel=False,
    progress=True,
):
    """Calculate the bootstrap probability for each cluster.

    We do this by resampling the data (see `r`), calculating the linkage and then testing
    for each cluster in the original linkage if it is present in the bootstrapped linkage.

    Parameters
    ----------
    data :      (M, N) np.ndarray
                Observations to cluster on.
    method :    str
                Linkage method to use.
    metric :    "cosine" | "Euclidean"
                Distance metric to use.
    n_boot :    int
                Number of bootstrap iterations.
    r :         (start, stop, stepsize) tuple
                Range of multiscale bootstrap samples to use.
    seed :      int, optional
                Random seed.
    parallel :  bool | int
                Use parallel processing. If `True` will use all available cores.
                If an integer is given, will use that many cores.
    progress :  bool
                Show progress bar.

    Returns
    -------
    bp :        np.ndarray
                For each cluster in the full linkage, the fraction of boostrapped samples in which
                the exact same cluster is present.
    bp_fuzz :   np.ndarray
                For each cluster in the full linkage, the average fraction of neurons in that cluster
                that end up in the closest matching boostrapped cluster. This metric is more forgiving
                than `bp`.
    bp_leafs :  np.ndarray, optional
                Only if `leaf_prob` is True: For each leaf in the full linkage, the average fraction of
                times this leafs ended up in the "correct" cluster (according to `bp_fuzz`) during the
                bootstrapping.

    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    assert isinstance(data, np.ndarray)

    assert metric in ("cosine", "Euclidean")
    assert isinstance(r, tuple) and len(r) == 3

    # Calculate the original linkage
    dists = calculate_distance(data, metric=metric, verbose=False)
    Z = sch.linkage(squareform(dists), method=method, metric=metric)

    # Generate the graph
    G = tg.utils.linkage_to_graph(Z)

    # Construct a sparse boolean matrix where each row is a cluster and each column is an original observation
    # We will need dense matrices for the pairwise distance calculations but sparse matrices will make it
    # cheaper to send the data to the workers
    cluster_mat = _cluster_matrix(G, data.shape[0], sparse=True)

    # Now start the bootstrapping
    rng = np.random.default_rng(seed)

    # Bootstrap
    results = []
    results_leafs = []
    if not parallel:
        cluster_mat = cluster_mat.todense()
        for frac in tqdm(
            np.arange(r[0], r[1], r[2]), disable=not progress, desc="Bootstrapping"
        ):
            size = max(int(frac * data.shape[1]), 1)
            for i in range(n_boot):
                # Sample the array along the second axis
                sample = rng.choice(data.T, size=size, replace=size > data.shape[1]).T

                # Calculate the distance matrix
                dists_boot = calculate_distance(sample, metric=metric, verbose=False)

                # Calculate the linkage
                Z_boot = sch.linkage(
                    squareform(dists_boot), method=method, metric=metric
                )

                # Generate the graph
                G_boot = tg.utils.linkage_to_graph(Z_boot)

                # Construct cluster x leaf matrix
                cluster_mat_boot = _cluster_matrix(G_boot, data.shape[0], sparse=False)

                # Calculate distance between original and bootstrapped cluster dist
                mat_dist = cdist(cluster_mat, cluster_mat_boot, metric="euclidean")

                # The minimum distance tells us how many clusters have an exact match:
                # if the distance is zero, there is an identical clusters are identical
                # if the distance is non-zero, there is no identical cluster
                mat_dist_min = mat_dist.min(axis=1)

                # If asked for, also calculate the probability of a leaf being in the same clusters
                if leaf_prob:
                    mat_dist_arg_min = np.argmin(mat_dist, axis=1)
                    best_cluster_match = cluster_mat_boot[mat_dist_arg_min]
                    bf_leafs = (cluster_mat & best_cluster_match).sum(axis=0)
                    results_leafs.append(bf_leafs)

                results.append(mat_dist_min)
    else:

        parallel = None if parallel is True else parallel
        indices = np.arange(data.shape[1])
        with SharedMemoryManager() as ssm:
            # Generate shared memory object
            data_shared = ssm.SharedMemory(data.nbytes)

            # Copy data to shared memory object
            data_shared_buf = np.ndarray(
                data.shape, dtype=data.dtype, buffer=data_shared.buf
            )
            data_shared_buf[:] = data

            # N.B. we're initialising each worker with the shared memory object
            # That way, we only have to send it over the wire once
            with Pool(
                parallel,
                initializer=_make_data_global,
                initargs=(data.shape, data.dtype, data_shared.name),
            ) as pool:
                for frac in tqdm(
                    np.arange(r[0], r[1], r[2]),
                    disable=not progress,
                    desc="Bootstrapping",
                ):
                    size = max(int(frac * data.shape[1]), 1)
                    jobs = []
                    for i in range(n_boot):
                        # Sample the array along the second axis
                        sample_ix = rng.choice(
                            indices, size=size, replace=size > data.shape[1]
                        ).T

                        # Submit task
                        jobs.append(
                            pool.apply_async(
                                _bootstrap_prob_worker_parallel,
                                [],
                                dict(
                                    sample_ix=sample_ix,
                                    metric=metric,
                                    method=method,
                                    cluster_mat=cluster_mat,
                                    leaf_prob=leaf_prob,
                                ),
                            )
                        )

                    # Fill results
                    for i, res in enumerate(jobs):
                        if not leaf_prob:
                            results.append(res.get())
                        else:
                            r, bf_leafs = res.get()
                            results.append(r)
                            results_leafs.append(bf_leafs)

    # Calculate the bootstrap probability
    bp = (np.array(results) == 0).mean(axis=0)
    # Calculate and normalize the bootstrap distance
    bp_fuzz = (cluster_mat.sum(axis=1) - np.mean(results, axis=0)) / cluster_mat.sum(
        axis=1
    )

    # If we didn't calculate the leaf probabilities, we can stop here
    if not leaf_prob:
        return bp, bp_fuzz

    # Calculate and normalize the leaf probabilities
    bp_leafs = np.mean(results_leafs, axis=0) / cluster_mat.sum(axis=0)
    return bp, bp_fuzz, bp_leafs


def _bootstrap_prob_worker_parallel(sample_ix, metric, method, cluster_mat, leaf_prob):
    """Worker for parallel bootstrap probability calculation."""
    # `data` is a global variable (see worker initialization)
    # Here, we are subsetting the data to the indices we were told to sample
    sample = data[sample_ix].T

    # Calculate the distance matrix
    dists_boot = calculate_distance(sample, metric=metric, verbose=False)

    # Calculate the linkage
    Z_boot = sch.linkage(squareform(dists_boot), method=method, metric=metric)

    # Generate the graph
    G_boot = tg.utils.linkage_to_graph(Z_boot)

    # Construct cluster x leaf matrix
    cluster_mat_boot = _cluster_matrix(G_boot, data.shape[1], sparse=False)

    # Calculate distance between original and bootstrapped cluster dist
    cluster_mat = cluster_mat.todense()
    mat_dist = cdist(cluster_mat, cluster_mat_boot, metric="euclidean")

    # The minimum distance tells us how many clusters have an exact match:
    # if the distance is zero, there is an identical clusters are identical
    # if the distance is non-zero, there is no identical cluster
    mat_dist_min = mat_dist.min(axis=1)

    if not leaf_prob:
        return mat_dist_min

    # If asked for, also calculate the probability of a leaf being in the same clusters
    mat_dist_arg_min = np.argmin(mat_dist, axis=1)
    best_cluster_match = cluster_mat_boot[mat_dist_arg_min]
    bf_leafs = (cluster_mat & best_cluster_match).sum(axis=0)

    return mat_dist_min, bf_leafs


def _cluster_matrix(G, n_org, sparse=True):
    """Generate a cluster x leaf matrix.

    Parameters
    ----------
    G :         nx.DiGraph
                Graph representing the linkage.
    n_org :     int
                Number of original observations.
    sparse :    bool
                Whether to return a sparse or dense matrix.

    Returns
    -------
    cluster_mat :   np.ndarray | scipy.sparse.coo_matrix
                    Matrix where each row is a cluster and each column is an original observation.

    """
    # Construct a sparse boolean matrix where each row is a cluster
    # and each column is an original observation
    rows = []
    cols = []
    vals = []
    for node, path_lengths in nx.shortest_path_length(G):
        # Skip if this is an original observation
        if node < n_org:
            continue
        # Track
        org_obs = [n for n in path_lengths if n < n_org]
        rows.extend([node - n_org] * len(org_obs))
        cols.extend(org_obs)
        vals.extend([True] * len(org_obs))
    cluster_mat = coo_array((vals, (rows, cols)), shape=(len(G) - n_org, n_org))

    if not sparse:
        cluster_mat = cluster_mat.todense()

    return cluster_mat


def _make_data_global(shape, dtype, buf_name):
    # N.B. we need to make both the data and the shared memory object global
    global data, shm
    shm = shared_memory.SharedMemory(name=buf_name)
    data = np.ndarray(shape, dtype=dtype, buffer=shm.buf).T


def add_hinge_labels(R, Z, labels, ax=None, **text_kwargs):
    """Add labels to the dendrogram

    Parameters
    ----------
    R :         dict
                Output of scipy's dendrogram: a dictionary of data structures
                computed to render the dendrogram.
    Z :         np.ndarray
                Linkage matrix.
    labels :    iterable
                Labels to add to the dendrogram. Must be in order of the
                original linkage.
    ax :        matplotlib.axes.Axes, optional
                Axes to add the labels to. If not provided will get the current
                axes.
    **text_kwargs
                Additional keyword arguments to pass to `ax.text`.

    """
    assert isinstance(R, dict)
    assert len(R["dcoord"]) == len(labels)

    if ax is None:
        ax = plt.gca()

    assert isinstance(ax, plt.Axes)

    # The dendrogram will contain the coordinates for each element as "icoord" and "dcoord"
    # However, the order of the elements in the dendrogram is not the same as the order of the
    # linkage (and hence the labels). Hence, we have to map the coordinates in the dendrogram
    # back to the index into the original linkage.
    index2coord = map_linkage_to_dendrogram(R, Z)

    default_args = dict(
        verticalalignment="bottom",
        horizontalalignment="center",
        clip_on=True,
        size=6,
    )
    default_args.update(text_kwargs)

    # Add the labels
    for i, label in enumerate(labels):
        x, y = index2coord[i]
        ax.text(x, y, label, **default_args)


def map_linkage_to_dendrogram(R, Z):
    """Map linkage to coordinates dendrogram.

    Parameters
    ----------
    R :     dict
            Output of scipy's dendrogram: a dictionary of data structures
            computed to render the dendrogram.
    Z :     np.ndarray
            Linkage matrix.

    Returns
    -------
    coords :   np.array
               Mapping from the index in the linkage to the (x, y) coordinates
               in the dendrogram.

    """
    G = nx.DiGraph()
    edges = []
    for i, (ico, dco) in enumerate(zip(R["icoord"], R["dcoord"])):
        # Each of these elements connects two sources and one target
        # In the first instance, we will use the coordinates to track nodes
        s1 = (float(ico[0]), float(dco[0]))
        s2 = (float(ico[2]), float(dco[-1]))
        t = (float(ico[0] + (ico[-1] - ico[0]) / 2), float(dco[1]))
        edges += [(t, s1), (t, s2)]
    G.add_edges_from(edges)
    # root_dend = [n for n, d in G.in_degree() if d == 0][0]

    # Add coordinates as node attributes
    nx.set_node_attributes(G, {n: {"pos": n} for n in G.nodes})

    # Next, we will label the nodes according to where they show up in the linkage
    # (i.e. the index of the node in the linkage)
    index2coord = {}

    # First, we will need to find the leaves as anchor points
    for i, leaf in enumerate(R["leaves"]):
        index2coord[leaf] = (5 + i * 10, 0)

    # Now we can simply use the linkage to find the rest
    G_link = tg.utils.linkage_to_graph(Z)
    # root_link = max(G_link.nodes)

    # Go over the leaf nodes
    for node in list(index2coord):
        while True:
            try:
                parent_link = next(G_link.predecessors(node))
                parent_dend = next(G.predecessors(index2coord[node]))

                # We can stop if we already mapped this node
                if parent_link in index2coord:
                    break

                index2coord[parent_link] = parent_dend

                node = parent_link
            except StopIteration:
                break

    # At this point index2coord contains leaf positions - let's drop them and
    # convert to numpy array
    coords = np.array([index2coord[i + (len(Z) + 1)] for i in range(len(Z))])

    return coords
