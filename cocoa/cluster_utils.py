import networkx as nx
import tanglegram as tg
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform, cdist
from tqdm.auto import tqdm
from multiprocessing import Pool, shared_memory
from multiprocessing.managers import SharedMemoryManager
from scipy.sparse import coo_array

from .distance import calculate_distance


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
    linkage=None,
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
                    Consider two homogenous clusters that are adjacent to each
                    other in the dendrogram: if the difference in distance
                    between the two clusters and their supercluster is smaller
                    than `min_dist_diff` they will be merged. Or in other words:
                    we will merge if the three horizontal lines in the dendrograms
                    are closer together than `min_dist_diff`.
    link_method :   str
                    Method to use for generating the linkage.
    linkage :       np.ndarray
                    Precomputed linkage. If this is given `link_method` is ignored.

    Returns
    -------
    cl :        np.ndarray

    """
    if dists.values[0, 0] >= 0.999:
        dists = 1 - dists

    # Make linkage
    if linkage is None:
        Z = sch.linkage(squareform(dists, checks=False), method=link_method)
    else:
        Z = linkage

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

    # At this point singletons might not be assigned a cluster - we need
    # to account for that and give them a unique cluster
    for i in np.arange((len(dists))):
        if i not in clusters:
            clusters[i] = len(set(clusters.values()))

    cl = np.array([clusters[i] for i in np.arange(len(dists))])

    if min_dist_diff:
        cl = _merge_similar_clusters(
            cl=cl, G=G, Z=G, dist_thresh=min_dist_diff, verbose=verbose
        )

    return cl


def _find_clusters_rec(
    G, clusters, eval_func, label_dict, max_dist=None, min_dist=None, verbose=False
):
    """Recursively find clusters."""
    if G.is_directed():
        G = G.to_undirected()

    # The root node should always be the last in the graph
    root = max(G.nodes)

    try:
        dist = G.nodes[root]["distance"]  # the distance between the two prior clusters
    except KeyError:
        # If this is a leaf-node it won't have a "distance" property
        dist = 0

    # Remove the root in this (sub)graph
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
    # If one or both of the clusters are bad...
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

        if len(sg) == 0:
            continue

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
    bp_fuzz = (cluster_mat.sum(axis=1) - np.mean(results, axis=0)) / cluster_mat.sum(axis=1)

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