"""
The Mapper generates a mapping between all the datasets involved by turning
all known type labels and synonyms into a graph and then trying to find
the minimal connected components of the graph.

The rule is this:
- for each neuron find *a* shortest path to a neuron in the other datasets (if it exists)
- keep only the shortest paths in the graph
- use the connected components of the graph to map labels between datasets

Some examples:

FlyWire left vs hemibrain:

720575940636053594 --> AOTU008a ------|
                                      v
720575940623374218 --> AOTU008b --> AOTU008
                                      ^
861237679-----------------------------|

FlyWire left vs right:

720575940643300974(L) --> AOTU008a - - - - |
                             ^             |
                             |             |
                             v             v
720575940622365991(R) --> AOTU008a - - > AOTU008
                                           ^ ^
720575940623374218(L) --> AOTU008b - - - - | |
                             ^               |
                             |               |
                             v               |
720575940629805327(R) --> AOTU008b - - - - - -

Note how we can map AOTU008a and AOTU008b between the two datasets without having
to go through AOTU008. This is because we have a direct mapping between AOTU008a and AOTU008b.


FlyWire left vs right vs hemibrain:

720575940643300974(L) --> AOTU008a --------|
                             ^             |
                             |             |
                             v             v
720575940622365991(R) --> AOTU008a ----> AOTU008 <----- 861237679
                                           ^ ^
720575940623374218(L) --> AOTU008b --------| |
                             ^               |
                             |               |
                             v               |
720575940629805327(R) --> AOTU008b -----------

In this scenario, we need to go through AOTU008 to map AOTU008a and AOTU008b between the two datasets.

Each dataset is responsible for generating its own intrinsic mappings, which are then combined
in the mapper. The mapper then trims the graph to only include the shortest paths between datasets
and uses the connected components to assign labels that map between datasets.

"""

import os
import itertools

import networkx as nx
import numpy as np
import pandas as pd

from abc import abstractmethod
from joblib import Parallel, delayed
from itertools import combinations

from .datasets.core import DataSet
from .utils import printv


# TODO:
# - add method to quickly plot/export the graph with labels & annotations
# - add `use_sides` option (basically adding e.g. _L to the types)


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance


class BaseMapper(Singleton):
    """Abstract base class for mappers."""
    _mappings = {}

    def __init__(self):
        pass

    def __repr__(self):
        s = f"{self.type} with {len(self)} cached mappings(s)"
        if len(self):
            s += ":"
            for datasets, map in self._mappings.items():
                s += f"\n  {datasets}: {len(map)} neurons"
        return s

    def __len__(self):
        return len(self._mappings)

    def __str__(self):
        return self.__repr__()

    @property
    def type(self):
        return str(type(self))[:-2].split(".")[-1]

    def validate_dataset(self, ds):
        """Validate dataset(s)."""
        if isinstance(ds, (list, tuple)):
            for d in ds:
                self.validate_dataset(d)
            labels = [d.label for d in ds]
            if len(set(labels)) != len(labels):
                raise ValueError("All datasets must have unique labels.")
        else:
            # We expect either an instance of a dataset or a dataset type
            if not isinstance(ds, DataSet) and not issubclass(ds, DataSet):
                raise TypeError(f'Expected dataset(s), got "{type(ds)}"')

    def clear_mappings(self):
        """Clear mappings."""
        self._mappings = {}

    @abstractmethod
    def build_mapping(self, verbose=True):
        """Build mappings between datasets.

        This method must:
         - minimally accept a `verbose` keyword
         - return a `{id: label}` dictionary
        """
        pass


class SimpleMapper(BaseMapper):
    """A mapper that uses simple string matching to map labels between datasets."""

    def build_mapping(self, *datasets, force_rebuild=False, verbose=True):
        """Build mappings between datasets.

        Parameters
        ----------
        *datasets
                        List of datasets to map between.
        force_rebuild : bool
                        By default, mappers cache mappings between datasets. Use this
                        parameter to ignore any cached data and force a rebuild.
        verbose :       bool
                        If True, will print progress messages.

        """
        # Validate datasets
        self.validate_dataset(datasets)

        # N.B. that we really only need one dataset per dataset type! E.g. having two
        # FlyWire datasets is redundant because they will have the same labels.
        # In the future, we could implement "hemispheres", i.e. FlyWire left vs right
        # but the assumption is that the primary labels should already be the same.
        datasets = list({type(d): d for d in datasets}.values())

        # Check if we already have a mapping for this combination of datasets
        ds_identifier = tuple(sorted([d.type for d in datasets]))
        ds_string = ", ".join([d.label for d in datasets])
        if ds_identifier in self._mappings and not force_rebuild:
            printv(
                f"Using cached across-dataset mapping for {ds_string}.",
                verbose=verbose,
                flush=True,
            )
            return self._mappings[ds_identifier]

        printv(
            f"Building across-dataset mapping for {ds_string}... ",
            end="",
            verbose=verbose,
            flush=True,
        )
        # Get labels for all datasets
        mappings = {}
        for ds in datasets:
            # Here we simply grab all available labels in the dataset
            # These are `{id: label}` dictionarys which should be equivalent
            # to the "primary" labels in the label graph.
            mappings.update(ds.get_labels(None))

        # Depending on the combination of datasets, we may have to collapse some of the labels.
        # For example, if FlyWire contains a `PS008,PS009` label, we will have to collapse
        # those two labels in the `hemibrain` (if present).
        # `collapse_types` will be a dictionary containing e.g.
        # {"PS009": "PS008,PS009", "PS008": "PS008,PS009"}
        collapse_types = {
            c: cc
            for cc in list(mappings.values())
            for c in cc.split(",")
            if ("," in cc)
        }

        # Update labels for collapsed types
        mappings = {k: collapse_types.get(v, v) for k, v in mappings.items()}
        printv("Done.", verbose=verbose, flush=True)

        self._mappings[ds_identifier] = mappings

        return self._mappings[ds_identifier]


class GraphMapper(BaseMapper):
    """A mapper using a label graph to map between datasets.

    This mapper generates a graph consisting of `id->label(s)` and `label-labels`
    edges, where the latter indicates either synonyms or less granular labels.

    We then process that graph to maximize the number of connected components
    (= maximize granularity of labels) while ensuring that we maintain a
    mapping between all involved datasets where possible.

    This relies on the individual datasets providing sensible labels graphs.
    See their `.compile_label_graph()` methods for more information.

    """
    _graphs = {}

    # TODOs:
    # - use direct mappings where available (e.g. the "hb123456" hemibrain types in FlyWire)

    def build_mapping(self, *datasets, force_rebuild=False, verbose=True):
        """Build mappings between datasets.

        Parameters
        ----------
        *datasets
                        List of datasets to map between.
        force_rebuild : bool
                        By default, mappers cache mappings between datasets. Use this
                        parameter to ignore any cached data and force a rebuild.
        verbose :       bool
                        If True, will print progress messages.

        """
        # Validate datasets
        self.validate_dataset(datasets)

        # N.B. that we really only need one dataset per dataset type! E.g. having two
        # FlyWire datasets is redundant because they will have the same labels.
        # In the future, we could implement "hemispheres", i.e. FlyWire left vs right
        # but the assumption is that the primary labels should already be the same.
        datasets = list({type(d): d for d in datasets}.values())

        # Check if we already have a mapping for this combination of datasets
        ds_identifier = tuple(sorted([d.type for d in datasets]))
        ds_string = ", ".join([d.type for d in datasets])
        if tuple(ds_identifier) in self._mappings and not force_rebuild:
            printv(
                f"Using cached across-dataset mapping for {ds_string}.",
                verbose=verbose,
                flush=True,
            )
            return self._mappings[ds_identifier]

        # If we only have one dataset, we can just go with the first label for each neuron
        # This is effectively the same as the SimpleMapper
        if len(datasets) < 2:
            printv(
                f"Building mapping for {ds_string}... ",
                end="",
                verbose=verbose,
                flush=True,
            )
            ds = datasets[0]
            # Generate a graph for the dataset - the compile_label_graph knows which are
            # the primary labels (i.e. the most granular within the dataset)
            G = ds.compile_label_graph(which_neurons="all")
            mappings = {}
            for n in ds.get_all_neurons():
                try:
                    # For each node, its first neighbor should be the primary label
                    mappings[n] = next(G.neighbors(n))
                except StopIteration:
                    pass
            self._mappings[ds_identifier] = mappings
            printv("Done.", verbose=verbose, flush=True)
            return self._mappings[ds_identifier]

        printv(
            f"Building across-dataset mapping for {ds_string}... ",
            end="",
            verbose=verbose,
            flush=True,
        )

        # Generate a graph consisting of `id->label(s)` for each dataset; these graphs
        # have a concept of primary, secondary, etc. labels where the primary label is
        # the most specific label and subsequent labels are either just synonyms or less
        # granular labels from other dataset
        graphs = []
        for ds in datasets:
            G = ds.compile_label_graph(which_neurons="all")
            # Here we set e.g. "FWR=True" so we can later identify which dataset(s)
            # a node belongs to (can be multiple!)
            nx.set_node_attributes(G, True, ds.label)
            # Also set a normal node attribute for the neurons in this dataset
            # (this is mainly for convenience when inspecting the graph)
            nx.set_node_attributes(G, {n: ds.label for n in G.nodes if G.nodes[n].get('type', None) == 'neuron'}, name='dataset')
            graphs.append(G)

        # Combine graphs
        G = nx.compose_all(graphs)

        # Update edge weights
        weights = nx.get_edge_attributes(graphs[0], "weight")
        for G2 in graphs[1:]:
            weights2 = nx.get_edge_attributes(G2, "weight")
            for n, w in weights2.items():
                weights[n] = weights.get(n, 0) + w
        nx.set_edge_attributes(G, weights, "weight")

        # Remove "neurons" from the graph, so that we only have "label" nodes
        G_labels = G.copy()
        neurons = [n for n in G.nodes if G.nodes[n].get("type", None) == "neuron"]
        G_labels.remove_nodes_from(neurons)

        # Get all shortest paths - this is reasonably fast on the labels-only graph (<1s)
        paths = nx.shortest_path(G_labels)

        # Get all primary labels (we need to keep those regardless of whether they are in the other datasets)
        all_prim = set()
        for n in neurons:
            try:
                all_prim.add(next(G.neighbors(n)))
            except StopIteration:
                pass

        # First find those primary labels that are actually present in all datasets
        # This is our initial set of nodes to keep
        keep = {
            p
            for p in all_prim
            if all(G.nodes[p].get(ds.label, False) for ds in datasets)
        }

        # Next, we need to try to walk from each primary label to a primary label in another dataset
        left = all_prim - keep
        for ds in datasets:
            # Go over all primary labels in this dataset
            for p in (p for p in left if G.nodes[p].get(ds.label, False)):
                # Get paths from this label to all other neuron
                this_paths = paths[p]

                # These are the other nodes, sorted by length of the path (shortest first)
                reachable = sorted(this_paths, key=lambda x: len(this_paths[x]))

                # Iterate over all other datasets and find the shortest path to *a* label
                # in the other dataset (if it exists)
                for ds2 in (d for d in datasets if d is not ds):
                    for p2 in reachable:
                        if G.nodes[p2].get(ds2.label, False):
                            for n in this_paths[p2]:
                                keep.add(n)
                            break

        # For debugging: `all_prim - keep` are the labels that are not present in another dataset
        # print(all_prim - keep)

        # Now subset the graph to only include the shortest paths
        G_trimmed = G_labels.subgraph(keep)

        # Generate mappings between labels
        mappings_labels = {}
        for ccn in nx.connected_components(G_trimmed.to_undirected()):

            # Note to future self:
            # We should probably introduce some weight metric when finding the paths earlier
            # Then we could use the weights within the connected component to suss out potential
            # pathological cases - i.e. where a single weight-1 edge connects two otherwise strongly
            # connected subgraphs within this connecte component.

            # Now we have to make sure this connected component is actually connected to all datasets
            missing = {ds for ds in datasets if not any(G.nodes[n].get(ds.label, False) for n in ccn)}
            if any(missing):
                continue

            # Make a new label for this connected component
            # (make sure we first split compound labels)
            new_label = ",".join(set([l for label in ccn for l in label.split(",")]))

            for l in ccn:
                mappings_labels[l] = new_label

        # Now apply the new labels to the neurons
        mappings = {}
        for n in neurons:
            try:
                p = next(G.neighbors(n))
                if p in mappings_labels:
                    mappings[n] = mappings_labels[p]
            except StopIteration:
                pass

        printv("Done.", verbose=verbose, flush=True)

        self._mappings[ds_identifier] = mappings

        # Keep a version of the graph for later inspection
        self._graphs[ds_identifier] = G.subgraph(keep | set(mappings.keys()))

        return self._mappings[ds_identifier]


def all_pairs_all_shortest_paths(
    G,
    sources=None,
    targets=None,
    weight=None,
    method="dijkstra",
    get_chunks="chunks",
    cores=None,
):
    """The parallel implementation first divides the nodes into chunks and then
    creates a generator to lazily compute all shortest paths between all nodes for
    each node in `node_chunk`, and then employs joblib's `Parallel` function to
    execute these computations in parallel across all available CPU cores.

    Parameters
    ----------
    get_chunks : str, function (default = "chunks")
        A function that takes in an iterable of all the nodes as input and returns
        an iterable `node_chunks`. The default chunking is done by slicing the
        `G.nodes` into `n` chunks, where `n` is the number of CPU cores.

    networkx.single_source_all_shortest_paths : https://github.com/networkx/networkx/blob/de85e3fe52879f819e7a7924474fc6be3994e8e4/networkx/algorithms/shortest_paths/generic.py#L606
    """

    def _process_node_chunk(node_chunk):
        # Note: the VAST majority of time is spending returning the data
        # to the parent process. The only thing we can do about this is
        # to return the amount of data we need to return
        res = [
            (
                n,
                dict(nx.shortest_path(G, source=n)),
            )
            for n in node_chunk
        ]
        # Slim down the payload
        if targets is not None:
            res = [
                (n, {t: paths.get(t, {}) for t in targets[np.isin(targets, paths)]})
                for n, paths in res
            ]

        return res

    if sources is None:
        nodes = G.nodes
    else:
        assert all(n in G.nodes for n in sources), "All sources must be in G.nodes"
        nodes = sources

    if cores is None:
        total_cores = max(1, os.cpu_count() // 2)

    if targets is not None:
        targets = np.asarray(targets)

    if get_chunks == "chunks":
        num_in_chunk = max(len(nodes) // (total_cores - 1), 1)
        node_chunks = chunks(nodes, num_in_chunk)
    else:
        node_chunks = get_chunks(nodes)

    paths_chunk_generator = (
        delayed(_process_node_chunk)(node_chunk) for node_chunk in node_chunks
    )

    for path_chunk in Parallel(n_jobs=total_cores)(paths_chunk_generator):
        for path in path_chunk:
            yield path


def chunks(iterable, n):
    it = iter(iterable)
    while True:
        x = tuple(itertools.islice(it, n))
        if not x:
            return
        yield x
