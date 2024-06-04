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

from .datasets.core import DataSet
from .utils import collapse_neuron_nodes, printv


# TODO:
# - add method to quickly plot/export the graph with labels & annotations
# - add `use_sides` option (basically adding e.g. _L to the types)
# - add some quality control functions:
#   - labels with mismatch in numbers of neurons between datasets
#   - large groups of unmatched labels


class BaseMapper:
    """Abstract base class for mappers."""

    _CACHE = {}

    def __init__(self, *datasets, verbose=True):
        self.verbose = verbose
        self._stale = True
        self.datasets = []
        self.add_dataset(*datasets)

    def __repr__(self):
        s = f"{self.type} with {len(self)} datasets"
        if len(self):
            s += ":"
            for ds in self.datasets:
                s += f"\n  {type(ds)}"
        return s

    def __len__(self):
        return len(self.datasets)

    def __str__(self):
        return self.__repr__()

    @property
    def type(self):
        return str(type(self))[:-2].split(".")[-1]

    def add_dataset(self, *datasets, skip_existing=True):
        """Add dataset(s)."""
        for ds in datasets:
            # We expect either an instance of a dataset or a dataset type
            if isinstance(ds, DataSet):
                dstype = type(ds)
            elif issubclass(ds, DataSet):
                dstype = ds
                ds = ds()  # initialise the dataset
            else:
                raise TypeError(f'Expected dataset(s), got "{type(ds)}"')
            # Check if we already have this dataset
            if dstype in [type(d) for d in self.datasets]:
                if skip_existing:
                    continue
                else:
                    # Pop the existing dataset
                    self.datasets = [d for d in self.datasets if type(d) != dstype]
                    # Mark as stale
                    self._stale = True
            self.datasets.append(ds)

        return self

    def report(self, *args, **kwargs):
        """Print a report."""
        if getattr(self, "verbose", False):
            print(*args, **kwargs)

    def get_mappings(self):
        """Get mappings. Compile if necessary."""
        if not hasattr(self, "mappings_") or self._stale:
            self.compile()
        return self.mappings_

    def get_label_counts(self, split_sides=False, label_suspicious=True):
        """Get label counts per dataset.

        Parameters
        ----------
        split_sides : bool
                      If True, will split the counts by side (if available).
        label_suspicious : bool
                        If True, will check for labels that are suspiciously different
                        between datasets.

        Returns
        -------
        counts : pd.DataFrame
                 DataFrame with label counts per dataset.

        """
        # TODOs:
        # - split by side
        if not hasattr(self, "mappings_") or self._stale:
            self.compile()

        # Invert the mappings
        labels = {l: [] for l in set(self.mappings_.values())}
        for k, v in self.mappings_.items():
            labels[v].append(k)

        counts = pd.DataFrame()
        counts["label"] = list(labels)

        for ds in self.datasets:
            if not split_sides:
                this_counts = []
                for label in counts.label.values:
                    this_counts.append(
                        len(
                            [
                                l
                                for l in labels[label]
                                if self.graph_.nodes[l].get("dataset", None) == ds.label
                            ]
                        )
                    )
                counts[ds.label] = this_counts
            else:
                # Get sides for all neurons
                sides = ds.get_sides(None)
                for side in set(sides.values()):
                    # Skip side if unclear
                    if side in (None, "", np.nan, "na"):
                        continue

                    this_counts = []
                    for label in counts.label.values:
                        this_counts.append(
                            len(
                                [
                                    l
                                    for l in labels[label]
                                    if self.graph_.nodes[l].get("dataset", None)
                                    == ds.label
                                    and sides.get(l, None) == side
                                ]
                            )
                        )
                    counts[f"{ds.label}_{side}"] = this_counts

        if label_suspicious:
            # Check for labels that are suspiciously different between datasets
            counts["suspicious"] = False
            for ds1 in self.datasets:
                ds1_cols = [c for c in counts.columns if ds1.label in c]
                for ds2 in self.datasets:
                    if ds1 == ds2:
                        continue
                    ds2_cols = [c for c in counts.columns if ds2.label in c]
                    for c1, c2 in itertools.product(ds1_cols, ds2_cols):
                        # Make sure we don't compare e.g. "center" with "left"
                        c1_is_cent = any(
                            c1.endswith(f"_{s}") for s in ("C", "M", "center")
                        )
                        c2_is_cent = any(
                            c2.endswith(f"_{s}") for s in ("C", "M", "center")
                        )
                        if c1_is_cent != c2_is_cent:
                            continue
                        ratio = counts[[c1, c2]].min(axis=1) / counts[[c1, c2]].max(
                            axis=1
                        )
                        counts.loc[ratio < 0.5, "suspicious"] = True

        return counts

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

    def clear_cache(self):
        """Clear mappings."""
        self._CACHE.clear()
        self._stale = True

    @abstractmethod
    def compile(self, verbose=True):
        """Build mappings between datasets.

        This method must:
         - minimally accept a `verbose` keyword
         - return a `{id: label}` dictionary
        """
        pass


class SimpleMapper(BaseMapper):
    """A mapper that uses simple string matching to map labels between datasets.

    Parameters
    ----------
    *datasets
                    List of datasets to map between. Alternatively, use the
                    `.add_dataset()` method to add datasets.
    strict :        bool
                    If False (default), will try to establish a mapping greedily
                    by looking for matching labels. If True, will only match labels
                    that are meant to be match - e.g. FlyWire "cell_type" to
                    maleCNS "flywire_type".
    verbose :       bool
                    If True, will print progress messages.
    """

    # Class-level cache for mappings
    _CACHE = {}

    def compile(self, force_rebuild=False):
        """Build mappings between datasets.

        Parameters
        ----------
        force_rebuild : bool
                        By default, Mappers cache mappings between datasets. Use this
                        parameter to ignore any cached data and force a rebuild. Also
                        clears the cache for the individual datasets.
        verbose :       bool
                        If True, will print progress messages.

        """
        if not self.datasets:
            raise ValueError("No datasets to map between.")

        # N.B. that we really only need one dataset per dataset type! E.g. having two
        # FlyWire datasets is redundant because they will have the same labels.
        # In the future, we could implement "hemispheres", i.e. FlyWire left vs right
        # but the assumption is that the primary labels should already be the same.
        datasets = list({type(d): d for d in self.datasets}.values())

        # Check if we already have a mapping for this combination of datasets
        ds_identifier = tuple(sorted([d.type for d in datasets]))
        ds_string = ", ".join([d.label for d in datasets])
        if ds_identifier in self._CACHE and not force_rebuild:
            self.report(
                f"Using cached across-dataset mapping for {ds_string}.",
                flush=True,
            )
            # Copy the mappings to the current instance
            other = self._CACHE[ds_identifier]
            for attr in ("mappings_",):
                setattr(self, attr, getattr(other, attr))

            return self

        # Clear dataset caches if we're forcing a rebuild
        # This will clear e.g. cached annotations
        if force_rebuild:
            for ds in self.datasets:
                ds.clear_cache()

        self.report(
            f"Building across-dataset mapping for {ds_string}... ",
            end="",
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
        self.mappings_ = {k: collapse_types.get(v, v) for k, v in mappings.items()}
        self.report("Done.", flush=True)

        self._CACHE[ds_identifier] = self
        self._stale = False
        return self


class GraphMapper(BaseMapper):
    """A mapper using a label graph to map between datasets.

    This mapper generates a graph consisting of `id->label(s)` and `label-labels`
    edges, where the latter indicates either synonyms or less granular labels.

    We then process that graph to maximize the number of connected components
    (= maximize granularity of labels) while ensuring that we maintain a
    mapping between all involved datasets where possible.

    This relies on the individual datasets providing sensible labels graphs.
    See their `.compile_label_graph()` methods for more information.

    Parameters
    ----------
    *datasets
                    List of datasets to map between. Alternatively, use the
                    `.add_dataset()` method to add datasets.
    post_process :  bool
                    If True, will attempt to remove edges from the graph that are
                    not necessary for maintaining the mapping between datasets.
                    This seems to be reasonably good at removing "accidental" edges
                    (from spurious annotations) but it may also produce mappings
                    that are imbalanced.
    strict :        bool
                    If False (default), will try to establish a mapping greedily
                    by looking for matching labels. If True, will only match labels
                    that are meant to be match - e.g. FlyWire "cell_type" to
                    maleCNS "flywire_type".
    verbose :       bool
                    If True, will print progress messages.

    """

    # Class-level cache for mappings
    _CACHE = {}

    # TODOs:
    # - use direct mappings where available (e.g. the "hb123456" hemibrain types in FlyWire)

    def __init__(self, *datasets, post_process=True, strict=False, verbose=True):
        self.post_process = post_process
        self.strict = strict
        self._synonyms = {}
        super().__init__(*datasets, verbose=verbose)

    def add_synonym(self, label, synonym):
        """Add a synonym to the mapping.

        Parameters
        ----------
        label : str
                The label to add a synonym to.
        synonym : str
                The synonym to add.

        """
        self._synonyms[label] = self._synonyms.get(label, set()) | {synonym}
        # Mark as stale
        self._stale = True

        return self

    def compile(self, force_rebuild=False):
        """Build mappings between datasets.

        Parameters
        ----------
        force_rebuild : bool
                        By default, Mappers cache mappings between datasets. Use this
                        parameter to ignore any cached data and force a rebuild. Also
                        clears the cache for the individual datasets.
        verbose :       bool
                        If True, will print progress messages.

        """
        if not self.datasets:
            raise ValueError("No datasets to map between.")

        # N.B. that we really only need one dataset per dataset type! E.g. having two
        # FlyWire datasets is redundant because they will have the same labels.
        # In the future, we could implement "hemispheres", i.e. FlyWire left vs right
        # but the assumption - right or wrong - is that the primary labels should already be the same.
        datasets = list({type(d): d for d in self.datasets}.values())

        # Check if we already have a mapping for this combination of datasets and settings
        ds_identifier = tuple(sorted([d.type for d in datasets]))
        # Build a hashable identifier for this instance
        self_identifier = (ds_identifier, self.post_process, self.strict)
        ds_string = ", ".join([d.type for d in datasets])
        if self_identifier in self._CACHE and not force_rebuild:
            self.report(
                f"Using cached across-dataset mapping for {ds_string}.",
                flush=True,
            )
            # Copy the mappings to the current instance
            other = self._CACHE[self_identifier]
            for attr in ("mappings_", "graph_", "spurious_edges_"):
                setattr(self, attr, getattr(other, attr))

            return self

        # Clear dataset caches if we're forcing a rebuild
        # This will clear e.g. cached annotations
        if force_rebuild:
            for ds in self.datasets:
                ds.clear_cache()

        # If we only have one dataset we only need to get to *a* label and that label
        # should ideally be the most granular label within reach. So for example,
        # a male CNS neuron might have type='MeTu4e' and `flywire_type='MeTu4'` in
        # which case we need to make sure to map to 'MeTu4e' and not 'MeTu4'.
        if len(datasets) < 2:
            self.report(
                f"Building mapping for {ds_string}... ",
                end="",
                flush=True,
            )
            ds = datasets[0]
            # Generate a graph for the dataset
            G = ds.compile_label_graph(which_neurons="all", strict=self.strict)
            mappings = {}
            keep_edges = set()
            for n in ds.get_all_neurons():
                # Get neighbors and sort by in-degree of the label
                # In addition we're also de-prioritizing `group` and `instance` based labels
                neighbors = sorted(
                    list(G.neighbors(n)),
                    key=lambda x: (G.in_degree[x], "group" in x or "instance" in x),
                )
                if not neighbors:
                    continue
                mappings[n] = neighbors[0]  # use the most granular label
                keep_edges.add((n, neighbors[0]))

            # Subset the graph. N.B. we're using subgraph_view to make sure we
            # get all the nodes but only a subset of the edges
            G_trimmed = nx.subgraph_view(
                G, filter_edge=lambda e1, e2: (e1, e2) in keep_edges
            ).copy()

            # Store the mappings and the graph
            self.mappings_ = mappings
            self.graph_ = G_trimmed
            self.spurious_edges_ = []  # no spurious edges in this case

            # Cache the results
            self._CACHE[self_identifier] = self

            self.report("Done.", flush=True)
            self._stale = False
            return self

        self.report(
            f"Building across-dataset mapping for {ds_string}... ",
            end="",
            flush=True,
        )

        # Generate a graph consisting of `id->label(s)` for each dataset; these graphs
        # have a concept of primary, secondary, etc. labels where the primary label is
        # the most specific label and subsequent labels are either just synonyms or less
        # granular labels from other dataset
        graphs = []
        for ds in datasets:
            G = ds.compile_label_graph(which_neurons="all", strict=self.strict)
            # Here we set e.g. "FWR=True" so we can later identify which dataset(s)
            # a node belongs to (can be multiple!)
            nx.set_node_attributes(G, True, ds.label)
            # Also set a normal node attribute for the neurons in this dataset
            # (this is mainly for convenience when inspecting the graph)
            neurons = [n for n in G.nodes if G.nodes[n].get("type", None) == "neuron"]
            nx.set_node_attributes(G, {n: ds.label for n in neurons}, name="dataset")
            # Track how many neurons from this dataset point towards a given label
            n_in = {}
            for n in neurons:
                for nn in G.neighbors(n):
                    n_in[nn] = n_in.get(nn, 0) + 1
            nx.set_node_attributes(G, n_in, name=f"{ds.label}_in")
            graphs.append(G)

        # Combine graphs
        G = nx.compose_all(graphs)

        # Add manual synonyms
        for label, synonyms in self._synonyms.items():
            for syn in synonyms:
                # Because the graph is directed we have to add the edge both ways
                if (syn, label) not in G.edges:
                    G.add_edge(syn, label, weight=0)
                if (label, syn) not in G.edges:
                    G.add_edge(label, syn, weight=0)

        # Update edge weights -> these should be the number of neurons that point to a label
        weights = nx.get_edge_attributes(graphs[0], "weight")
        for G2 in graphs[1:]:
            weights2 = nx.get_edge_attributes(G2, "weight")
            for n, w in weights2.items():
                weights[n] = weights.get(n, 0) + w
        nx.set_edge_attributes(G, weights, "weight")

        # Get all shortest path (this is faster than getting them one by one)
        paths = nx.shortest_path(G)

        # Note: DO NOT remove the `neurons = ...` here (or overwrite it elsewhere)
        # because we need it again later
        neurons = set([n for n in G.nodes if G.nodes[n].get("type", None) == "neuron"])
        keep = set()
        keep_edges = {}
        # Go over all neurons and try to get to a label also present in the other dataset
        for source in neurons:
            keep.add(source)  # Always keep the source
            targets = paths[source]  # Get all shortest paths from this source

            # Skip if there are no targets
            if (
                len(targets) == 1
            ):  # the source will always hit itself, so len == 1 means no other targets
                continue

            # Iterate over all datasets and get the shortest path to a label that is present in the other dataset
            for ds in datasets:
                # Skip if this is the source's dataset
                if G.nodes[source].get("dataset", None) == ds.label:
                    continue

                # Keep only paths to labels in the other dataset
                this_targets = {
                    t: d for t, d in targets.items() if G.nodes[t].get(ds.label, False)
                }

                # Skip if there are no targets
                if not any(this_targets):
                    continue

                # Calculate weights for each source->label path
                # We basically want to pick the paths between datasets that give us the highest granularity
                # possible. For that, we will punish paths that go through less granular labels.
                # Note to self: should this be the MAX or the SUM along the path?
                weights = {
                    t: max([G.in_degree[p] for p in d[1:]])
                    for t, d in this_targets.items()
                }

                # Get the label with the lowest weight
                target = sorted(weights, key=weights.get)[0]

                # Add all nodes in the path to the keep set
                keep.update(this_targets[target])

                for s, t in zip(this_targets[target], this_targets[target][1:]):
                    keep_edges[(s, t)] = keep_edges.get((s, t), 0) + 1

        # For debugging: `all_prim - keep` are the labels that are not present in another dataset
        # print(all_prim - keep)

        # Now subset the graph to only include the shortest paths between neurons
        G_trimmed = G.subgraph(
            keep
        ).to_undirected()  # note: do NOT remove the to_undirected here
        nx.set_edge_attributes(G_trimmed, keep_edges, "weight")

        # At this point we may still have edges in the graph that we don't actually need
        # This can happen when e.g. the malecns_type has a fine-grained setting but the
        # hemibrain_type is still a huge compound type. What we can try is this:
        # - iterate over all connnected components
        # - check all the edges in the cc for whether we can remove them without
        #   one of the new connected components losing a connection to a dataset
        self.spurious_edges_ = []
        if self.post_process:
            # Collapse neurons into groups - this should speed things up quite a lot
            G_trimmed_grp = collapse_neuron_nodes(G_trimmed).to_undirected()

            for ccn in nx.connected_components(G_trimmed_grp):
                # Get the subgraph for this connected component
                sg = G_trimmed_grp.subgraph(ccn)

                # If the connected component contains exactly 1 label we can skip
                labels = {
                    n
                    for n in sg.nodes
                    if G_trimmed_grp.nodes[n].get("type", None) != "neuron"
                }
                if len(labels) == 1:
                    continue

                # Check if we can split this connected component into smaller components
                partitions = split_check_recursive(sg)

                # If we can't split this connected component, we can skip
                if len(partitions) == 1:
                    continue

                # If we can split this connected component, we need to find edge that need to be removed
                sg_keep_edges = set()
                for p in partitions:
                    sg_keep_edges.update(
                        [(s, t) for s, t in sg.edges if s in p and t in p]
                    )

                # Add the difference to the spurious edges
                for edge in set(sg.edges) - sg_keep_edges:
                    # Translate any groups back into individual body IDs
                    if str(edge[0]).startswith("group"):
                        source = [int(s) for s in sg.nodes[edge[0]]["ids"].split(",")]
                    else:
                        source = [edge[0]]

                    if str(edge[1]).startswith("group"):
                        target = [int(s) for s in sg.nodes[edge[1]]["ids"].split(",")]
                    else:
                        target = [edge[1]]

                    for s in source:
                        for t in target:
                            self.spurious_edges_.append((s, t))

            # Remove edges
            if len(self.spurious_edges_):
                self.report(
                    f"\nRemoving {len(self.spurious_edges_)} potentially spurious edges from the graph.",
                    flush=True,
                )
                # Note to self: in a previous version I had issues that some edges were not removed
                # Turned out that was because I collected the edges from the *undirected* graph
                # but tried to remove them from the *directed* graph which silently failed.
                # We have since changed the code such that `G_trimmed` is always undirected.
                G_trimmed.remove_edges_from(self.spurious_edges_)

        # Generate mappings between labels
        mappings = {}
        labels = {
            n
            for n in G_trimmed.nodes
            if G_trimmed.nodes[n].get("type", None) != "neuron"
        }
        # Iterate over all connected components
        for ccn in nx.connected_components(G_trimmed):
            # Now we have to make sure this connected component is actually connected to all datasets
            skip = False
            for ds in datasets:
                if not any(G_trimmed.nodes[n].get(ds.label, False) for n in ccn):
                    skip = True
                    break
            if skip:
                continue

            # Generate a new label for this connected component
            ccn_labels = ccn & labels
            # Make sure we first split compound labels
            new_label = ",".join(
                set([l for label in ccn_labels for l in label.split(",")])
            )

            # Assign the new label to the neurons in this connected component
            for l in ccn - ccn_labels:
                mappings[l] = new_label

        # Store the results
        self.mappings_ = mappings
        self.graph_ = G_trimmed

        # Cache the results
        self._CACHE[self_identifier] = self

        self.report("Done.", flush=True)

        self.report(
            f"Found {len(set(self.mappings_.values()))} unique labels covering {len(self.mappings_)} neurons.",
            flush=True,
        )
        self._stale = False
        return self


def sanity_check_mapping(G, verbose=True):
    """Perform a sanity check on the mapping graph."""
    # Check that all neurons have a mapping
    indices = []
    records = []
    for ccn in nx.connected_components(G.to_undirected()):
        neurons = [n for n in ccn if G.nodes[n].get("type", None) == "neuron"]
        labels = [n for n in ccn if G.nodes[n].get("type", None) != "neuron"]

        # Split neurons into datasets
        datasets = {G.nodes[n].get("dataset", None) for n in neurons}
        counts = {
            ds: sum([G.nodes[n].get(f"{ds}", False) for n in neurons])
            for ds in datasets
        }
        records.append(counts)
        indices.append(",".join(labels))

    df = pd.DataFrame.from_records(records, index=indices)

    strange = df[(df.max(axis=1) / df.min(axis=1)) > 2]

    if not strange.empty:
        printv("Found some misbehaving labels:", verbose=verbose)
        printv(strange, verbose=verbose)

    return strange


def show_label_subgraph(x, labels, show_neurons=True, layout="shell"):
    """Quick visualization of a subgraph of the label graph.

    Parameters
    ----------
    G : nx.Graph | tuple
        The label graph. Can also be a tuple of datasets.
    labels : list
        List of labels to include in the subgraph.

    """
    if isinstance(x, GraphMapper):
        G = x.graph_
    elif isinstance(x, (nx.Graph, nx.DiGraph)):
        G = x
    else:
        raise ValueError("Unknown input.")

    if not isinstance(labels, (list, set, tuple)):
        labels = [labels]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("This function requires matplotlib.")

    keep = set()
    for ccn in nx.connected_components(G.to_undirected()):
        if any(l in ccn for l in labels):
            keep.update(ccn)

    sg = G.subgraph(keep).copy()  # copy to avoid frozen graph issues

    if not show_neurons:
        neurons = [n for n in sg.nodes if sg.nodes[n].get("type", None) == "neuron"]
        sg.remove_nodes_from(neurons)

    colors = {
        n: "coral" if G.nodes[n].get("type", None) == "neuron" else "lightblue"
        for n in sg.nodes
    }

    if layout == "shell":
        pos = nx.layout.shell_layout(
            sg,
            nlist=[
                [n for n in sg.nodes if sg.nodes[n].get("type", None) != "neuron"],
                [n for n in sg.nodes if sg.nodes[n].get("type", None) == "neuron"],
            ],
        )
    elif layout == "layers":
        nx.set_node_attributes(
            G,
            {n: 0 if sg.nodes[n].get("type", None) else 1 for n in sg.nodes},
            name="subset",
        )
        pos = nx.layout.multipartite_layout(sg)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    nx.draw(sg, with_labels=True, node_color=[colors[n] for n in sg.nodes], pos=pos)


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


def split_check_recursive(G, partitions=None, check_ratio=True):
    """Recursively split the graph in two until split is invalid.

    Invalid split means that one of the connected components does not
    contain neurons from all datasets.

    """
    if partitions is None:
        partitions = []

    # Find out which datasets are present
    ds = {
        G.nodes[n].get("dataset", None)
        for n in G.nodes
        if G.nodes[n].get("type", None) == "neuron"
    }

    # Get the ratio between datasets in the full set
    n_ds = {ds: sum([G.nodes[n].get(f"{ds}_in", 0) for n in G.nodes]) for ds in ds}

    try:
        ratio = max(n_ds.values()) / min(n_ds.values())
    except ZeroDivisionError:
        ratio = None

    # Split in two. Note: this may not actually split into two connected components!
    split = nx.community.greedy_modularity_communities(G, best_n=2, weight="weight")

    # Check if the split is valid
    valid = len(split) > 1
    for neuron_set in split:
        # If the set of datasets in this connected component is not the same as the
        # set of datasets in the entire graph, we must reject the split
        if not ds == {
            G.nodes[n].get("dataset", None)
            for n in neuron_set
            if G.nodes[n].get("type", None) == "neuron"
        }:
            valid = False
            # print(f"Invalid split {neuron_set} of {G.nodes}.")
            break

        # Check if the ratios between datasets are similar
        if ratio and check_ratio:
            n_ds = {
                ds: sum([G.nodes[n].get(f"{ds}_in", 0) for n in neuron_set])
                for ds in ds
            }
            ratio2 = max(n_ds.values()) / min(n_ds.values())
            # Reject split if the ratios get much worse
            if (ratio2 / ratio > 2.5) or (ratio / ratio2 > 2.5):
                valid = False
                # print(f"Invalid ratio for split {neuron_set} of {G.nodes}: {ratio2} vs {ratio}.")
                break

    if not valid:
        partitions.append(set(G.nodes))
    else:
        for neuron_set in split:
            split_check_recursive(G.subgraph(neuron_set).copy(), partitions)

    return partitions
