import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import caveclient as cc
import seaserpent as ss

from abc import ABC
from fafbseg import flywire

from .core import DataSet
from .ds_utils import _is_int, _find_column, _add_types
from ..utils import collapse_neuron_nodes


class CaveDataset(DataSet, ABC):
    """Base class for datasets using the CaveClient API."""

    required_attributes = ["_datastack_name", "_type_cols", "_side_cols"]

    def __init__(self, label):
        super().__init__(label)
        self._materialization = None

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure required attributes are defined
        for attr in cls.required_attributes:
            if not hasattr(cls, attr):
                raise NotImplementedError(f"{cls.__name__} must define {attr}")

    @property
    def caveclient(self):
        if not hasattr(self, "_caveclient"):
            self._caveclient = cc.CAVEclient(self._datastack_name)
        return self._caveclient

    @property
    def materialization(self):
        return self._materialization

    @materialization.setter
    def materialization(self, value):
        if value == "latest":
            value = max(self.caveclient.materialization.get_versions())
        self._materialization = value

    def add_neurons(self, x, regex="auto", sides=None):
        """Add neurons to dataset.

        Parameters
        ----------
        x :         int | str | list | np.ndarray | pd.Series | None
                    Root IDs or cell types to add. Can also use "{column}:{value}" to filter
                    for values in given column.
        regex :     "auto" | bool
                    Whether strings are interpreted as regular expressions.
                    If "auto" (default), will treat strings starting with "/" as regex.
        sides :     str | iterable | None
                    If provided, will only add neurons on the given side(s).

        """
        new_neurons = self._parse_ids(x, regex=regex, sides=sides)

        if len(new_neurons):
            self.neurons = np.unique(np.append(self.neurons, new_neurons))
        else:
            print(f'Warning: No neurons found for query "{x}"')
        return self

    def _parse_ids(self, x, regex="auto", sides=None):
        """Parse `x` into root IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if regex == "auto" and isinstance(x, str):
            regex = x.startswith("/")
            x = x[1:] if regex else x

        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(ids, self._parse_ids(t, regex=regex, sides=sides))
        elif _is_int(x):
            ids = [int(x)]
        else:
            annot = self.get_annotations()

            if ":" not in x:
                filt = pd.Series(False, index=annot.index)
                if not regex:
                    for col in self._type_cols:
                        filt = filt | (annot[col] == x)
                else:
                    # filt = annot.cell_type.str.contains(
                    #     x, na=False, case=False
                    # ) | annot.hemibrain_type.str.contains(x, na=False, case=False)
                    for col in self._type_cols:
                        filt = filt | annot[col].str.contains(x, na=False, case=False)
            else:
                # If this is e.g. "cell_class:L1-5"
                col, val = x.split(":")
                if col not in annot.columns:
                    raise ValueError(f"Column '{col}' not found in annotations.")
                if not regex:
                    filt = annot[col] == val
                else:
                    filt = annot[col].str.contains(val, na=False)

            if isinstance(sides, str):
                filt = filt & (annot.side == sides)
            elif isinstance(sides, (tuple, list, np.ndarray)):
                filt = filt & annot.side.isin(sides)
            ids = annot.loc[filt, "root_id"].unique().astype(np.int64).tolist()

        return ids

    def get_all_neurons(self):
        """Get a list of all neurons in this dataset."""
        return self.get_annotations().root_id.values

    def get_labels(self, x):
        """Fetch labels (compiled from types) for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Root IDs to fetch labels for. If `None`, will return all labels.

        """
        if not hasattr(self, "_types_dict"):
            annot = self.get_annotations().set_index("root_id")[self._type_cols].copy()

            # Make sure we don't have empty strings
            for col in self._type_cols:
                annot.loc[annot[col] == "", col] = np.nan

            for col in self._type_cols[1:]:
                annot[self._type_cols[0]] = annot[self._type_cols[0]].fillna(annot[col])
            self._types_dict = annot[self._type_cols[0]].to_dict()

        if x is None:
            return self._types_dict

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([self._types_dict.get(i, i) for i in x])

    def get_sides(self, x):
        """Fetch sides for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Root IDs to fetch sides for. If `None`, will return all labels.

        """
        if not hasattr(self, "_sides_dict"):
            annot = self.get_annotations().set_index("root_id")[self._side_cols].copy()

            # Make sure we don't have empty strings
            for col in self._side_cols:
                annot.loc[annot[col] == "", col] = np.nan

            for col in self._side_cols[1:]:
                annot[self._side_cols[0]] = annot[self._side_cols[0]].fillna(annot[col])
            self._sides_dict = annot[self._side_cols[0]].to_dict()

        if x is None:
            return self._sides_dict

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([self._sides_dict.get(i, i) for i in x])

    def clear_cache(self):
        """Clear cached data (e.g. annotations). Does not clear data cached on disk."""
        for attr in ("_annotations", "_types_dict", "_sides_dict"):
            if hasattr(self, attr):
                delattr(self, attr)

    def label_exists(self, x):
        """Check if labels exists in dataset."""
        x = np.asarray(x)

        # This graph contains all possible labels in this dataset,
        # including synonyms and split compound types
        G = self.compile_label_graph(which_neurons="all")

        # Remove the neurons themselves
        G.remove_nodes_from(
            [k for k, v in nx.get_node_attributes(G, "type").items() if v == "neuron"]
        )

        return np.isin(x, list(G.nodes))

    # Should this be cached and/or turned into a classmethod?
    def compile_label_graph(self, which_neurons="all", collapse_neurons=False):
        """Compile label graph.

        Parameters
        ----------
        which_neurons : "all" | "self"
                        Whether to use only the neurons in this dataset or all neurons in the entire dataset.
        collapse_neurons : bool
                        If True, will collapse neurons with the same connectivity into
                        a single node. Useful for e.g. visualization.

        Returns
        -------
        G : nx.DiGraph
            A graph with neurons and labels as nodes.

        """
        assert which_neurons in (
            "self",
            "all",
        ), "`which_neurons` must be 'self' or 'all'"

        ann = self.get_annotations()

        # Subset to the neurons in this dataset
        if which_neurons == "self":
            if not len(self):
                raise ValueError("No neurons in dataset")
            ann = ann[ann.root_id.isin(self.neurons)]

        # Initialise graph
        G = nx.DiGraph()

        # Add neuron nodes
        G.add_nodes_from(ann.root_id, type="neuron")

        # Order of labels
        for col in self._type_cols:
            # Map column to the correct column in the annotation
            col = _find_column(col, ann)
            # Skip if this column doesn't exist
            if not col:
                continue
            # Get entries where this column is not null
            this = ann[ann[col].notnull()]
            # Add edges
            G.add_edges_from(zip(this.root_id, this[col]))
            # Track which column(s) this label came from
            nx.set_edge_attributes(
                G, {e: {col: True} for e in zip(this.root_id, this[col])}
            )

            # Take care of compound types
            comp = this[
                this[col].str.contains(",", na=False)
                & ~this[col].str.startswith(
                    "(", na=False
                )  # ignore e.g. "(M_adPNm4,M_adPNm5)b"
            ][col].values

            for c, count in zip(*np.unique(comp, return_counts=True)):
                # We have to avoid splitting e.g. "P1_17a,b" in "P1_17a" and "b"
                # If any of the split labels is just a single letter, we'll skip it
                if any(len(s.strip()) == 1 for s in c.split(",")):
                    continue

                for c2 in c.split(","):
                    G.add_edge(c.strip(), c2.strip(), weight=count)
                    nx.set_edge_attributes(G, {(c.strip(), c2.strip()): {col: True}})

        if collapse_neurons:
            G = collapse_neuron_nodes(G)

        return G

    def compile_adjacency(self, collapse_types=False):
        """Compile adjacency between all neurons in this dataset.

        Parameters
        ----------
        collapse_types : bool
                        Whether to collapse by type.
        collapse_rois : bool
                        Whether to collapse across ROIs.

        Returns
        -------
        self :          DataSet
                        The dataset with compiled adjacency as `adj_` attribute.

        """
        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(np.int64)

        # Fetch edges between given IDs
        edges = _fetch_edges(
            source=x,
            target=x,
            client=self.caveclient,
            materialization=self.materialization,
        )

        # For grouping by type simply replace pre and post IDs with their types
        # -> we'll aggregate later
        if self.use_types:
            types = self.get_labels(None)
            sides = self.get_sides(None)

            edges = _add_types(
                edges,
                types=types,
                col=("pre", "post"),
                expand_morphology_types=True,
                sides=None if not self.use_sides else sides,
                sides_rel=True if self.use_sides == "relative" else False,
            )

        self.adj_ = edges

        if collapse_types:
            self.adj_ = self.adj_.groupby(["pre", "post"], as_index=False).weight.sum()

        # Keep track of whether this used types and side
        self.adj_types_used_ = self.use_types
        self.adj_sides_used_ = self.use_sides

        # Translate morphology types into connectivity types
        # This makes it easier to align with hemibrain
        # self.connectivity_.columns = _morphology_to_connectivity_types(
        #    self.connectivity_.columns
        # )

        return self

    def compile(self, collapse_types=False, drop_unannotated=True):
        """Compile edges for the neurons in this dataset.

        Parameters
        ----------
        collapse_types :    bool
                            Whether to collapse by type.
        drop_unannotated :  bool
                            If True, will drop edges to/from IDs that aren't
                            among the annotations.

        Returns
        -------
        self :          DataSet
                        The dataset with compiled edges as `edges_` attribute.

        """
        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(np.int64)

        if self.upstream:
            us = _fetch_edges(
                source=None,
                target=x,
                client=self.caveclient,
                materialization=self.materialization,
            )
            if drop_unannotated:
                us = us[us.pre.isin(self.get_annotations().root_id)]
        if self.downstream:
            ds = _fetch_edges(
                source=x,
                target=None,
                client=self.caveclient,
                materialization=self.materialization,
            )
            if drop_unannotated:
                ds = ds[ds.post.isin(self.get_annotations().root_id)]

        if self.exclude_queries:
            if self.upstream:
                us = us[~us.pre.isin(x)]
            if self.downstream:
                ds = ds[~ds.post.isin(x)]

        # For grouping by type simply replace pre and post IDs with their types
        # -> we'll aggregate later
        # For grouping by type simply replace pre and post IDs with their types
        # -> we'll aggregate later
        if self.use_types:
            types = self.get_labels(None)
            sides = self.get_sides(None)

            if self.upstream:
                us = _add_types(
                    us,
                    types=types,
                    col=("pre", "post"),
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else sides,
                    sides_rel=True if self.use_sides == "relative" else False,
                )
            if self.downstream:
                ds = _add_types(
                    ds,
                    types=types,
                    col=("pre", "post"),
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else sides,
                    sides_rel=True if self.use_sides == "relative" else False,
                )

        if self.upstream and self.downstream:
            self.edges_ = pd.concat(
                (
                    us.groupby(["pre", "post"], as_index=False).weight.sum(),
                    ds.groupby(["pre", "post"], as_index=False).weight.sum(),
                ),
                axis=0,
            ).drop_duplicates()
        elif self.upstream:
            self.edges_ = us.groupby(["pre", "post"], as_index=False).weight.sum()
        elif self.downstream:
            self.edges_ = ds.groupby(["pre", "post"], as_index=False).weight.sum()
        else:
            raise ValueError("`upstream` and `downstream` must not both be False")

        if collapse_types:
            self.edges_ = self.edges_.groupby(
                ["pre", "post"], as_index=False
            ).weight.sum()

        # Keep track of whether this used types and side
        self.edges_types_used_ = self.use_types
        self.edges_sides_used_ = self.use_sides

        return self


def _fetch_synapses(source, target, client, materialization):
    """Helper function to fetch connectivity via the CAVE client."""

    # Make sure we're working on integers
    all_ids = np.array([], dtype=np.int64)
    if source is not None:
        source = np.asarray(source).astype(np.int64)
        all_ids = np.concat((all_ids, source))
    if target is not None:
        target = np.asarray(target).astype(np.int64)
        all_ids = np.concat((all_ids, target))
    all_ids = np.unique(all_ids)

    if materialization == "auto":
        materialization = flywire.utils.find_mat_version(
            all_ids, dataset=client.datastack_name
        )
    elif materialization == "latest":
        materialization = max(client.materialization.get_versions())
    else:
        timestamp = None if materialization == "live" else f"mat_{materialization}"

        il = flywire.is_latest_root(
            all_ids, timestamp=timestamp, dataset=client.datastack_name
        )
        if any(~il):
            raise ValueError(
                "Some of the root IDs does not exist for the specified "
                f"materialization ({materialization}): {all_ids[~il]}"
            )

    # Grab connectivity from CAVE
    if materialization != "live":
        syn = client.materialize.synapse_query(
            pre_ids=source, post_ids=target, materialization=materialization
        )
    else:
        filter_in_dict = {}
        if source is not None:
            filter_in_dict["pre_pt_root_id"] = source
        if target is not None:
            filter_in_dict["post_pt_root_id"] = target
        syn = client.materialize.live_query(
            filter_in_dict=filter_in_dict,
            timestamp=dt.datetime.utcnow(),
            table=client.materialize.synapse_table,
        )

    return syn


def _fetch_edges(source, target, client, materialization):
    """Helper class to fetch edges."""
    syn = _fetch_synapses(source, target, client, materialization)

    return (
        syn.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False)
        .size()
        .rename(
            columns={
                "size": "weight",
                "pre_pt_root_id": "pre",
                "post_pt_root_id": "post",
            }
        )
    )
