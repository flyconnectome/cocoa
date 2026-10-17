import re
import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt

from abc import ABC
from pathlib import Path
from fafbseg import flywire

from .core import DataSet
from .ds_utils import (
    _is_int,
    _find_column,
    _add_types,
    _get_cave_client,
    _morphology_to_connectivity_types,
)
from ..utils import collapse_neuron_nodes

# Accepted column names for local connectivity (`cn_object`)
CN_COLUMN_ALIASES = {
    "pre": ("pre_pt_root_id", "pre_root_id", "pre"),
    "post": ("post_pt_root_id", "post_root_id", "post"),
    "weight": ("syn_count", "weight", "count"),
}


class CaveDataset(DataSet, ABC):
    """Base class for datasets using the CaveClient API."""

    required_attributes = ["_datastack_name", "_type_cols", "_side_cols"]

    # N.B. the `cn_object` setter renames e.g. "neuropil" to "roi"
    _roi_col = "roi"

    # Attributes copied over by `.copy()` (in addition to neurons & connectivity)
    _copy_attrs = (
        "upstream",
        "downstream",
        "use_types",
        "use_sides",
        "exclude_queries",
    )

    def __init__(
        self,
        label,
        upstream=True,
        downstream=True,
        use_types=False,
        use_sides=False,
        exclude_queries=False,
        cn_object=None,
        materialization="latest",
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label)
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries
        self.cn_object = cn_object
        self.materialization = materialization

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure required attributes are defined
        for attr in cls.required_attributes:
            if not hasattr(cls, attr):
                raise NotImplementedError(f"{cls.__name__} must define {attr}")

    @property
    def caveclient(self):
        return _get_cave_client(self._datastack_name)

    @property
    def materialization(self):
        return self._materialization

    @materialization.setter
    def materialization(self, value):
        if value not in ("auto", "latest", "live") and not _is_int(value):
            raise ValueError(
                '`materialization` must be "auto", "latest", "live" or an integer, '
                f'got "{value}"'
            )
        self._check_materialization(value)
        if value == "latest":
            value = max(self.caveclient.materialize.get_versions())
        elif value not in ("auto", "live"):
            value = int(value)
        self._materialization = value

    def _check_materialization(self, value):
        """Raise if `value` isn't supported by this dataset. Override as needed."""
        pass

    def _resolve_materialization(self, ids):
        """Resolve "auto" materialization and check that `ids` exist at it.

        This is the only place where "auto" gets resolved: we pick a
        materialization at which all `ids` co-exist and store it on the
        dataset so that subsequent queries use the same version.

        Returns
        -------
        int | "live"

        """
        ids = np.unique(np.asarray(ids).astype(np.int64))
        client = self.caveclient

        if self.materialization == "auto":
            self.materialization = flywire.utils.find_mat_version(
                ids, dataset=client.datastack_name
            )
            return self.materialization

        if self.materialization == "live":
            timestamp = None
        else:
            timestamp = client.materialize.get_version_metadata(
                self.materialization
            )["time_stamp"]
        il = client.chunkedgraph.is_latest_roots(ids, timestamp=timestamp)
        if any(~il):
            raise ValueError(
                "Some of the root IDs do not exist for the specified "
                f"materialization ({self.materialization}): {ids[~il]}"
            )
        return self.materialization

    @property
    def cn_object(self):
        """Local connectivity used instead of fetching edges from CAVE."""
        # Lazy-load the connectivity object if it is a file path
        if isinstance(self._cn_object, Path):
            self.cn_object = pd.read_feather(self._cn_object)
        return self._cn_object

    @cn_object.setter
    def cn_object(self, value):
        """Set the connectivity object."""
        if value is None:
            self._cn_object = None
        elif isinstance(value, pd.DataFrame):
            # Treat "pre/post_root_{materialization_version}" like "pre/post_root_id"
            columns = [
                re.sub(r"^(pre|post)_root_\d+$", r"\1_root_id", c) for c in value.columns
            ]

            for aliases in CN_COLUMN_ALIASES.values():
                if sum(columns.count(c) for c in aliases) != 1:
                    raise ValueError(
                        f"DataFrame must contain exactly one of the columns: {', '.join(aliases)}"
                    )

            renames = {
                a: col for col, aliases in CN_COLUMN_ALIASES.items() for a in aliases
            }
            renames["neuropil"] = "roi"
            self._cn_object = value.set_axis(
                [renames.get(c, c) for c in columns], axis=1
            )
        elif isinstance(value, (str, Path)):
            self._cn_object = Path(value).expanduser()
            if not self._cn_object.is_file():
                raise ValueError(f'"{self._cn_object}" is not a valid file')
        else:
            raise ValueError("`cn_object` must be a DataFrame, a file path or None.")

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
            annot = self.get_annotations().set_index("root_id")[list(self._type_cols)].copy()

            # Make sure we don't have empty strings
            for col in self._type_cols:
                annot.loc[annot[col] == "", col] = np.nan

            for col in self._type_cols[1:]:
                annot[self._type_cols[0]] = annot[self._type_cols[0]].fillna(annot[col])

            # Drop untyped neurons (i.e. those with NaN in the type column(s))
            annot = annot[annot[self._type_cols[0]].notnull()]

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
            annot = self.get_annotations().set_index("root_id")[list(self._side_cols)].copy()

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
        for attr in ("_annotations", "_types_dict", "_cn_types_dict", "_sides_dict"):
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
    def compile_label_graph(self, which_neurons="all", collapse_neurons=False, strict=False):
        """Compile label graph.

        Parameters
        ----------
        which_neurons : "all" | "self"
                        Whether to use only the neurons in this dataset or all neurons in the entire dataset.
        collapse_neurons : bool
                        If True, will collapse neurons with the same connectivity into
                        a single node. Useful for e.g. visualization.
        strict :        bool
                        Currently not supported for CAVE datasets and must be False.

        Returns
        -------
        G : nx.DiGraph
            A graph with neurons and labels as nodes.

        """
        assert which_neurons in (
            "self",
            "all",
        ), "`which_neurons` must be 'self' or 'all'"

        if strict:
            raise ValueError(
                f"`strict=True` is currently not supported for {type(self).__name__}."
            )

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

    def _get_edges(self, source, target, by_roi=False):
        """Get edges between `source` and `target` IDs (`None` = any neuron).

        Uses the local connectivity (`cn_object`) if provided and otherwise
        fetches edges from CAVE. Edges are always aggregated to one row per
        pre-post pair (or per pre-post-roi triplet if `by_roi=True`).
        """
        cn = self.cn_object
        if by_roi and (cn is None or "roi" not in cn.columns):
            raise ValueError(
                "Per-neuropil connectivity requires a `cn_object` with a neuropil column."
            )

        if cn is None:
            ids = [i for i in (source, target) if i is not None]
            return _fetch_edges(
                source=source,
                target=target,
                client=self.caveclient,
                materialization=self._resolve_materialization(np.concatenate(ids)),
            )

        filt = np.ones(len(cn), dtype=bool)
        if source is not None:
            filt &= cn.pre.isin(source).values
        if target is not None:
            filt &= cn.post.isin(target).values
        cols = ["pre", "post", "roi"] if by_roi else ["pre", "post"]
        return cn[filt].groupby(cols, as_index=False).weight.sum()

    def _apply_types(self, edges):
        """Replace pre/post IDs in `edges` with their types (and sides)."""
        # Expanding morphology into connectivity types is expensive -> cache it
        if not hasattr(self, "_cn_types_dict"):
            types = self.get_labels(None)
            self._cn_types_dict = dict(
                zip(types, _morphology_to_connectivity_types(list(types.values())))
            )
        return _add_types(
            edges,
            types=self._cn_types_dict,
            col=("pre", "post"),
            sides=self.get_sides(None) if self.use_sides else None,
            sides_rel=self.use_sides == "relative",
        )

    def copy(self):
        """Make copy of dataset."""
        # N.B. `_copy_attrs` must all be constructor arguments. We also pass the
        # resolved materialization so we don't look up "latest" again.
        x = type(self)(
            label=self.label,
            materialization=self.materialization,
            **{attr: getattr(self, attr) for attr in self._copy_attrs},
        )
        x.neurons = self.neurons.copy()
        # Bypass the setter: no need to (lazy-)load or re-validate the connectivity
        x._cn_object = self._cn_object
        return x

    def compile_adjacency(self, collapse_types=False, collapse_rois=True):
        """Compile adjacency between all neurons in this dataset.

        Parameters
        ----------
        collapse_types : bool
                        Whether to collapse by type.
        collapse_rois : bool
                        Whether to collapse across neuropils. If False, requires
                        a `cn_object` with a neuropil column.

        Returns
        -------
        self :          DataSet
                        The dataset with compiled adjacency as `adj_` attribute.

        """
        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(np.int64)

        # Fetch edges between given IDs
        edges = self._get_edges(source=x, target=x, by_roi=not collapse_rois)

        # For grouping by type simply replace pre and post IDs with their types
        # -> we'll aggregate later
        if self.use_types:
            edges = self._apply_types(edges)

        self.adj_ = edges

        if collapse_types:
            self.adj_ = self.adj_.groupby(["pre", "post"], as_index=False).weight.sum()

        # Keep track of whether this used types and side
        self.adj_types_used_ = self.use_types
        self.adj_sides_used_ = self.use_sides

        return self

    def compile(self, collapse_types=False, drop_unannotated=True):
        """Compile edges for the neurons in this dataset.

        Parameters
        ----------
        collapse_types :    bool
                            Whether to collapse by type.
        drop_unannotated :  bool
                            If True (default), will drop edges to/from partners
                            that aren't among the annotations.

        Returns
        -------
        self :          DataSet
                        The dataset with compiled edges as `edges_` attribute.

        """
        if not self.upstream and not self.downstream:
            raise ValueError("`upstream` and `downstream` must not both be False")

        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(np.int64)

        # Pairs of (edges, column with the partner neurons)
        parts = []
        if self.upstream:
            parts.append((self._get_edges(source=None, target=x), "pre"))
        if self.downstream:
            parts.append((self._get_edges(source=x, target=None), "post"))

        if drop_unannotated:
            annotated = pd.Index(self.get_annotations().root_id.values)

        edges = []
        for e, partner in parts:
            if drop_unannotated:
                e = e[e[partner].isin(annotated)]
            if self.exclude_queries:
                e = e[~e[partner].isin(x)]
            # For grouping by type simply replace pre and post IDs with their
            # types and aggregate (N.B. `_get_edges` already aggregates by ID)
            if self.use_types:
                e = self._apply_types(e)
                e = e.groupby(["pre", "post"], as_index=False).weight.sum()
            edges.append(e)

        # N.B. edges between query neurons show up both up- and downstream
        self.edges_ = pd.concat(edges, axis=0).drop_duplicates()

        if collapse_types:
            self.edges_ = self.edges_.groupby(
                ["pre", "post"], as_index=False
            ).weight.sum()

        # Keep track of whether this used types and side
        self.edges_types_used_ = self.use_types
        self.edges_sides_used_ = self.use_sides

        return self


def _fetch_synapses(source, target, client, materialization):
    """Helper function to fetch connectivity via the CAVE client.

    `materialization` must already be resolved (i.e. an integer or "live"),
    see `CaveDataset._resolve_materialization`.
    """
    assert materialization == "live" or _is_int(materialization)

    # Make sure we're working on integers
    if source is not None:
        source = np.asarray(source).astype(np.int64)
    if target is not None:
        target = np.asarray(target).astype(np.int64)

    # Grab connectivity from CAVE
    if materialization != "live":
        syn = client.materialize.synapse_query(
            pre_ids=source, post_ids=target, materialization_version=materialization
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
