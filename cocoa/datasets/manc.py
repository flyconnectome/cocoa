import copy

import numpy as np
import pandas as pd
import neuprint as neu
import networkx as nx

from pathlib import Path

from .janelia import JaneliaDataSet
from .ds_utils import (
    _get_manc_meta,
    _get_neuprint_manc_client,
    _get_manc_types,
    _get_manc_sides,
    _add_types,
    _get_clio_client,
    _parse_neuprint_roi,
    _find_column
)
from ..utils import collapse_neuron_nodes

__all__ = ["MaleVNC"]

VNC_INTRINSIC_CLASSES = ("intrinsic_neuron", "ascending")

_DEFAULT_NEUROGLANCER_SOURCE = "precomputed://gs://manc-seg-v1p2/manc-seg-v1.2"


class MaleVNC(JaneliaDataSet):
    """Male Adult Nerve Cord (MANC) dataset.

    See https://neuprint.janelia.org/?dataset=manc%3Av1.2.3&qt=findneurons for more information.

    Parameters
    ----------
    label :             str
                        An identified used for reporting, plotting, etc.
    up/downstream :     bool
                        Whether to use up- and/or downstream connectivity.
    use_types :         bool
                        Whether to group by type.  Note that this may be overwritten
                        when used in the context of a `cocoa.Clustering`.
    backfill_types :    str | iterable, optional
                        A list of columns to use (in order) to backfill the `type`
                        column. Ignored if ``use_types=False``. If `True`, will use
                        all available type columns.
    exclude_autapses :  bool
                        Whether to exclude autapses from the connectivity vectors.
    use_side :          bool | 'relative'
                        Only relevant if `group_by_type=True`:
                         - if `True`, will split cell types into left/right/center
                         - if `relative`, will label cell types as `ipsi` or
                           `contra` depending on the side of the connected neuron
    rois :              str | list thereof, optional
                        Restrict connectivity to these regions of interest. Works
                        with super-level ROIs: e.g. "Brain" or "VNC" will be
                        automatically parsed into the appropriate sub-ROIs.
    meta_source :       "neuprint" (default) | "clio"
                        Source for meta data. You can also provide a specific
                        dataset by passing e.g. "neuprint/manc:v1.2.3". If not
                        specified, will use the latest version available.
    exclude_queries :   bool
                        If True (default), will exclude connections between query
                        neurons from the connectivity vector.
    cn_object :         str | pd.DataFrame
                        Either a DataFrame or path to a `.feather` connectivity file which
                        will be loaded into a DataFrame. The DataFrame is expected to
                        come from `neuprint.fetch_adjacencies` and include all relevant
                        IDs.

    """

    _flybrains_space = "JRCFIB2022Mraw"
    _type_columns = ("type",)
    _color = "blue"

    def __init__(
        self,
        label="MANC",
        upstream=True,
        downstream=True,
        use_types=False,
        backfill_types=False,
        exclude_autapses=True,
        use_sides=False,
        rois=None,
        meta_source="neuprint",
        exclude_queries=False,
        cn_object=None,
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries
        self.meta_source = meta_source
        self.cn_object = cn_object
        self.exclude_autapses = exclude_autapses

        if isinstance(backfill_types, str):
            backfill_types = [backfill_types]
        elif isinstance(backfill_types, np.ndarray):
            backfill_types = backfill_types.tolist()
        elif isinstance(backfill_types, bool):
            if not backfill_types:
                backfill_types = None
            else:
                backfill_types = ("systematic_type", "group", "instance")
        elif not isinstance(backfill_types, (list, tuple)):
            raise ValueError(
                "`backfill_types` must be a str, a list or tuple or `None`"
            )
        self.backfill_types = backfill_types

        if rois is not None:
            self.rois = _parse_neuprint_roi(rois, client=self.neuprint_client)
        else:
            self.rois = None

        self._neuroglancer_source = _DEFAULT_NEUROGLANCER_SOURCE

    @property
    def neuprint_client(self):
        """Return neuprint client."""
        return _get_neuprint_manc_client()

    @classmethod
    def hemisegments(cls, hemisegments, label=None, **kwargs):
        """Generate a dataset for left or right male VNC hemisegments.

        Parameters
        ----------
        hemisegments :  str
                        "left" or "right"
        label :         str, optional
                        Label for the dataset. If not provided will generate
                        one based on the hemisegments.
        **kwargs
            Additional keyword arguments for the dataset.

        Returns
        -------
        ds :            MaleVNC
                        A dataset for the specified hemisegments.

        """
        assert hemisegments in (
            "left",
            "right",
        ), f"Invalid hemisegments '{hemisegments}'"

        hemisegments = {"left": "LHS", "right": "RHS"}[hemisegments]

        if label is None:
            label = f"MaleVNC({hemisegments[0]})"
        ds = cls(label=label, **kwargs)

        ann = ds.get_annotations()
        to_add = ann[
            (ann.somaSide == hemisegments)
            | (ann.rootSide == hemisegments) & ann["class"].isin(VNC_INTRINSIC_CLASSES)
        ].bodyId.values

        ds.add_neurons(to_add)

        return ds

    def copy(self):
        """Make copy of dataset."""
        x = type(self)(label=self.label)
        x.neurons = self.neurons.copy()
        x.upstream = self.upstream
        x.downstream = self.downstream
        x.use_types = self.use_types
        x.backfill_types = self.backfill_types
        x.use_sides = self.use_sides
        x.exclude_queries = self.exclude_queries
        x.exclude_autapses = self.exclude_autapses
        x.meta_source = self.meta_source
        x.cn_object = self.cn_object
        x.rois = self.rois

        return x

    def clear_cache(self):
        """Clear cached in-memory data (e.g. annotations). Does not clear data cached on disk."""
        _get_manc_meta.cache_clear()
        _get_manc_types.cache_clear()
        _get_manc_meta.cache_clear()
        print("Cleared cached male VNC data.")

        return self

    def get_annotations(self):
        """Return annotations."""
        # Clio returns a "bodyid" column, neuprint a "bodyId" column
        ann = _get_manc_meta(source=self.meta_source).copy()

        # Drop empty strings (from e.g. `type`` column)
        for c in ann.columns:
            ann[c] = ann[c].replace("", np.nan)

        # Drop "TBD" instances
        ann.loc[ann.instance == 'TBD', 'instance'] = None

        return ann

    def get_all_neurons(self):
        """Get a list of all neurons in this dataset."""
        return self.get_annotations().bodyId.values

    def get_labels(self, x):
        """Fetch labels for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Body IDs to fetch labels for. If `None`, will return all labels.

        """
        # Fetch all types for this version
        types = _get_manc_types(
            add_side=False,
            source=self.meta_source,
            backfill_types=self.backfill_types,
        )

        if x is None:
            return types

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([types.get(i, i) for i in x])

    def get_sides(self, x):
        """Fetch labels for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Body IDs to fetch labels for. If `None`, will return all labels.

        """
        # Fetch all sides for this version
        sides = _get_manc_sides(source=self.meta_source)

        if x is None:
            return sides

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([sides.get(i, i) for i in x])

    def get_ngl_scene(self):
        client = _get_clio_client("MANC")
        seg_source = f'dvid://{client.meta["dvid"]}/{client.meta["uuid"]}/segmentation?dvid-service=https://ngsupport-bmcp5imp6q-uk.a.run.app'
        scene = copy.deepcopy(client.meta["neuroglancer"])
        scene.layers.append(
            {
                "source": {
                    "url": seg_source,
                    "subsources": {"default": True, "meshes": True},
                },
                "name": client.meta["tag"],
            }
        )
        return scene

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

    def compile_label_graph(
        self, which_neurons="all", collapse_neurons=False, strict=False
    ):
        """Compile label graph.

        For the MaleVNC, this means:
         1. Use the `type` plus columns defined in `backfill_types`
         2. Split compound `types` and `flywire_types` such that e.g. "PS008,PS009"
            produces two edges: (PS008,PS009 -> PS008) and (PS008,PS009 -> PS009)

        Parameters
        ----------
        which_neurons : "all" | "self"
                        Whether to use only the neurons in this dataset or all neurons
                        in the entire MaleVNC dataset (default).
        collapse_neurons : bool
                        If True, will collapse neurons with the same connectivity into
                        a single node. Useful for e.g. visualization.
        strict :        bool
                        If True, will prefix the labels with the type of the label (e.g. "manc:SNta19").

        Returns
        -------
        G : nx.DiGraph
            A graph with neurons and labels as nodes.

        Examples
        --------
        >>> import cocoa as cc
        >>> import networkx as nx
        >>> G = cc.MaleVNC().compile_label_graph()
        >>> nx.write_gml(G, "MANC_label_graph.gml", stringizer=str)

        """
        assert which_neurons in ("self", "all"), "Invalid `which_neurons`"

        ann = self.get_annotations()

        # Subset to the neurons in this dataset
        if which_neurons == "self":
            if not len(self):
                raise ValueError("No neurons in dataset")
            ann = ann[ann.bodyId.isin(self.neurons)].copy()

        # Initialise graph
        G = nx.DiGraph()

        # Add neuron nodes
        G.add_nodes_from(ann.bodyId, type="neuron")

        # Add labels
        cols = ["type"]
        if self.backfill_types:
            cols.extend(self.backfill_types)

        for col in cols:
            # Map column to the correct column in the annotation
            col = _find_column(col, ann)
            # Skip if this column doesn't exist
            if not col:
                continue
            # Get entries where this column is not null
            this = ann[ann[col].notnull()]
            if strict:
                this = this.copy()
                this[col] = "manc:" + this[col]
            # Add edges
            G.add_edges_from(zip(this.bodyId, this[col]))
            # Track which column(s) this label came from
            nx.set_edge_attributes(G, {e: {col: True} for e in zip(this.bodyId, this[col])})

            # Take care of compound types
            comp = this[
                this[col].str.contains(",", na=False)
                & ~this[col].str.endswith(
                    ", b", na=False
                )  # ignore e.g. "DVMn 3a, b"
            ][col].values

            for c, count in zip(*np.unique(comp, return_counts=True)):
                # We have to avoid splitting e.g. "DVMn 3a, b" into "DVMn 3a" and "b"
                # If any of the split labels is just a single letter, we'll skip it
                if any(
                    len(s.strip()) == 1 for s in c.split(",")
                ):
                    continue

                for c2 in c.split(","):
                    G.add_edge(c.strip(), c2.strip(), weight=count)
                    nx.set_edge_attributes(G, {(c.strip(), c2.strip()): {col: True}})

        # For known antonyms (i.e. labels that are the same in another dataset but do not indicate matches)
        # we will use the node properties to indicate which datasets it must not be matched against.
        # For example:
        # G.nodes['node']['antonyms_in'] = ("MANC")

        if collapse_neurons:
            G = collapse_neuron_nodes(G)

        return G

    def compile(self, collapse_types=False, collapse_rois=True):
        """Compile connectivity vector."""
        client = self.neuprint_client

        x = self.neurons.astype(np.int64)

        if not len(x):
            raise ValueError("No body IDs provided")

        if self.use_types:
            # Types is a {bodyId: type} dictionary
            if hasattr(self, "types_"):
                types = self.types_
            else:
                types = _get_manc_types(
                    add_side=False,
                    backfill_types=self.backfill_types,
                    source=self.meta_source,
                )

        # Fetch hemibrain vectors
        if self.upstream:
            # print("Fetching upstream connectivity... ", end="", flush=True)
            if isinstance(self.cn_object, pd.DataFrame):
                us = self.cn_object[self.cn_object.bodyId_post.isin(x)]
                if self.rois is not None:
                    us = us[us.roi.isin(self.rois)]
                us = us.copy()  # avoid SettingWithCopyWarning
            else:
                _, us = neu.fetch_adjacencies(
                    targets=neu.NeuronCriteria(bodyId=x, client=client),
                    rois=self.rois,
                    client=client,
                )
            if self.exclude_queries:
                us = us[~us.bodyId_pre.isin(x)]
            if self.exclude_autapses:
                us = us[us.bodyId_pre != us.bodyId_post].copy()
            us.rename(
                {"bodyId_pre": "pre", "bodyId_post": "post"}, axis=1, inplace=True
            )
            # Collapse ROIs here before we (potentially) add types
            if collapse_rois:
                us = us.groupby(["pre", "post"], as_index=False).weight.sum()

            if self.use_types:
                us = _add_types(
                    us,
                    types=types,
                    col="pre",
                    sides=None
                    if not self.use_sides
                    else _get_manc_sides(source=self.meta_source),
                    sides_rel=True if self.use_sides == "relative" else False,
                )
            # print("Done!")

        if self.downstream:
            # print("Fetching downstream connectivity... ", end="", flush=True)
            if isinstance(self.cn_object, pd.DataFrame):
                ds = self.cn_object[self.cn_object.bodyId_pre.isin(x)]
                if self.rois is not None:
                    ds = ds[ds.roi.isin(self.rois)]
                ds = ds.copy()  # avoid SettingWithCopyWarning
            else:
                _, ds = neu.fetch_adjacencies(
                    sources=neu.NeuronCriteria(bodyId=x, client=client),
                    rois=self.rois,
                    client=client,
                )
            if self.exclude_queries:
                ds = ds[~ds.bodyId_post.isin(x)]
            if self.exclude_autapses:
                ds = ds[ds.bodyId_pre != ds.bodyId_post].copy()
            ds.rename(
                {"bodyId_pre": "pre", "bodyId_post": "post"}, axis=1, inplace=True
            )
            # Collapse ROIs here before we (potentially) add types
            if collapse_rois:
                ds = ds.groupby(["pre", "post"], as_index=False).weight.sum()

            if self.use_types:
                ds = _add_types(
                    ds,
                    types=types,
                    col="post",
                    sides=None if not self.use_sides else _get_manc_meta(),
                    sides_rel=True if self.use_sides == "relative" else False,
                )

        if self.upstream and self.downstream:
            self.edges_ = pd.concat((us, ds), axis=0).drop_duplicates()
        elif self.upstream:
            self.edges_ = us
        elif self.downstream:
            self.edges_ = ds
        else:
            raise ValueError("`upstream` and `downstream` must not both be False")

        if collapse_types:
            # Make sure to keep "roi" if it still exits
            cols = [c for c in ["pre", "post", "roi"] if c in self.edges_.columns]
            self.edges_ = self.edges_.groupby(cols, as_index=False).weight.sum()

        # Keep track of whether this used types and side
        self.edges_types_used_ = self.use_types
        self.edges_sides_used_ = self.use_sides

        return self


def _collapse_connectivity_types(type_dict, source="clio"):
    """Remove connectivity type suffixes from {ID: type} dictionary."""
    type_dict = type_dict.copy()
    hb_meta = _get_manc_meta(source=source)
    cn2morph = hb_meta.set_index("type").morphology_type.to_dict()
    for k, v in type_dict.items():
        new_v = ",".join([cn2morph.get(t, t) for t in v.split(",")])
        type_dict[k] = new_v
    return type_dict
