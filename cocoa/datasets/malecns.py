import copy

import numpy as np
import pandas as pd
import neuprint as neu
import networkx as nx

from pathlib import Path

from .core import DataSet
from .scenes import _get_mcns_scene
from .ds_utils import (
    _get_mcns_meta,
    _get_neuprint_mcns_client,
    _is_int,
    _get_mcns_types,
    _get_mcns_sides,
    _add_types,
    _get_clio_client,
    _parse_neuprint_roi,
    MCNS_BAD_TYPES,
)
from ..utils import collapse_neuron_nodes

__all__ = ["MaleCNS"]

itable = None
otable = None


class MaleCNS(DataSet):
    """Male CNS dataset.

    Parameters
    ----------
    label :             str
                        A label used for reporting, plotting, etc.
    up/downstream :     bool
                        Whether to use up- and/or downstream connectivity.
    use_types :         bool
                        Whether to group by type.  Note that this may be overwritten
                        when used in the context of a `cocoa.Clustering`.
    backfill_types :    bool
                        If True, will backfill the type with information
                        extracted from other columns such as flywire_type,
                        hemibrain_type, instance and the group fields. Ignored
                        if ``use_types=False``.
    use_side :          bool | 'relative'
                        Only relevant if `group_by_type=True`:
                         - if `True`, will split cell types into left/right/center
                         - if `relative`, will label cell types as `ipsi` or
                           `contra` depending on the side of the connected neuron
    rois :              str | list thereof, optional
                        Restrict connectivity to these regions of interest. Works
                        with super-level ROIs: e.g. "Brain" or "VNC" will be
                        automatically parsed into the appropriate sub-ROIs.
    meta_source :       "clio" | "neuprint
                        Source for meta data.
    exclude_queries :   bool
                        If True (default), will exclude connections between query
                        neurons from the connectivity vector.
    cn_object :         str | pd.DataFrame 
                        Either a DataFrame or path to a `.feather` connectivity file which 
                        will be loaded into a DataFrame. The DataFrame is expected to 
                        come from `neuprint.fetch_adjacencies` and include all relevant 
                        IDs.

    """

    def __init__(
        self,
        label="maleCNS",
        upstream=True,
        downstream=True,
        use_types=False,
        backfill_types=False,
        use_sides=False,
        rois=None,
        meta_source="clio",
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
        self.backfill_types = backfill_types
        self.meta_source = meta_source
        self.cn_object = cn_object

        if rois is not None:
            self.rois = _parse_neuprint_roi(rois, client=_get_neuprint_mcns_client())
        else:
            self.rois = None

        if self.cn_object is not None:
            if isinstance(self.cn_object, (str, Path)):
                self.cn_object = Path(self.cn_object).expanduser()
                if not self.cn_object.is_file():
                    raise ValueError(f'"{self.cn_object}" is not a valid file')
                else:
                    self.cn_object = pd.read_feather(self.cn_object)
            elif not isinstance(self.cn_object, pd.DataFrame):
                raise ValueError("`cn_object` must be a path or a DataFrame")

    def _add_neurons(self, x, exact=False, right_only=False):
        """Turn `x` into male CNS body IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if isinstance(x, str) and "," in x:
            x = x.split(",")

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(
                    ids, self._add_neurons(t, exact=exact, right_only=right_only)
                )
        elif _is_int(x):
            ids = [np.int64(x)]
        else:
            meta = self.get_annotations()
            if right_only:
                ids = meta.loc[
                    (meta.type == x) & meta.side.isin(("R", "right")),
                    "bodyId",
                ].values.astype(np.int64)
            else:
                ids = meta.loc[(meta.type == x), "bodyId"].values.astype(np.int64)

        return np.unique(np.array(ids, dtype=np.int64))

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

        return x

    def clear_cache(self):
        """Clear cached data (e.g. annotations)."""
        _get_mcns_meta.cache_clear()
        _get_mcns_types.cache_clear()
        _get_mcns_meta.cache_clear()
        print("Cleared cached male CNS data.")

    def get_annotations(self):
        """Return annotations."""
        # Clio returns a "bodyid" column, neuprint a "bodyId" column
        ann = _get_mcns_meta(source=self.meta_source).rename(
            {"bodyid": "bodyId"}, axis=1
        )

        # Drop empty strings (from e.g. `type`` column)
        for c in ann.columns:
            ann[c] = ann[c].replace("", np.nan)

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
        types = _get_mcns_types(add_side=False, source=self.meta_source)

        if x is None:
            return types

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([types.get(i, i) for i in x])

    def get_ngl_scene(self, in_flywire_space=False):
        client = _get_clio_client("CNS")
        seg_source = f'dvid://{client.meta["dvid"]}/{client.meta["uuid"]}/segmentation?dvid-service=https://ngsupport-bmcp5imp6q-uk.a.run.app'
        if not in_flywire_space:
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

        else:
            scene = copy.deepcopy(_get_mcns_scene())
            scene["layers"][0]["source"] = seg_source
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
        self, which_neurons="all", exclude_bad_types=True, collapse_neurons=False
    ):
        """Compile label graph.

        For the MaleCNS, this means:
         1. Use the `type` as primary labels backfill with (in order):
            - `flywire_type`
            - `group`
            - `instance`
         2. Add synonyms for `type` and `flywire_type`
         3. Split compound `types` and `flywire_types` such that e.g. "PS008,PS009"
            produces two edges: (PS008,PS009 -> PS008) and (PS008,PS009 -> PS009)

        Parameters
        ----------
        which_neurons : "all" | "self"
                        Whether to use only the neurons in this dataset or all neurons
                        in the entire MaleCNS dataset (default).
        exclude_bad_types : bool
                        Whether to exclude known bad types such as "KC" or "FB".
        collapse_neurons : bool
                        If True, will collapse neurons with the same connectivity into
                        a single node. Useful for e.g. visualization.

        Returns
        -------
        G : nx.DiGraph
            A graph with neurons and labels as nodes.

        Examples
        --------
        >>> import cocoa as cc
        >>> import networkx as nx
        >>> G = cc.MaleCNS().compile_label_graph()
        >>> nx.write_gml(G, "MCNS_label_graph.gml", stringizer=str)

        """
        assert which_neurons in ("self", "all"), "Invalid `which_neurons`"

        ann = self.get_annotations()

        # Subset to the neurons in this dataset
        if which_neurons == "self":
            if not len(self):
                raise ValueError("No neurons in dataset")
            ann = ann[ann.bodyId.isin(self.neurons)].copy()

        # Drop some known bad types
        if exclude_bad_types:
            ann.loc[ann.type.isin(MCNS_BAD_TYPES), "type"] = None

        if "group" in ann.columns:
            # `group` is a body ID of one of the neurons in that group (e.g. 10063)
            # However, that identity neuron often doesn't have the group itself
            # so we need to manually fix that
            groups = (
                ann[ann.group.notnull()]
                .set_index("bodyId")["group"]
                .astype(int)
                .astype(str)
                .to_dict()
            )
            # For each {bodyID: group} also add {group: group}
            groups.update({v: v for v in groups.values()})

            # Rename groups to "mcns_group_{group}"
            groups = {k: f"mcns_group_{v}" for k, v in groups.items()}

            ann["group"] = ann.bodyId.map(groups)

        if "instance" in ann.columns:
            # Instance is a bit of a mixed bag: we can get things like
            # `{bodyID}_L` or `({type})_L`, where the latter is a tentative type
            # which we will ignore for now

            # First get {ID}_L types
            num_inst = ann.instance.str.extract("^([0-9]+)_[LRM]$")
            num_inst.columns = ["instance"]
            num_inst["bodyId"] = ann.bodyId.values
            num_inst = num_inst[num_inst.instance.notnull()]
            num_inst = num_inst.set_index("bodyId").instance.to_dict()
            num_inst.update({v: v for v in num_inst.values()})

            # Rename these ID instances to "mcns_instance_{ID}"
            num_inst = {k: f"mcns_instance_{v}" for k, v in num_inst.items()}

            ann["instance_clean"] = ann.bodyId.map(num_inst)

        # Initialise graph
        G = nx.DiGraph()

        # Add neuron nodes
        G.add_nodes_from(ann.bodyId, type="neuron")

        # Add mappings to primary label
        prim = ann[ann.type.notnull()]
        G.add_edges_from(zip(prim.bodyId, prim.type))

        # Backfill `type` with `flywire_type` if `type` is missing
        sec = ann[ann.type.isnull() & ann.flywire_type.notnull()]
        G.add_edges_from(zip(sec.bodyId, sec.flywire_type))

        # Add mappings to tertiary and quaternary labels
        if "group" in ann.columns:
            tert = ann[
                ~ann.bodyId.isin(prim.bodyId)
                & ~ann.bodyId.isin(sec.bodyId)
                & ann.group.notnull()
            ]
            G.add_edges_from(zip(tert.bodyId, tert.group))
        else:
            tert = ann.iloc[0:0]

        if "instance_clean" in ann.columns:
            quart = ann[
                ~ann.bodyId.isin(prim.bodyId)
                & ~ann.bodyId.isin(sec.bodyId)
                & ~ann.bodyId.isin(tert.bodyId)
                & ann.instance_clean.notnull()
            ]
            G.add_edges_from(zip(quart.bodyId, quart.instance_clean))

        # Add synonyms:
        syn = (
            ann.loc[ann.type.notnull() & ann.flywire_type.notnull(),]
            .groupby(["type", "flywire_type"], as_index=False)
            .size()
        )
        syn = syn[syn.type != syn.flywire_type]
        G.add_weighted_edges_from(zip(syn.type, syn.flywire_type, syn["size"]))
        # 2. Take care of compound types in both FlyWire and cell type columns (probably the former)
        comp = np.append(
            ann[ann.type.str.contains(",", na=False)].type.values,
            ann[ann.flywire_type.str.contains(",", na=False)].flywire_type.values,
        )
        for c, count in zip(*np.unique(comp, return_counts=True)):
            for c2 in c.split(","):
                G.add_edge(c.strip(), c2.strip(), weight=count)

        # For known antonyms (i.e. labels that are the same in another dataset but do not indicate matches)
        # we will use the node properties to indicate which datasets it must not be matched against.
        # For example:
        # G.nodes['node']['antonyms_in'] = ("MCNS")

        if collapse_neurons:
            G = collapse_neuron_nodes(G)

        return G

    def compile(self):
        """Compile connectivity vector."""
        client = _get_neuprint_mcns_client()

        x = self.neurons.astype(np.int64)

        if not len(x):
            raise ValueError("No body IDs provided")

        if self.use_types:
            # Types is a {bodyId: type} dictionary
            types = _get_mcns_types(
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
                    targets=neu.NeuronCriteria(bodyId=x, client=client), rois=self.rois, client=client
                )
            if self.exclude_queries:
                us = us[~us.bodyId_pre.isin(x)]
            us.rename(
                {"bodyId_pre": "pre", "bodyId_post": "post"}, axis=1, inplace=True
            )
            if self.use_types:
                us = _add_types(
                    us,
                    types=types,
                    col="pre",
                    sides=None if not self.use_sides else _get_mcns_sides(source=self.meta_source),
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
                    sources=neu.NeuronCriteria(bodyId=x, client=client), rois=self.rois, client=client
                )
            if self.exclude_queries:
                ds = ds[~ds.bodyId_post.isin(x)]
            ds.rename(
                {"bodyId_pre": "pre", "bodyId_post": "post"}, axis=1, inplace=True
            )
            if self.use_types:
                ds = _add_types(
                    ds,
                    types=types,
                    col="post",
                    sides=None if not self.use_sides else _get_mcns_meta(),
                    sides_rel=True if self.use_sides == "relative" else False,
                )
            # print("Done!")

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

        # Keep track of whether this used types and side
        self.edges_types_used_ = self.use_types
        self.edges_sides_used_ = self.use_sides

        return self


def _collapse_connectivity_types(type_dict, source="clio"):
    """Remove connectivity type suffixes from {ID: type} dictionary."""
    type_dict = type_dict.copy()
    hb_meta = _get_mcns_meta(source=source)
    cn2morph = hb_meta.set_index("type").morphology_type.to_dict()
    for k, v in type_dict.items():
        new_v = ",".join([cn2morph.get(t, t) for t in v.split(",")])
        type_dict[k] = new_v
    return type_dict
