import copy

import numpy as np
import pandas as pd
import neuprint as neu

from pathlib import Path

from .core import DataSet
from .scenes import _get_mcns_scene
from .utils import (
    _get_mcns_meta,
    _get_neuprint_mcns_client,
    _is_int,
    _get_mcns_types,
    _get_hb_sides,
    _add_types,
    _get_clio_client,
    _parse_neuprint_roi,
)

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
                        Whether to group by type.
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
        use_types=True,
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
            self.rois = _parse_neuprint_roi(rois, client = _get_neuprint_mcns_client())
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
                    (meta.type == x) & (meta.side == "right"),
                    "bodyid",
                ].values.astype(np.int64)
            else:
                ids = meta.loc[(meta.type == x), "bodyid"].values.astype(np.int64)

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

    def get_annotations(self):
        """Return annotations."""
        return _get_mcns_meta(source=self.meta_source)

    def get_labels(self, x):
        """Fetch labels for given IDs."""
        if not isinstance(x, (list, np.ndarray)):
            x = []
        x = np.asarray(x).astype(np.int64)

        # Fetch all types for this version
        types = _get_mcns_types(add_side=False, source=self.meta_source)

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

        types = _get_mcns_types(add_side=False, source=self.meta_source)
        all_types = np.unique(list(types.values()))

        return np.isin(x, list(all_types))

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
                    sides=None if not self.use_sides else _get_hb_sides(),
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
                    sides=None if not self.use_sides else _get_hb_sides(),
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
