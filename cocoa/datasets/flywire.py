import copy

import numpy as np
import pandas as pd

from pathlib import Path
from fafbseg import flywire

from .core import DataSet
from .scenes import FLYWIRE_MINIMAL_SCENE, FLYWIRE_FLAT_MINIMAL_SCENE
from .utils import (
    _add_types,
    _get_fw_types,
    _load_live_flywire_annotations,
    _load_static_flywire_annotations,
    _get_fw_sides,
    _is_int,
)

__all__ = ["FlyWire"]

itable = None
otable = None


class FlyWire(DataSet):
    """FlyWire dataset.

    Uses type annotations from Schlegel et al., bioRxiv (2023).

    Parameters
    ----------
    label :         str
                    A label used for reporting, plotting, etc.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type. This will use `cell_type` first
                    and where that doesn't exist fall back to `hemibrain_type`.
    use_side  :     bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                          `contra` depending on the side of the connected neuron
    exclude_queries :  bool
                    If True (default), will exclude connections between query
                    neurons from the connectivity vector.
    cn_file :       str, optional
                    Filepath to one of the connectivity dumps. Using this is
                    faster than querying the CAVE backend for connectivity.
    live_annot :    bool
                    If False (default), will download (and cache) annotations
                    from the Schlegel et al. data repo at
                    https://github.com/flyconnectome/flywire_annotations. If
                    True, will pull from a table where we stage annotations
                    - this requires special permissions and is for internal use
                    only.
    materialization : int | "live"
                    Which materialization to use. If `cn_file` is provided,
                    must match that materialization version.

    """

    def __init__(
        self,
        label="FlyWire",
        upstream=True,
        downstream=True,
        use_types=True,
        use_sides=False,
        exclude_queries=True,
        cn_file=None,
        live_annot=False,
        materialization=630,
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.cn_file = cn_file
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries
        self.live_annot = live_annot
        self.materialization = materialization

        if self.cn_file:
            self.cn_file = Path(self.cn_file).expanduser()
            if not self.cn_file.is_file():
                raise ValueError(f'"{self.cn_file}" is not a valid file')
            file_mat = int(str(self.cn_file).split("_")[-1].split(".")[0])
            if file_mat != self.materialization:
                raise ValueError(
                    "Connectivity file name suggests it is from "
                    f"materialization {file_mat} but dataset was "
                    f"initialized with `materialization={self.materialization}`"
                )

    def _add_neurons(self, x, exact=True, sides=None):
        """Turn `x` into FlyWire root IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if not exact and isinstance(x, str) and "," in x:
            x = x.split(",")

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(ids, self._add_neurons(t, exact=exact, sides=sides))
        elif _is_int(x):
            ids = [int(x)]
        else:
            if self.live_annot:
                annot = _load_live_flywire_annotations(mat=self.materialization)
            else:
                annot = _load_static_flywire_annotations(mat=self.materialization)

            if ":" not in x:
                if exact:
                    filt = (annot.cell_type == x) | (annot.hemibrain_type == x)
                else:
                    filt = annot.cell_type.str.contains(
                        x, na=False
                    ) | annot.hemibrain_type.str.contains(x, na=False)
            else:
                # If this is e.g. "cell_class:L1-5"
                col, val = x.split(":")
                if exact:
                    filt = annot[col] == val
                else:
                    filt = annot[col].str.contains(val)

            if isinstance(sides, str):
                filt = filt & (annot.side == sides)
            elif isinstance(sides, (tuple, list, np.ndarray)):
                filt = filt & annot.side.isin(sides)
            ids = annot.loc[filt, "root_id"].unique().astype(np.int64).tolist()

        return np.unique(np.array(ids, dtype=np.int64))

    def get_labels(self, x, verbose=False):
        """Fetch labels for given IDs."""
        if not isinstance(x, (list, np.ndarray)):
            x = []
        x = np.asarray(x).astype(np.int64)

        types = _get_fw_types(live=self.live_annot, mat=self.materialization)

        return np.array([types.get(i, i) for i in x])

    def get_ngl_scene(self, flat=False, open=False):
        """Return a minimal neuroglancer scene for this dataset.

        Parameters
        ----------
        flat :      bool
                    If True will use the flat 630 segmentation instead of the
                    production dataset.
        open
        """
        if not flat:
            return copy.deepcopy(FLYWIRE_MINIMAL_SCENE)
        else:
            return copy.deepcopy(FLYWIRE_FLAT_MINIMAL_SCENE)

    def label_exists(self, x):
        """Check if labels exists in dataset."""
        x = np.asarray(x)

        types = _get_fw_types(live=self.live_annot, mat=self.materialization)

        all_types = np.unique(list(types.values()))
        all_types = np.append(
            all_types, np.unique([l for t in all_types for l in t.split(",")])
        )

        return np.isin(x, list(all_types))

    def compile(self):
        """Compile edges."""
        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(int)

        mat = self.materialization
        il = flywire.is_latest_root(x, timestamp=f"mat_{mat}")
        if any(~il):
            raise ValueError(
                "Some of the root IDs did not exist at the specified "
                f"materialization ({mat}): {x[~il]}"
            )

        us, ds = None, None
        if self.cn_file:
            cn = pd.read_feather(self.cn_file).rename(
                {
                    "pre_pt_root_id": "pre",
                    "post_pt_root_id": "post",
                    "syn_count": "weight",
                },
                axis=1,
            )
            if self.upstream:
                us = cn[cn.post.isin(x)]
                us = us.groupby(["pre", "post"], as_index=False).weight.sum()

            if self.downstream:
                ds = cn[cn.pre.isin(x)]
                ds = ds.groupby(["pre", "post"], as_index=False).weight.sum()
        else:
            if self.upstream:
                us = flywire.fetch_connectivity(
                    x,
                    upstream=True,
                    downstream=False,
                    proofread_only=True,
                    filtered=True,
                    min_score=50,
                    progress=False,
                    mat=mat,
                )
            if self.downstream:
                ds = flywire.fetch_connectivity(
                    x,
                    upstream=False,
                    downstream=True,
                    proofread_only=True,
                    filtered=True,
                    min_score=50,
                    progress=False,
                    mat=mat,
                )

        if self.exclude_queries:
            if self.upstream:
                us = us[~us.pre.isin(x)]
            if self.downstream:
                ds = ds[~ds.pre.isin(x)]

        # For grouping by type simply replace pre and post IDs with their types
        # -> we'll aggregate later
        if self.use_types:
            fw_types = _get_fw_types(mat, add_side=False, live=self.live_annot)
            fw_sides = _get_fw_sides(mat, live=self.live_annot)
            if self.upstream:
                us = _add_types(
                    us,
                    types=fw_types,
                    col="pre",
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else fw_sides,
                    sides_rel=True if self.use_sides == "relative" else False,
                )

            if self.downstream:
                ds = _add_types(
                    ds,
                    types=fw_types,
                    col="post",
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else fw_sides,
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
            self.edges_ = us.groupby(["pre", "post"]).weight.sum()
        elif self.downstream:
            self.edges_ = ds.groupby(["pre", "post"]).weight.sum()
        else:
            raise ValueError("`upstream` and `downstream` must not both be False")

        # Translate morphology types into connectivity types
        # This makes it easier to align with hemibrain
        # self.connectivity_.columns = _morphology_to_connectivity_types(
        #    self.connectivity_.columns
        # )

        return self
