import os

import numpy as np
import pandas as pd

from fafbseg import flywire

from .core import DataSet
from .utils import (
    _add_types,
    _get_table,
    _get_flywire_types,
    _get_fw_sides,
    _is_int
)

__all__ = ["FlyWire"]

itable = None
otable = None


class FlyWire(DataSet):
    """FlyWire dataset.

    Parameters
    ----------
    label :         str
                    A label.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type. This will use `hemibrain_type` first
                    and where that doesn't exist fall back to `cell_type`.
    use_side  :     bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                        `contra` depending on the side of the connected neuron
    file :          str, optional
                    Filepath to one of the connectivity dumps.

    """

    def __init__(
        self,
        label="FlyWire",
        upstream=True,
        downstream=True,
        use_types=True,
        use_sides=False,
        file=None,
    ):
        assert use_sides in (True, False, "relative")
        if file:
            assert os.path.isfile(file)

        super().__init__(label=label)
        self.file = file
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides

    def _add_neurons(self, x, exact=True, left=True, right=True):
        """Turn `x` into FlyWire root IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if not exact and isinstance(x, str) and "," in x:
            x = x.split(",")

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(
                    ids, self._add_neurons(t, exact=exact, left=left, right=right)
                )
        elif _is_int(x):
            ids = [int(x)]
        else:
            info = _get_table(which="info")
            if exact:
                filt = (info.cell_type == x) | (info.hemibrain_type == x)
            else:
                filt = info.cell_type.str.contains(
                    x, na=False
                ) | info.hemibrain_type.str.contains(x, na=False)

            if not left:
                filt = filt & (info.side != "left")
            if not right:
                filt = filt & (info.side != "right")

            ids = info.loc[filt, "root_id"].values.astype(np.int64).tolist()

            optic = _get_table(which="optic")
            if exact:
                filt = (optic.cell_type == x) | (optic.hemibrain_type == x)
            else:
                filt = optic.cell_type.str.contains(
                    x, na=False
                ) | optic.hemibrain_type.str.contains(x, na=False)

            if not left:
                filt = filt & (optic.side != "left")
            if not right:
                filt = filt & (optic.side != "right")

            ids += optic.loc[filt, "root_id"].values.astype(np.int64).tolist()

        return np.unique(np.array(ids, dtype=np.int64))

    def get_labels(self, x):
        """Fetch labels for given IDs."""
        if not isinstance(x, (list, np.ndarray)):
            x = []
        x = np.asarray(x).astype(np.int64)

        # Find a matching materialization version
        mat = flywire.utils.find_mat_version(x)

        # Fetch all types for this version
        types = _get_flywire_types(mat, add_side=False)

        return np.array([types.get(i, i) for i in x])

    def compile(self):
        """Compile edges."""
        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(int)

        us, ds = None, None
        if self.file:
            # Extract mat version from filename e.g. "syn_proof_[...]_587.feather"
            us_mat = ds_mat = int(self.file.split("_")[-1].split(".")[0])

            # Check if root IDs existed at the time of the synapse dump
            il = flywire.is_latest_root(x, timestamp=f"mat_{us_mat}")
            if any(~il):
                raise ValueError(
                    "Some root IDs did not exist at the time of the "
                    f"synapse dump (mat {us_mat}): {x[~il]}"
                )

            cn = pd.read_feather(self.file).rename(
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
                    x, upstream=True, downstream=False, proofread_only=True
                )
                us_mat = us.attrs["materialization"]
            if self.downstream:
                ds = flywire.fetch_connectivity(
                    x, upstream=False, downstream=True, proofread_only=True
                )
                ds_mat = ds.attrs["materialization"]

        # For grouping by type simple replace pre and post IDs with their types
        # -> well aggregate later
        if self.use_types:
            if self.upstream:
                us = _add_types(
                    us,
                    types=_get_flywire_types(us_mat, add_side=False),
                    col="pre",
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else _get_fw_sides(us_mat),
                    sides_rel=True if self.use_sides == "relative" else False,
                )

            if self.downstream:
                ds = _add_types(
                    ds,
                    types=_get_flywire_types(ds_mat, add_side=False),
                    col="post",
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else _get_fw_sides(ds_mat),
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
