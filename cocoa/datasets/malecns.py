import numpy as np
import pandas as pd
import neuprint as neu

from .core import DataSet
from .utils import (
    _get_mcns_meta,
    _get_neuprint_mcns_client,
    _is_int,
    _get_mcns_types,
    _get_hb_sides,
    _add_types,
)

__all__ = ["MaleCNS"]

itable = None
otable = None


class MaleCNS(DataSet):
    """Male CNS dataset.

    Parameters
    ----------
    label :         str
                    A label.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type.
    use_side :      bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                        `contra` depending on the side of the connected neuron

    """

    def __init__(
        self,
        label="maleCNS",
        upstream=True,
        downstream=True,
        use_types=True,
        use_sides=False,
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides

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
            meta = _get_mcns_meta()
            if right_only:
                ids = meta.loc[
                    (meta.type == x) & (meta.side == "right"),
                    "bodyid",
                ].values.astype(np.int64)
            else:
                ids = meta.loc[
                    (meta.type == x), "bodyid"
                ].values.astype(np.int64)

        return np.unique(np.array(ids, dtype=np.int64))

    def get_labels(self, x):
        """Fetch labels for given IDs."""
        if not isinstance(x, (list, np.ndarray)):
            x = []
        x = np.asarray(x).astype(np.int64)

        # Fetch all types for this version
        types = _get_mcns_types(add_side=False)

        return np.array([types.get(i, i) for i in x])

    def compile(self):
        """Compile connectivity vector."""
        client = _get_neuprint_mcns_client()

        x = self.neurons.astype(np.int64)

        if not len(x):
            raise ValueError("No body IDs provided")

        if self.use_types:
            # Types is a {bodyId: type} dictionary
            types = _get_mcns_types(add_side=False)
            # For cases where {'AVLP123': 'AVLP123,AVLP323'} we need to change
            # # {bodyId: 'AVLP123'} -> {bodyId: 'AVLP123,AVLP323'}
            # types = {k: collapse_types.get(v, v) for k, v in types.items()}
            # For cases where {12345: '12345,56788'} (i.e. new types)
            # types.update(collapse_types)

        # Fetch hemibrain vectors
        if self.upstream:
            print('Fetching upstream connectivity... ', end='', flush=True)
            _, us = neu.fetch_adjacencies(
                targets=neu.NeuronCriteria(bodyId=x), client=client
            )
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
            print('Done!')

        if self.downstream:
            print('Fetching downstream connectivity... ', end='', flush=True)
            _, ds = neu.fetch_adjacencies(
                sources=neu.NeuronCriteria(bodyId=x), client=client
            )
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
            print('Done!')

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


def _collapse_connectivity_types(type_dict):
    """Remove connectivity type suffixes from {ID: type} dictionary."""
    type_dict = type_dict.copy()
    hb_meta = _get_mcns_meta()
    cn2morph = hb_meta.set_index("type").morphology_type.to_dict()
    for k, v in type_dict.items():
        new_v = ",".join([cn2morph.get(t, t) for t in v.split(",")])
        type_dict[k] = new_v
    return type_dict
