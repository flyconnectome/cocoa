import numpy as np
import pandas as pd
import neuprint as neu

from .core import DataSet
from .utils import (
    _get_hemibrain_meta,
    _get_neuprint_client,
    _is_int,
    _get_hemibrain_types,
    _get_hb_sides,
    _add_types,
)

__all__ = ["Hemibrain"]

itable = None
otable = None


class Hemibrain(DataSet):
    """Hemibrain dataset.

    Parameters
    ----------
    label :         str
                    A label used for reporting, plotting, etc.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type. This will use `type`, not
                    `morphology_type`.
    use_sides :     bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                        `contra` depending on the side of the connected neuron
    exclude_queries :  bool
                    If True (default), will exclude connections between query
                    neurons from the connectivity vector.
    live_annot :    bool
                    By False (default), will download (and cache) annotations
                    from the Schlegel et al. data repo at
                    https://github.com/flyconnectome/flywire_annotations. If
                    True, will pull from a table where we stage annotations
                    - for internal use only.

    """
    _NGL_LAYER = HEMIBRAIN_MINIMAL_SCENE

    def __init__(
        self,
        label="Hemibrain",
        upstream=True,
        downstream=True,
        use_types=True,
        use_sides=False,
        exclude_queries=True,
        live_annot=False,
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries
        self.live_annot = live_annot

    def _add_neurons(self, x, exact=False, left=False, right=True):
        """Turn `x` into hemibrain body IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if isinstance(x, str) and "," in x:
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
            meta = _get_hemibrain_meta(live=self.live_annot)

            if exact:
                filt = (meta.type == x) | (meta.morphology_type == x)
            else:
                filt = meta.type.str.contains(
                    x, na=False
                ) | meta.morphology_type.str.contains(x, na=False)

            if not left:
                filt = filt & (meta.side != "left")
            if not right:
                filt = filt & (meta.side != "right")

            ids = meta.loc[filt, "bodyId"].values.astype(np.int64).tolist()

        return np.unique(np.array(ids, dtype=np.int64))

    def get_labels(self, x):
        """Fetch labels for given IDs."""
        if not isinstance(x, (list, np.ndarray)):
            x = []
        x = np.asarray(x).astype(np.int64)

        # Fetch all types for this version
        types = _get_hemibrain_types(add_side=False, live=self.live_annot)

        return np.array([types.get(i, i) for i in x])


    def label_exists(self, x):
        """Check if label(s) exists in dataset."""
        x = np.asarray(x)

        types = _get_hemibrain_types(add_side=False, live=self.live_annot)
        morph_types = _get_hemibrain_types(add_side=False, use_morphology_type=True, live=self.live_annot)
        all_types = np.unique(
            np.append(list(types.values()), list(morph_types.values()))
        )

        return np.isin(x, list(all_types))

    def compile(self):
        """Compile connectivity vector."""
        client = _get_neuprint_client()

        x = self.neurons.astype(np.int64)

        if not len(x):
            raise ValueError("No body IDs provided")

        if self.use_types:
            # Types is a {bodyId: type} dictionary
            types = _get_hemibrain_types(add_side=False, live=self.live_annot)
            # For cases where {'AVLP123': 'AVLP123,AVLP323'} we need to change
            # # {bodyId: 'AVLP123'} -> {bodyId: 'AVLP123,AVLP323'}
            # types = {k: collapse_types.get(v, v) for k, v in types.items()}
            # For cases where {12345: '12345,56788'} (i.e. new types)
            # types.update(collapse_types)

        # Fetch hemibrain vectors
        if self.upstream:
            _, us = neu.fetch_adjacencies(
                targets=neu.NeuronCriteria(bodyId=x), client=client
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

        if self.downstream:
            _, ds = neu.fetch_adjacencies(
                sources=neu.NeuronCriteria(bodyId=x), client=client
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

        return self
