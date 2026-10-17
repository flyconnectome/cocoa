from .cave import CaveDataset
from .ds_utils import (
    BANC_CAVE_DATASTACK,
    _load_live_banc_annotations,
    _load_cave_banc_annotations,
)

__all__ = ["BANC"]


class BANC(CaveDataset):
    """Brain-and-Nerve-cord dataset.

    Parameters
    ----------
    label :         str
                    A label used for reporting, plotting, etc.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type. Note that this may be overwritten
                    when used in the context of a `cocoa.Clustering`.
    use_side  :     bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                          `contra` depending on the side of the connected neuron
    exclude_queries :  bool
                    If True, will exclude connections between query
                    neurons from the connectivity vector.
    meta_source :   "cave" | "flytable"
                    Source for annotations. If "cave" (default), will download (and cache)
                    annotations from the `codex_annotations` table.
                    The "flytable" option is for internal use only and requires special
                    permissions.
    cn_object :     str | pd.DataFrame, optional
                    Either a DataFrame or path to a `.feather` connectivity file which
                    will be used instead of fetching connectivity from CAVE. Must
                    contain pre ("pre_pt_root_id", "pre_root_id" or "pre"), post
                    ("post_pt_root_id", "post_root_id" or "post") and weight
                    ("syn_count", "weight" or "count") columns. Root IDs must
                    match the materialization.
    materialization : "latest" (default) | int | "live"
                    Which materialization to use when fetching connectivity/annotations data.
                    If "latest" (default), will use the latest materialization.
                    "live" requires `meta_source="flytable"`.

    References
    ----------
    Distributed control circuits across a brain-and-cord connectome
    Alexander Shakeel Bates, Jasper S. Phelps, Minsu Kim [...] Rachel I. Wilson, Wei-Chung Allen Lee
    bioRxiv 2025.07.31.667571; doi: https://doi.org/10.1101/2025.07.31.667571

    """

    _datastack_name = BANC_CAVE_DATASTACK
    _type_cols = ("type",)
    _side_cols = ("side",)
    _copy_attrs = CaveDataset._copy_attrs + ("meta_source",)

    def __init__(
        self,
        label="BANC",
        upstream=True,
        downstream=True,
        use_types=False,
        use_sides=False,
        exclude_queries=False,
        meta_source="cave",
        cn_object=None,
        materialization="latest",
    ):
        assert meta_source in (
            "cave",
            "flytable",
        ), "`meta_source` must be 'cave' or 'flytable'"
        # N.B. must be set before `materialization` (see `_check_materialization`)
        self.meta_source = meta_source
        super().__init__(
            label=label,
            upstream=upstream,
            downstream=downstream,
            use_types=use_types,
            use_sides=use_sides,
            exclude_queries=exclude_queries,
            cn_object=cn_object,
            materialization=materialization,
        )

    def _check_materialization(self, value):
        """Raise if `value` isn't supported by this dataset."""
        # Annotations are tied to a materialization and are required before
        # we know the neurons -> we can't pick a materialization automatically
        if value == "auto":
            raise ValueError('BANC does not support `materialization="auto"`.')
        if value == "live" and self.meta_source != "flytable":
            raise ValueError(
                '`materialization="live"` requires `meta_source="flytable"`.'
            )

    def get_annotations(self):
        """Return annotations."""
        if not hasattr(self, "_annotations"):
            if self.meta_source == "cave":
                self._annotations = _load_cave_banc_annotations(
                    mat=self.materialization
                )
            elif self.meta_source == "flytable":
                self._annotations = _load_live_banc_annotations(
                    mat=self.materialization
                )
        return self._annotations

