import numpy as np
import seaserpent as ss

from .cave import CaveDataset

__all__ = ["Aedes"]


class Aedes(CaveDataset):
    """Mosquito (Aedes aegypti) brain dataset.

    Note: this dataset is currently private and is not accessible to the public.

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
    cn_object :     str | pd.DataFrame, optional
                    Either a DataFrame or path to a `.feather` connectivity file which
                    will be used instead of fetching connectivity from CAVE. Must
                    contain pre ("pre_pt_root_id", "pre_root_id" or "pre"), post
                    ("post_pt_root_id", "post_root_id" or "post") and weight
                    ("syn_count", "weight" or "count") columns.
    materialization : "auto" (default) | "latest" | "live" | int
                    Which materialization to use when fetching connectivity data. If
                    "auto" (default), will try to find a materialization version (including "live")
                    where all IDs in the dataset co-existed.

    """

    _datastack_name = "wclee_aedes_brain"
    _type_cols = ("type",)
    _side_cols = ("side",)

    def __init__(
        self,
        label="Aedes",
        upstream=True,
        downstream=True,
        use_types=False,
        use_sides=False,
        exclude_queries=False,
        cn_object=None,
        materialization="auto",
    ):
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

    def get_annotations(self):
        """Return annotations."""
        if hasattr(self, "_annotations"):
            return self._annotations
        self._annotations = (
            ss.Table("aedes_main", "aedes")
            .to_frame()
            .astype({"root_id": np.int64})
        )
        return self._annotations
