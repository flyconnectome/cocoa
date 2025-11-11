import seaserpent as ss

from .cave import CaveDataset

__all__ = ["Aedes"]


class Aedes(CaveDataset):
    """Mosquite (Aedes aegypti) brain dataset.

    Note: this dataset is currently private and is not accessible to the public.

    Parameters
    ----------
    label :         str
                    A label used for reporting, plotting, etc.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type. This will use `cell_type` first
                    and where that doesn't exist fall back to `hemibrain_type`.
                    Note that this may be overwritten when used in the context
                    of a `cocoa.Clustering`.
    use_side  :     bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                          `contra` depending on the side of the connected neuron
    exclude_queries :  bool
                    If True (default), will exclude connections between query
                    neurons from the connectivity vector.
    materialization : "auto" (default) | "latest" | "live" | int
                    Which materialization to use when fetching connectivity data. If
                    "auto" (default), will try to find a materialization version (including "live")
                    where all IDs in the dataset co-existed.

    """

    _roi_col = "neuropil"
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
        materialization="auto",
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.materialization = materialization  # must be set before `cn_object`
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries

    def copy(self):
        """Make copy of dataset."""
        x = type(self)(label=self.label)
        x.neurons = self.neurons.copy()
        x.upstream = self.upstream
        x.downstream = self.downstream
        x.use_types = self.use_types
        x.use_sides = self.use_sides
        x.exclude_queries = self.exclude_queries
        x.materialization = self.materialization

        return x

    def get_annotations(self):
        """Return annotations."""
        if hasattr(self, "_annotations"):
            return self._annotations
        self._annotations = ss.Table("aedes_main", "aedes").to_frame()
        return self._annotations

