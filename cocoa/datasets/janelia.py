import neuprint as neu
import numpy as np
import pandas as pd

from .core import DataSet
from .ds_utils import _add_types, _is_int

from abc import ABC, abstractproperty


class JaneliaDataSet(DataSet, ABC):
    """Base class for Janelia datasets which use the neuprint/clio API."""

    _roi_col = "roi"

    def __init__(self, label):
        super().__init__(label)

    @abstractproperty
    def neuprint_client(self):
        pass

    def _add_neurons(self, x, exact=True, sides=None):
        """Turn `x` into body IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if not exact and isinstance(x, str) and "," in x:
            x = x.split(",")

        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(ids, self._add_neurons(t, exact=exact, sides=sides))
        elif _is_int(x):
            ids = [int(x)]
        else:
            annot = self.get_annotations()

            if ":" not in x:
                filt = np.zeros(len(annot), dtype=bool)
                for c in self._type_columns:
                    if c not in annot.columns:
                        continue
                    if exact:
                        filt = filt | (annot[c] == x).values
                    else:
                        filt = filt | annot.type.str.contains(
                            x, na=False, case=False
                        )
            else:
                # If this is e.g. "cell_class:L1-5"
                col, val = x.split(":")
                if exact:
                    filt = annot[col] == val
                else:
                    filt = annot[col].str.contains(val, na=False)

            if isinstance(sides, str):
                filt = filt & (annot.side == sides)
            elif isinstance(sides, (tuple, list, np.ndarray)):
                filt = filt & annot.side.isin(sides)
            ids = annot.loc[filt, "bodyId"].unique().astype(np.int64).tolist()

        return np.unique(np.array(ids, dtype=np.int64))

    def get_roi_completeness(self):
        """Get ROI completeness for all neurons in this dataset."""
        return self.neuprint_client.fetch_roi_completeness()

    def get_meshes(self, x):
        """Fetch meshes for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray
                    Body IDs to fetch meshes for.

        """

        import navis.interfaces.neuprint as neu

        return neu.fetch_mesh_neuron(x, client=self.neuprint_client)

    def compile_adjacency(self, collapse_types=False, collapse_rois=True):
        """Compile adjacency between all neurons defined for this dataset."""
        client = self.neuprint_client

        x = self.neurons.astype(np.int64)

        if not len(x):
            raise ValueError("No body IDs provided")

        if self.use_types:
            # Types is a {bodyId: type} dictionary
            if hasattr(self, "types_"):
                types = self.types_
            else:
                types = self.get_labels(None)  # Get all labels

        if isinstance(self.cn_object, pd.DataFrame):
            adj = self.cn_object[
                self.cn_object.bodyId_post.isin(x) & self.cn_object.bodyId_pre.isin(x)
            ]
            if self.rois is not None:
                adj = adj[adj.roi.isin(self.rois)]
            adj = adj.copy()  # avoid SettingWithCopyWarning
        else:
            _, adj = neu.fetch_adjacencies(
                sources=neu.NeuronCriteria(bodyId=x, client=client),
                targets=neu.NeuronCriteria(bodyId=x, client=client),
                rois=self.rois,
                client=client,
            )
        adj.rename({"bodyId_pre": "pre", "bodyId_post": "post"}, axis=1, inplace=True)

        if self.exclude_autapses:
            adj = adj[adj.pre != adj.post].copy()

        if collapse_rois:
            adj = adj.groupby(["pre", "post"], as_index=False).weight.sum()

        if self.use_types:
            adj = _add_types(
                adj,
                types=types,
                col=("pre", "post"),
                sides=None
                if not self.use_sides
                else self.get_sides(None),  # Get all sides
                sides_rel=True if self.use_sides == "relative" else False,
            )

        self.adj_ = adj

        if collapse_types:
            # Make sure to keep "roi" if it still exits
            cols = [c for c in ["pre", "post", "roi"] if c in adj.columns]
            self.adj_ = self.adj_.groupby(cols, as_index=False).weight.sum()

        # Keep track of whether this used types and side
        self.adj_types_used_ = self.use_types
        self.adj_sides_used_ = self.use_sides

        return self
