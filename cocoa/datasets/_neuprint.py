import neuprint as neu
import numpy as np
import pandas as pd

from .core import DataSet
from .ds_utils import _add_types

from abc import ABC, abstractproperty


class NeuprintDataSet(DataSet, ABC):
    """Base class for datasets that use the neuprint API."""

    _roi_col = "roi"

    def __init__(self, label):
        super().__init__(label)

    @abstractproperty
    def neuprint_client(self):
        pass

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
