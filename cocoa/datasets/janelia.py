import neuprint as neu
import numpy as np
import pandas as pd

from pathlib import Path

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

    @property
    def cn_object(self):
        return getattr(self, "_cn_object", None)

    @cn_object.setter
    def cn_object(self, value):
        if value is not None:
            if isinstance(value, (str, Path)):
                value = Path(value).expanduser()
                if not value.is_file():
                    raise ValueError(f'"{self.cn_object}" is not a valid file')
                self._cn_object = pd.read_feather(value)
            elif isinstance(value, pd.DataFrame):
                self._cn_object = value
            else:
                raise ValueError("`cn_object` must be a path, a DataFrame or `None`")

            # Make sure we have the right column names: "bodyId_pre", "bodyId_post", "roi", "weight"
            self._cn_object = self._cn_object.rename(
                columns={"body_pre": "bodyId_pre", "body_post": "bodyId_post"}
            )
        else:
            self._cn_object = None

    def add_neurons(self, x, regex="auto", sides=None):
        """Add neurons to dataset.

        Parameters
        ----------
        x :         int | str | list | np.ndarray | pd.Series | None
                    Root IDs or cell types to add. Can also use "{column}:{value}" to filter
                    for values in given column.
        regex :     "auto" | bool
                    Whether strings are interpreted as regular expressions.
                    If "auto" (default), will treat strings starting with "/" as regex.
        sides :     str | iterable | None
                    If provided, will only add neurons on the given side(s).

        """
        new_neurons = self._parse_ids(x, regex=regex, sides=sides)

        if len(new_neurons):
            self.neurons = np.unique(np.append(self.neurons, new_neurons))
        else:
            print(f'Warning: No neurons found for query "{x}"')

        return self

    def _parse_ids(self, x, regex="auto", sides=None):
        """Turn `x` into body IDs.

        Parameters
        ----------
        x :         int | str | list | np.ndarray | pd.Series | None
                    Body IDs or cell types to add. Strings will be matched against all
                    available type columns. You can also use "{column}:{value}" to filter
                    for values in given column.
        regex :     "auto" | bool
                    Whether strings are interpreted as regular expressions.
                    If "auto" (default), will treat strings starting with "/" as regex.
        sides :     str | iterable | None
                    If provided, will only add neurons on the given side(s).

        """
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if regex == "auto" and isinstance(x, str):
            regex = x.startswith("/")
            x = x[1:] if regex else x

        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(ids, self._parse_ids(t, regex=regex, sides=sides))
        elif _is_int(x):
            ids = [int(x)]
        else:
            annot = self.get_annotations()

            if ":" not in x:
                filt = np.zeros(len(annot), dtype=bool)
                for c in self._type_columns:
                    if c not in annot.columns:
                        continue
                    if not regex:
                        filt = filt | (annot[c] == x).values
                    else:
                        filt = filt | annot.type.str.contains(x, na=False, case=False)
            else:
                # If this is e.g. "cell_class:L1-5"
                col, val = x.split(":")
                if not regex:
                    filt = annot[col] == val
                else:
                    filt = annot[col].str.contains(val, na=False)

            if isinstance(sides, str):
                filt = filt & (annot.side == sides)
            elif isinstance(sides, (tuple, list, np.ndarray)):
                filt = filt & annot.side.isin(sides)
            ids = annot.loc[filt, "bodyId"].unique().astype(np.int64).tolist()

        if not len(ids):
            print(f'Warning: No neurons found for query "{x}"')

        return ids

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
