import numpy as np
import pandas as pd
import neuprint as neu
import networkx as nx

from .janelia import JaneliaDataSet
from .scenes import HEMIBRAIN_MINIMAL_SCENE
from .ds_utils import (
    _get_hemibrain_meta,
    _get_neuprint_hemibrain_client,
    _is_int,
    _get_hemibrain_types,
    _get_hb_sides,
    _add_types,
)
from ..utils import collapse_neuron_nodes

__all__ = ["Hemibrain"]


class Hemibrain(JaneliaDataSet):
    """Hemibrain dataset.

    Parameters
    ----------
    label :         str
                    A label used for reporting, plotting, etc.
    up/downstream : bool
                    Whether to use up- and/or downstream connectivity.
    use_types :     bool
                    Whether to group by type. This will use `type`, not
                    `morphology_type`. Note that this may be overwritten
                    when used in the context of a `cocoa.Clustering`.
    use_sides :     bool | 'relative'
                    Only relevant if `group_by_type=True`:
                        - if `True`, will split cell types into left/right/center
                        - if `relative`, will label cell types as `ipsi` or
                        `contra` depending on the side of the connected neuron
    exclude_queries :  bool
                    If True (default), will exclude connections between query
                    neurons from the connectivity vector.
    live_annot :    bool
                    If False (default), will download (and cache) annotations
                    from the Schlegel et al. data repo at
                    https://github.com/flyconnectome/flywire_annotations. If
                    True, will pull from a table where we stage annotations
                    - this requires special permissions and is for internal use
                    only.
    cn_object :     str | pd.DataFrame
                    Either a DataFrame or path to a `.feather` connectivity file which
                    will be loaded into a DataFrame. The DataFrame is expected to
                    come from `neuprint.fetch_adjacencies` and include all relevant
                    IDs.

    """

    _NGL_LAYER = HEMIBRAIN_MINIMAL_SCENE
    _flybrains_space = "JRCFIB2018Fraw"
    _type_columns = ["type", "morphology_type"]

    def __init__(
        self,
        label="Hemibrain",
        upstream=True,
        downstream=True,
        use_types=False,
        use_sides=False,
        exclude_queries=False,
        live_annot=False,
        cn_object=None,
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries
        self.live_annot = live_annot
        self.cn_object = cn_object

    def _add_neurons(self, x, exact=False, sides=("left", "right")):
        """Turn `x` into hemibrain body IDs."""
        if isinstance(x, type(None)):
            return np.array([], dtype=np.int64)

        if isinstance(x, pd.Series):
            x = x.values

        if isinstance(x, str) and "," in x:
            x = x.split(",")

        if isinstance(x, (list, np.ndarray, set, tuple)):
            ids = np.array([], dtype=np.int64)
            for t in x:
                ids = np.append(ids, self._add_neurons(t, exact=exact, sides=sides))
        elif _is_int(x):
            ids = [int(x)]
        else:
            annot = self.get_annotations()

            if ":" not in x:
                if exact:
                    filt = (annot.type == x) | (annot.morphology_type == x)
                else:
                    filt = annot.type.str.contains(
                        x, na=False
                    ) | annot.morphology_type.str.contains(x, na=False, case=False)
            else:
                # If this is e.g. "type:L1-5"
                col, val = x.split(":")
                if exact:
                    filt = annot[col] == val
                else:
                    filt = annot[col].str.contains(val, na=False, case=False)

            if isinstance(sides, str):
                filt = filt & (annot.side == sides)
            elif isinstance(sides, (tuple, list, np.ndarray)):
                filt = filt & annot.side.isin(sides)

            ids = annot.loc[filt, "bodyId"].values.astype(np.int64).tolist()

        return np.unique(np.array(ids, dtype=np.int64))

    @property
    def neuprint_client(self):
        """Return neuprint client."""
        return _get_neuprint_hemibrain_client()

    @classmethod
    def hemisphere(cls, hemisphere, label=None, **kwargs):
        """Generate a dataset for given hemibrain hemisphere.

        Parameters
        ----------
        hemisphere :    str
                        "left" or "right"
        label :         str, optional
                        Label for the dataset. If not provided will generate
                        one based on the hemisphere.
        **kwargs
            Additional keyword arguments for the dataset.

        Returns
        -------
        ds :            Hemibrain
                        A dataset for the specified hemisphere.

        """
        assert hemisphere in ("left", "right"), f"Invalid hemisphere '{hemisphere}'"

        hemisphere = {"left": "L", "right": "R"}[hemisphere]

        if label is None:
            label = f"Hemibrain({hemisphere[0]})"
        ds = cls(label=label, **kwargs)

        ann = ds.get_annotations()
        to_add = ann[(ann.side == hemisphere)].bodyId.values
        ds.add_neurons(to_add)

        return ds

    def copy(self):
        """Make copy of dataset."""
        x = type(self)(label=self.label)
        x.neurons = self.neurons.copy()
        x.upstream = self.upstream
        x.downstream = self.downstream
        x.use_types = self.use_types
        x.use_sides = self.use_sides
        x.exclude_queries = self.exclude_queries
        x.live_annot = self.live_annot
        x.cn_object = self.cn_object

        return x

    def clear_cache(self):
        """Clear cached data (e.g. annotations)."""
        _get_hemibrain_meta.cache_clear()
        _get_hemibrain_types.cache_clear()
        _get_hb_sides.cache_clear()
        print("Cleared cached hemibrain data.")

    def get_annotations(self):
        """Return annotations."""
        return _get_hemibrain_meta(live=self.live_annot).copy()

    def get_all_neurons(self):
        """Get a list of all neurons in this dataset."""
        return self.get_annotations().bodyId.values

    def get_labels(self, x):
        """Fetch labels for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Body IDs to fetch labels for. If `None`, will return all labels.

        """
        # Fetch all types for this version
        types = _get_hemibrain_types(add_side=False, live=self.live_annot)

        if x is None:
            return types

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([types.get(i, i) for i in x])

    def get_sides(self, x):
        """Fetch sides for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Body IDs to fetch labels for. If `None`, will return all labels.

        """
        # Fetch all sides for this version
        sides = _get_hb_sides(live=self.live_annot)

        if x is None:
            return sides

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([sides.get(i, i) for i in x])

    def label_exists(self, x):
        """Check if labels exists in dataset."""
        x = np.asarray(x)

        # This graph contains all possible labels in this dataset,
        # including synonyms and split compound types
        G = self.compile_label_graph(which_neurons="all")

        # Remove the neurons themselves
        G.remove_nodes_from(
            [k for k, v in nx.get_node_attributes(G, "type").items() if v == "neuron"]
        )

        return np.isin(x, list(G.nodes))

    def compile_label_graph(
        self, which_neurons="all", collapse_neurons=False, strict=False
    ):
        """Compile label graph.

        For the hemibrain, this means:
         1. Use the `type` primary labels
         2. Use `morphology_type` as secondary labels (i.e. less granular synonyms)

        Parameters
        ----------
        which_neurons : "all" | "self"
                        Whether to use only the neurons in this
                        dataset or all neurons in the entire hemibrain dataset.
        collapse_neurons : bool
                        If True, will collapse neurons with the same connectivity into
                        a single node. Useful for e.g. visualization.
        strict :        bool
                        If True, will prefix the labels with the type of the label (e.g. "hemibrain:PS008").

        Returns
        -------
        G : nx.DiGraph
            A graph with neurons and labels as nodes.

        """
        assert which_neurons in ("self", "all")

        ann = self.get_annotations()

        # Subset to the neurons in this dataset
        if which_neurons == "self":
            if not len(self):
                raise ValueError("No neurons in dataset")
            ann = ann[ann.bodyId.isin(self.neurons)]

        # Initialise graph
        G = nx.DiGraph()

        # Add neuron nodes
        G.add_nodes_from(ann.bodyId, type="neuron")

        # Add types
        for col in ["type", "morphology_type"]:
            types = ann[ann[col].notnull()]

            if strict:
                types = types.copy()
                types[col] = "hemibrain:" + types[col]

            G.add_edges_from(zip(types.bodyId, types[col]))

        if collapse_neurons:
            G = collapse_neuron_nodes(G)

        return G

    def compile(self):
        """Compile connectivity vector."""
        client = self.neuprint_client

        x = self.neurons.astype(np.int64)

        if not len(x):
            raise ValueError("No body IDs provided")

        if self.use_types:
            # Types is a {bodyId: type} dictionary
            if hasattr(self, "types_"):
                types = self.types_
            else:
                types = _get_hemibrain_types(add_side=False, live=self.live_annot)
            # For cases where {'AVLP123': 'AVLP123,AVLP323'} we need to change
            # # {bodyId: 'AVLP123'} -> {bodyId: 'AVLP123,AVLP323'}
            # types = {k: collapse_types.get(v, v) for k, v in types.items()}
            # For cases where {12345: '12345,56788'} (i.e. new types)
            # types.update(collapse_types)

        # Fetch hemibrain vectors
        if self.upstream:
            _, us = neu.fetch_adjacencies(
                targets=neu.NeuronCriteria(bodyId=x, client=client), client=client
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
                    sides=None
                    if not self.use_sides
                    else _get_hb_sides(live=self.live_annot),
                    sides_rel=True if self.use_sides == "relative" else False,
                )

        if self.downstream:
            _, ds = neu.fetch_adjacencies(
                sources=neu.NeuronCriteria(bodyId=x, client=client), client=client
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
                    sides=None
                    if not self.use_sides
                    else _get_hb_sides(live=self.live_annot),
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
            self.edges_ = us.groupby(["pre", "post"], as_index=False).weight.sum()
        elif self.downstream:
            self.edges_ = ds.groupby(["pre", "post"], as_index=False).weight.sum()
        else:
            raise ValueError("`upstream` and `downstream` must not both be False")

        # Keep track of whether this used types and side
        self.edges_types_used_ = self.use_types
        self.edges_sides_used_ = self.use_sides

        return self
