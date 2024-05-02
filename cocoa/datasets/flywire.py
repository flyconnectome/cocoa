import copy

import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from fafbseg import flywire

from .core import DataSet
from .scenes import FLYWIRE_MINIMAL_SCENE, FLYWIRE_FLAT_MINIMAL_SCENE
from .ds_utils import (
    _add_types,
    _get_fw_types,
    _load_live_flywire_annotations,
    _load_static_flywire_annotations,
    _get_fw_sides,
    _is_int,
)

__all__ = ["FlyWire"]

itable = None
otable = None


class FlyWire(DataSet):
    """FlyWire dataset.

    Uses type annotations from Schlegel et al., bioRxiv (2023).

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
    cn_file :       str, optional
                    Filepath to one of the connectivity dumps. Using this is
                    faster than querying the CAVE backend for connectivity.
    live_annot :    bool
                    If False (default), will download (and cache) annotations
                    from the Schlegel et al. data repo at
                    https://github.com/flyconnectome/flywire_annotations. If
                    True, will pull from a table where we stage annotations
                    - this requires special permissions and is for internal use
                    only.
    materialization : int | "live"
                    Which materialization to use. If `cn_file` is provided,
                    must match that materialization version.

    """

    def __init__(
        self,
        label="FlyWire",
        upstream=True,
        downstream=True,
        use_types=False,
        use_sides=False,
        exclude_queries=False,
        cn_file=None,
        live_annot=False,
        materialization=783,
    ):
        assert use_sides in (True, False, "relative")
        super().__init__(label=label)
        self.cn_file = cn_file
        self.upstream = upstream
        self.downstream = downstream
        self.use_types = use_types
        self.use_sides = use_sides
        self.exclude_queries = exclude_queries
        self.live_annot = live_annot
        self.materialization = materialization

        if self.cn_file is not None:
            self.cn_file = Path(self.cn_file).expanduser()
            if not self.cn_file.is_file():
                raise ValueError(f'"{self.cn_file}" is not a valid file')
            try:
                file_mat = int(str(self.cn_file).split("_")[-1].split(".")[0])
                if file_mat != self.materialization:
                    raise ValueError(
                        "Connectivity file name suggests it is from "
                        f"materialization {file_mat} but dataset was "
                        f"initialized with `materialization={self.materialization}`"
                    )
            except ValueError:
                print("Unable to parse materialization from filename. Please make "
                      "sure the connectivity represents materialization version "
                      f"'{self.materialization}'.")

    def _add_neurons(self, x, exact=True, sides=None):
        """Turn `x` into FlyWire root IDs."""
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
                if exact:
                    filt = (annot.cell_type == x) | (annot.hemibrain_type == x)
                else:
                    filt = annot.cell_type.str.contains(
                        x, na=False
                    ) | annot.hemibrain_type.str.contains(x, na=False)
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
            ids = annot.loc[filt, "root_id"].unique().astype(np.int64).tolist()

        return np.unique(np.array(ids, dtype=np.int64))

    def copy(self):
        """Make copy of dataset."""
        x = type(self)(label=self.label)
        x.neurons = self.neurons.copy()
        x.cn_file = self.cn_file
        x.upstream = self.upstream
        x.downstream = self.downstream
        x.use_types = self.use_types
        x.use_sides = self.use_sides
        x.exclude_queries = self.exclude_queries
        x.live_annot = self.live_annot
        x.materialization = self.materialization

        return x

    def get_annotations(self):
        """Return annotations."""
        if self.live_annot:
            return _load_live_flywire_annotations(mat=self.materialization)
        else:
            return _load_static_flywire_annotations(mat=self.materialization)

    def get_all_neurons(self):
        """Get a list of all neurons in this dataset."""
        return self.get_annotations().root_id.values

    def get_labels(self, x):
        """Fetch labels for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray | None
                    Root IDs to fetch labels for. If `None`, will return all labels.
        """
        types = _get_fw_types(live=self.live_annot, mat=self.materialization)

        if x is None:
            return types

        if not isinstance(x, (list, np.ndarray)):
            x = [x]
        x = np.asarray(x).astype(np.int64)

        return np.array([types.get(i, i) for i in x])

    def get_ngl_scene(self, flat=False, open=False):
        """Return a minimal neuroglancer scene for this dataset.

        Parameters
        ----------
        flat :      bool
                    If True will use the flat 630 segmentation instead of the
                    production dataset.
        open
        """
        if not flat:
            return copy.deepcopy(FLYWIRE_MINIMAL_SCENE)
        else:
            return copy.deepcopy(FLYWIRE_FLAT_MINIMAL_SCENE)

    def label_exists(self, x):
        """Check if labels exists in dataset."""
        x = np.asarray(x)

        # This graph contains all possible labels in this dataset,
        # including synonyms and split compound types
        G = self.compile_label_graph(which_neurons="all")

        # Remove the neurons themselves
        G.remove_nodes_from([k for k, v in nx.get_node_attributes(G, "type").items() if v == "neuron"])

        return np.isin(x, list(G.nodes))

    # Should this be cached and/or turned into a classmethod?
    def compile_label_graph(self, which_neurons="all"):
        """Compile label graph.

        For FlyWire, this means:
         1. Use the `cell_type` back-filled with the `hemibrain_type` as primary labels
         2. Use `hemibrain_type` as secondary labels (i.e. synonyms)
         3. Split hemibrain compound types such that e.g. "PS008,PS009"
            produces two edges: (PS008,PS009 -> PS008) and (PS008,PS009 -> PS009)

        Parameters
        ----------
        which_neurons : "all" | "self"
                        Whether to use only the neurons in this
                        dataset or all neurons in the entire FlyWire dataset.

        Returns
        -------
        G : nx.Graph
            A graph with neurons and labels as nodes.

        """
        assert which_neurons in ("self", "all"), "`which_neurons` must be 'self' or 'all'"

        ann = self.get_annotations()

        # Subset to the neurons in this dataset
        if which_neurons == "self":
            if not len(self):
                raise ValueError("No neurons in dataset")
            ann = ann[ann.root_id.isin(self.neurons)]

        # Initialise graph
        G = nx.Graph()

        # Add neuron nodes
        G.add_nodes_from(ann.root_id, type="neuron")

        # Add mappings to primary label
        prim = ann[ann.cell_type.notnull()]
        G.add_edges_from(zip(prim.root_id, prim.cell_type))

        # Add mappings to secondary label
        sec = ann[ann.cell_type.isnull() & ann.hemibrain_type.notnull()]
        G.add_edges_from(zip(sec.root_id, sec.hemibrain_type))

        # Add synonyms:
        # 1. Take care of cases where e.g. cell type is PS008a but the hemibrain type is PS008
        syn = ann.loc[
            ann.cell_type.notnull() & ann.hemibrain_type.notnull(),
            ["cell_type", "hemibrain_type"],
        ].drop_duplicates()
        syn = syn[syn.cell_type != syn.hemibrain_type]
        G.add_edges_from(zip(syn.cell_type, syn.hemibrain_type))
        # 2. Take care of compound types (both from hemibrain and cell type columns)
        comp = np.append(
            # Note: for cell type we sometimes have types like "(M_adPNm4,M_adPNm5)b" which we will ignore here
            ann[
                ann.cell_type.str.contains(",", na=False)
                & ~ann.cell_type.str.startswith("(", na=False)
            ].cell_type.unique(),
            ann[ann.hemibrain_type.str.contains(",", na=False)].hemibrain_type.unique(),
        )
        for c in comp:
            for c2 in c.split(","):
                G.add_edge(c.strip(), c2.strip())

        # For known antonyms (i.e. labels that are the same in another dataset but do not indicate matches)
        # we will use the node properties to indicate which datasets it must not be matched against.
        # For example:
        # G.nodes['node']['antonyms_in'] = ("MCNS")

        return G

    def compile(self, collapse_types=False):
        """Compile edges."""
        # Make sure we're working on integers
        x = np.asarray(self.neurons).astype(np.int64)

        if self.materialization == "auto":
            self.materialization = mat = flywire.utils.find_mat_version(x)
        else:
            mat = self.materialization
            timestamp = None if mat == "live" else f"mat_{mat}"

            il = flywire.is_latest_root(x, timestamp=timestamp)
            if any(~il):
                raise ValueError(
                    "Some of the root IDs does not exist for the specified "
                    f"materialization ({mat}): {x[~il]}"
                )

        us, ds = None, None
        if self.cn_file is not None:
            cn = pd.read_feather(self.cn_file).rename(
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
                us = flywire.get_connectivity(
                    x,
                    upstream=True,
                    downstream=False,
                    proofread_only=True,
                    filtered=True,
                    min_score=50,
                    progress=False,
                    materialization=mat,
                )
            if self.downstream:
                ds = flywire.get_connectivity(
                    x,
                    upstream=False,
                    downstream=True,
                    proofread_only=True,
                    filtered=True,
                    min_score=50,
                    progress=False,
                    materialization=mat,
                )

        if self.exclude_queries:
            if self.upstream:
                us = us[~us.pre.isin(x)]
            if self.downstream:
                ds = ds[~ds.post.isin(x)]

        # For grouping by type simply replace pre and post IDs with their types
        # -> we'll aggregate later
        if self.use_types:
            fw_types = _get_fw_types(mat, add_side=False, live=self.live_annot)
            fw_sides = _get_fw_sides(mat, live=self.live_annot)
            if self.upstream:
                us = _add_types(
                    us,
                    types=fw_types,
                    col="pre",
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else fw_sides,
                    sides_rel=True if self.use_sides == "relative" else False,
                )

            if self.downstream:
                ds = _add_types(
                    ds,
                    types=fw_types,
                    col="post",
                    expand_morphology_types=True,
                    sides=None if not self.use_sides else fw_sides,
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

        # Translate morphology types into connectivity types
        # This makes it easier to align with hemibrain
        # self.connectivity_.columns = _morphology_to_connectivity_types(
        #    self.connectivity_.columns
        # )

        return self
