import numpy as np
import pandas as pd

from abc import ABC

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree

from ..utils import printv


class GroundTruthModel(ABC):
    pass


class PotentialConnectivityModel(GroundTruthModel):
    """Model using potential connectivity for extrapolating connectivity.

    In a gist, what we're doing here is this:
      1. Take the full synapse table and split into proofread and unproofread
         connections for pre- and postsynapses, respectively.
      2. For each unassigned presynapses, assign them to the closest proofread
         neuron as measured by the closest proofread presynapse.
      3. For each unassigned postsynapse, connect it to the closest proofread
         postsynapse as measured by the closest proofread postsynapse.

    Parameters
    ----------
    synapse_table : pd.DataFrame
        DataFrame containing synapse table.
    proofread_ids : list | np.ndarray
        List of proofread IDs.
    max_dist : float
        Maximum distance to consider for assigning synapses. If None, no
        maximum distance is enforced.
    allow_autapses : bool
        Whether to allow autapses. If False, autapses are disallowed and simply
        discarded.

    """

    __expected_cols = {
        "pre": ("pre", "pre_pt_root_id"),
        "post": ("post", "post_pt_root_id"),
        "pre_x": ("pre_x", "pre_pt_position_x"),
        "pre_y": ("pre_y", "pre_pt_position_y"),
        "pre_z": ("pre_z", "pre_pt_position_z"),
        "post_x": ("post_x", "post_pt_position_x"),
        "post_y": ("post_y", "post_pt_position_y"),
        "post_z": ("post_z", "post_pt_position_z"),
    }

    def __init__(
        self, synapse_table, proofread_ids, max_dist=None, allow_autapses=False
    ):
        assert isinstance(synapse_table, pd.DataFrame)
        assert isinstance(proofread_ids, (list, np.ndarray))

        self.synapse_table_ = synapse_table
        self.proofread_ids_ = np.asarray(proofread_ids, dtype=np.int64)
        self.max_dist = max_dist
        self.allow_autapses = allow_autapses

        # Check if synapse table has the expected columns
        self._cols = {}
        for k, v in self.__expected_cols.items():
            for c in v:
                if c in synapse_table.columns:
                    self._cols[k] = c
                    break

            if k not in self._cols:
                raise ValueError(f"Missing column: {k} ({v})")

    def compile(self, verbose=True):
        """Compile ground truth model."""
        # ToDo:
        # - assign synapses probabilistically instead of deterministaclly based on distance
        # - assign synapses only within the same ROI (if available)?\
        # - disallow autapses but try again to re-assign them
        #   -> probably run a first pass, and then fix the autapses in a second pass using a mask

        # Add new columns to synapse table
        self.synapse_table_["pre_mod"] = np.int64(-1)
        self.synapse_table_["post_mod"] = np.int64(-1)

        # Split synapse table into proofread and unproofread pre- and postsynapses
        pre_proofread = (
            self.synapse_table_[self._cols["pre"]].isin(self.proofread_ids_).values
        )
        post_proofread = (
            self.synapse_table_[self._cols["post"]].isin(self.proofread_ids_).values
        )

        # Fill in already proofread synapses
        self.synapse_table_.loc[pre_proofread, "pre_mod"] = self.synapse_table_.loc[
            pre_proofread, self._cols["pre"]
        ]
        self.synapse_table_.loc[post_proofread, "post_mod"] = self.synapse_table_.loc[
            post_proofread, self._cols["post"]
        ]

        # Assign unproofread presynapses to the closest proofread neuron
        if (~pre_proofread).any():
            printv(
                f"Assigning {(~pre_proofread).sum():,} unproofread presynapses... ",
                verbose=verbose,
                end="",
                flush=True,
            )
            xyz_cols = [self._cols[c] for c in ["pre_x", "pre_y", "pre_z"]]
            tree = KDTree(self.synapse_table_.loc[pre_proofread, xyz_cols].values)
            d, idx = tree.query(
                self.synapse_table_.loc[~pre_proofread, xyz_cols].values
            )
            ids = self.synapse_table_.loc[pre_proofread, self._cols["pre"]].values[idx]

            if self.max_dist is not None:
                ids[d > self.max_dist] = -1

            if not self.allow_autapses:
                ids[(ids == self.synapse_table_.loc[~pre_proofread, self._cols["post"]]).values] = -1

            self.synapse_table_.loc[~pre_proofread, "pre_mod"] = ids
            printv("Done.", verbose=verbose, flush=True)

        # Assign unproofread postsynapses to the closest proofread neuron
        if (~post_proofread).any():
            printv(
                f"Assigning {(~post_proofread).sum():,} unproofread postsynapses... ",
                verbose=verbose,
                end="",
                flush=True,
            )
            xyz_cols = [self._cols[c] for c in ["post_x", "post_y", "post_z"]]
            tree = KDTree(self.synapse_table_.loc[post_proofread, xyz_cols].values)
            d, idx = tree.query(
                self.synapse_table_.loc[~post_proofread, xyz_cols].values
            )
            ids = self.synapse_table_.loc[post_proofread, self._cols["post"]].values[
                idx
            ]

            if self.max_dist is not None:
                ids[d > self.max_dist] = -1

            if not self.allow_autapses:
                ids[ids == self.synapse_table_.loc[~post_proofread, self._cols["pre"]]] = -1

            self.synapse_table_.loc[~post_proofread, "post_mod"] = ids
            printv("Done.", verbose=verbose, flush=True)

        unassigned = (
            (self.synapse_table_["pre_mod"] == -1)
            | (self.synapse_table_["post_mod"] == -1)
        ).sum()
        if unassigned:
            printv(
                f"{unassigned:,} synaptic connections ({unassigned/self.synapse_table_.shape[0]:.2%}) were left unassigned. "
                "These will show up with pre-/postsynaptic IDs of -1.",
                verbose=verbose,
            )

        # Agglomerate
        if "roi" not in self.synapse_table_.columns:
            self.edges_ = self.synapse_table_.groupby(
                ["pre_mod", "post_mod"], as_index=False
            ).size()
        else:
            self.edges_ = self.synapse_table_.groupby(
                ["pre_mod", "post_mod", "roi"], as_index=False
            ).size()

        self.edges_.rename(
            {"pre_mod": "pre", "post_mod": "post", "size": "syn_count"},
            axis=1,
            inplace=True,
        )

        printv(
            "Compiled ground truth model. Data is accessible via `synapse_table_` and `edges_`.",
            verbose=verbose,
        )

        return self
