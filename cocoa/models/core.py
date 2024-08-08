import functools

import pandas as pd
import numpy as np
from tqdm.auto import trange

from ..datasets.core import DataSet
from ..datasets.ds_utils import _add_types
from ..utils import printv
from ..mappers import GraphMapper, BaseMapper


__all__ = ["GroundTruthModel"]


"""
TODOs:

Refactor this module such into various different classes:
 - `StochasticConnectomeModel` class that handles the execution, results and analysis
 - (potentially multiple versions of) `GroundTruthModel` that handles the upsampling
 - `Process` classes that handle the application of the individual stochastic processes; e.g.:
   - `SynapsePrediction`
   - `ReconstructionLoss`
The StochasticConnectomeModel takes a single dataset, a GroundTruthModel and multiple Processes, and
runs them in sequence.

"""


def req_compile(func):
    """Check if we need to compile connectivity."""

    @functools.wraps(func)
    def inner(*args, **kwargs):
        if not hasattr(args[0], "edges_"):
            args[0].compile()
        return func(*args, **kwargs)

    return inner


class GroundTruthModel:
    """Class to model stochastic processes during connectome generation.

    Parameters
    ----------
    dataset :   DataSet
                The dataset to upsample. Must implement a
                `.roi_completion()` method.
    use_types : bool, optional
                Whether or not to run the analyses at the level
                of neuron types.
    link_pred_method : str, optional
                TODO! Method used to predict new links. If None
                will not predict new links and distribute
                missing synapses evenly across all neurons.

    """

    def __init__(
        self, dataset, roi_completeness=None, use_types=True, link_pred_method=None
    ):
        assert isinstance(dataset, DataSet)
        self._dataset = dataset
        self.use_types = use_types

        if link_pred_method is not None:
            raise NotImplementedError("Link prediction not yet implemented")

        if roi_completeness is None:
            if not hasattr(dataset, "get_roi_completeness"):
                raise ValueError("Dataset must implement `get_roi_completeness` method")
            self.roi_completeness_ = dataset.get_roi_completeness()
        else:
            self.roi_completeness_ = roi_completeness

    def __repr__(self):
        return f"GroundTruthModel <dataset={self.dataset}>"

    @property
    def dataset(self):
        """Dataset in this model."""
        return self._dataset

    @property
    def roi_completeness_(self):
        return self._roi_completeness

    @roi_completeness_.setter
    def roi_completeness_(self, x):
        assert isinstance(x, pd.DataFrame), "ROI completeness must be a DataFrame"
        for col in ("roi", "roipre", "roipost", "totalpre", "totalpost"):
            assert col in x.columns, f"ROI completeness must have a '{col}' column"
        self._roi_completeness = x

    def copy(self):
        """Return copy of model."""
        cl = GroundTruthModel()
        cl._datasets = list(self.datasets)
        for prop in ("edges_",):
            if hasattr(self, prop):
                setattr(cl, prop, getattr(self, prop).copy())
        return cl

    def compile(
        self,
        mapper=GraphMapper,
        force_recompile=False,
        verbose=True,
    ):
        """Compute the ground truth model.

        Parameters
        ----------
        mapper :    cocoa.Mapper
                    The mapper used to assign labels to neurons.
        force_recompile : bool
                    If True, will recompile connectivity vectors for each data
                    set even if they already exist.

        Returns
        -------
        self

        """

        # Compile edges for the dataset
        if (
            not hasattr(self.dataset, "adj_")
            or force_recompile
            or getattr(self.dataset, "adj_types_used_", False)
        ):
            printv(
                f'Compiling adjacencies for "{self.dataset.label}" '
                f"({self.dataset.type})",
                verbose=verbose,
            )
            _ot = self.dataset.use_types
            self.dataset.use_types = False
            self.dataset.compile_adjacency(collapse_rois=False)
            self.dataset.use_types = _ot

        # Make sure the ROI column is named "roi"
        self.edges_ = self.dataset.adj_.rename(
            columns={self.dataset._roi_col: "roi"}
        ).copy()

        # Collapse cell types if that's what we're doing
        if self.use_types:
            # Generate the mappings
            if isinstance(mapper, type):
                if not issubclass(mapper, BaseMapper):
                    raise TypeError(f'`mapper` must be a Mapper, got "{mapper}"')
                mapper = mapper()
            self.mapper_ = mapper.add_dataset(self.dataset).compile()
            self.mappings_ = self.mapper_.mappings_

            self.edges_ = _add_types(
                self.edges_,
                types=self.mappings_,
                col=("pre", "post"),
                sides=None,
                sides_rel=False,
            )

            # Drop untyped connections
            self.edges_ = self.edges_[
                self.edges_.pre.isin(self.mappings_.values())
                & self.edges_.post.isin(self.mappings_.values())
            ]

            self.edges_ = self.edges_.groupby(
                ["pre", "post", "roi"], as_index=False
            ).weight.sum()

        # Now scale each edge to 100% connectivity based on the ROIs' post-completion rates
        # Note to self: think about how to incorporate the pre-completion rates
        self.roi_scale_factors_ = 1 / (
            self.roi_completeness_.set_index("roi").roipost
            / self.roi_completeness_.totalpost.values
        )
        self.edges_["weight_scaled"] = (
            (
                self.edges_["weight"]
                * self.edges_.roi.map(self.roi_scale_factors_).fillna(1)
            )
            .round()
            .astype(np.int32)
        )

        printv(
            "All done! Results are available as `.edges_` property.",
            verbose=verbose,
        )
        return self

    @req_compile
    def sample(
        self,
        roi_completion_rates,
        it=1,
        w="weight_scaled",
        out="summary",
        false_neg=None,
        false_pos=None,
        progress=True,
        verbose=True,
    ):
        """Subsample scaled edges to match target completion rates.

        Parameters
        ----------
        roi_completion_rates : DataFrame
                    The target per-neuropil completion rates.
        it :        int
                    The number of iterations.
        w :         str
                    The column in `edges_` to use as weights.
        out :       "summary" | "full"
                    Whether to return a summary (mean, std, etc.) or all
                    draws.
        false_neg : float, optional
                    The false negative rate. E.g `0.1` means that we will
                    "miss" 10% of the synapses.
        false_pos : float, optional
                    The false positive rate.
        progress :  bool
                    Whether or not to show a progress bar.

        Returns
        -------
        pd.DataFrame
            The subsampled edge. Note to self: should this return a DataSet?
            That way it can be easily plugged into the "Comparison" class.

        """
        assert out in ("summary", "full"), f"Invalid `out` parameter: {out}"

        # We have to treat all our edges as a synapse soup from which we will draw
        # samples up to the target completion rates.
        indices = np.repeat(np.arange(len(self.edges_)), repeats=self.edges_[w]).astype(
            "int32"
        )

        # Make a integer -> ID neuropil map; this way we avoid having to deal with strings
        npmap = dict(
            zip(self.edges_.roi.unique(), range(len(self.edges_.roi.unique())))
        )
        np_indices = np.repeat(
            self.edges_.roi.map(npmap).astype(int), repeats=self.edges_[w]
        ).values

        # Make sure target_completion rates include all ROIs in our dataset
        average_completion_rate = (
            roi_completion_rates.roipost.sum() / roi_completion_rates.totalpost.sum()
        )
        miss = set(self.edges_.roi.unique()) - set(roi_completion_rates.roi)
        if any(miss):
            printv(
                f"Missing ROI(s) in target completion rates: {','.join(list(miss))}",
                verbose=verbose,
            )
            printv(
                f"These ROIs will be given the average completion rate of {average_completion_rate:.4}.",
                verbose=verbose,
            )

        # Calculate the number of synapses to draw for each neuropil
        self.target_completion_rates_ = (
            roi_completion_rates.set_index("roi").roipost
            / roi_completion_rates.totalpost.values
        )
        # Fill in missing ROIs with the average completion rate
        self.target_completion_rates_ = self.target_completion_rates_.reindex(
            self.edges_.roi.unique()
        ).fillna(average_completion_rate)

        # Count the number of synapses we have per ROI (this dataset might not contain all neurons,
        # so we can't just use the total number of synapses)
        per_roi_counts = self.edges_.groupby("roi")[w].sum()

        # How many synapses do we need to draw for each ROI?
        self.target_synapse_counts_ = (
            (self.target_completion_rates_ * per_roi_counts).round().astype(int)
        )

        # Drop ROIs that aren't in the dataset (speeds up the simulation)
        self.target_synapse_counts_ = self.target_synapse_counts_[
            self.target_synapse_counts_.index.isin(npmap)
        ]

        # To simulate false negatives, we will drop synapses from the target counts
        # (meaning we will draw fewer synapses than we should have)
        if false_neg is not None:
            self.target_synapse_counts_ = (self.target_synapse_counts_ * (1 - false_neg)).astype(int)

        # Prepare the output
        new_weights = np.zeros((it, len(self.edges_)), dtype=np.int32)

        printv(
            f"Drawing {self.target_synapse_counts_.sum():,} from {np_indices.shape[0]:,} ({self.target_synapse_counts_.sum()/np_indices.shape[0]:.1%}) synapses",
            verbose=verbose,
        )

        # We know exactly how many synapses we will draw. Hence we can preallocate
        # the array and fill it in one go instead of extending e.g. a list
        this_draw = np.empty(self.target_synapse_counts_.sum(), dtype="int32")
        for i in trange(it, desc="Sim.", disable=(not progress) or (it == 1)):
            ix = 0  # index into `this_draw``
            for roi, target in self.target_synapse_counts_.items():
                # Draw synapses from the pool
                this_draw[ix : ix + target] = np.random.choice(
                    indices[np_indices == npmap[roi]], size=target, replace=False
                )
                ix += target

            # Count the synapses
            this_draw_ind, this_draw_counts = np.unique(this_draw, return_counts=True)
            new_weights[i, this_draw_ind] = this_draw_counts

            # Add false positives
            if false_pos:
                N_add = int(new_weights[i, :].sum() * false_pos)
                # We're assuming that strong edges have a higher chance of adding
                # false positives (neurons are likely to be closer together)
                to_add = np.random.choice(indices, N_add, replace=True)
                ix, cnt = np.unique(to_add, return_counts=True)
                new_weights[i, ix] += cnt

        # Collate results
        res = self.edges_[["pre", "post", "roi", "weight", w]].copy()
        if out == "summary":
            res["mean"] = new_weights.mean(axis=0)
            res["std"] = new_weights.std(axis=0)
            res["min"] = new_weights.min(axis=0)
            res["max"] = new_weights.max(axis=0)
            res["median"] = np.median(new_weights, axis=0)
            res["q25"] = np.quantile(new_weights, 0.25, axis=0)
            res["q75"] = np.quantile(new_weights, 0.75, axis=0)
        else:
            new_cols = []
            for i in range(it):
                new_cols.append(pd.Series(new_weights[i], name=f"draw_{i}", index=res.index))
            res = pd.concat([res] + new_cols, axis=1)

        return res

    @req_compile
    def plot_all(self):
        """Run all plots."""
        for f in dir(self):
            if f.startswith("plot_") and callable(getattr(self, f)) and f != "plot_all":
                print(f"Running {f}")
                getattr(self, f)()


def swap_roi_sides(x):
    """Swap the sides in ROI completion data."""
    # This is a utility function
    assert isinstance(x, pd.DataFrame), "Input must be a DataFrame"
    assert "roi" in x.columns, "Input must have a 'roi' column"

    x = x.copy()

    # Swap "(R)" for "(L)" and vice versa
    is_left = x.roi.str.contains(r"\(L\)", na=False)
    is_right = x.roi.str.contains(r"\(R\)", na=False)
    x.loc[is_left, "roi"] = x.loc[is_left, "roi"].str.replace(
        r"\(L\)", "(R)", regex=True
    )
    x.loc[is_right, "roi"] = x.loc[is_right, "roi"].str.replace(
        r"\(R\)", "(L)", regex=True
    )

    return x
