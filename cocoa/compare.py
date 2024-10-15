import re
import functools

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcl
import matplotlib.pyplot as plt

from itertools import combinations

from .datasets.core import DataSet
from .datasets.ds_utils import _add_types
from .utils import make_iterable, printv
from .mappers import GraphMapper, BaseMapper


__all__ = ["Comparison"]

# TODOs:
# - connectivity comparison per ROI
# - correlation by label (type, super class, etc)


def req_compile(func):
    """Check if we need to compile connectivity."""

    @functools.wraps(func)
    def inner(*args, **kwargs):
        if not hasattr(args[0], "adj_"):
            args[0].compile()
        return func(*args, **kwargs)

    return inner


class Comparison:
    """Class to compare edge weights across matched labels in two or more datasets.

    Parameters
    ----------
    datasets :  DataSet | list of DataSet, optional
                One or more datasets to include in the clustering.
                Alternatively, datasets can be added using the `add_dataset`.

    """

    def __init__(self, *datasets):
        self._datasets = []
        self.add_dataset(datasets)

    def __repr__(self):
        n_neurons = sum(len(ds.neurons) for ds in self.datasets)
        return f"Comparison <datasets={len(self.datasets)};neurons={n_neurons}>"

    def __len__(self):
        """The number of datasets in the Clustering."""
        return len(self._datasets)

    @property
    def datasets(self):
        """Datasets in this Clustering."""
        return self._datasets

    def copy(self):
        """Return copy of clustering."""
        cl = Comparison()
        cl._datasets = list(self.datasets)
        for prop in (
            "adj_",
        ):
            if hasattr(self, prop):
                setattr(cl, prop, getattr(self, prop).copy())
        return cl

    def add_dataset(self, ds):
        """Add dataset(s)."""
        if isinstance(ds, (list, tuple)):
            for d in ds:
                self.add_dataset(d)
        else:
            if not isinstance(ds, DataSet):
                raise TypeError(f'Expected dataset(s), got "{type(ds)}"')
            if len(ds.neurons) == 0:
                raise ValueError(f"Dataset {ds.label} ({ds.type}) has no neurons.")
            if ds.label in [d.label for d in self.datasets]:
                raise ValueError(
                    f"Comparison already contains a dataset with label '{ds.label}'."
                )
            self._datasets.append(ds)

        return self

    def compile(
        self,
        mapper=GraphMapper,
        force_recompile=False,
        verbose=True,
    ):
        """Compile and compare connectivity vectors.

        Parameters
        ----------
        mapper :    cocoa.Mapper
                    The mapper used to match neuron labels across datasets.
                    Examples are `cocoa.GraphMapper` and `cocoa.SimpleMapper`.
                    See the mapper's documentation for more information.
        force_recompile : bool
                    If True, will recompile connectivity vectors for each data
                    set even if they already exist.

        Returns
        -------
        self

        """
        if len(self) <= 1:
            raise ValueError("Comparison requires >=2 datasets")

        mapper_type = type(mapper) if not isinstance(mapper, type) else mapper
        if not issubclass(mapper_type, BaseMapper):
            raise TypeError(f'`mapper` must be a Mapper, got "{mapper_type}"')

        # First compile datasets if necessary
        for i, ds in enumerate(self.datasets):
            # Recompile if (a) not yet compiled, (b) forced recompile or (c) the current edge list contains types
            if (
                not hasattr(ds, "adj_")
                or force_recompile
                or getattr(ds, "adj_types_used_", False)
            ):
                printv(
                    f'Compiling adjacencies for "{ds.label}" '
                    f"({ds.type}) [{i+1}/{len(self.datasets)}]",
                    verbose=verbose,
                )
                _ot = ds.use_types
                ds.use_types = False
                ds.compile_adjacency()
                ds.use_types = _ot

        # Generate the mappings
        if isinstance(mapper, type):
            mapper = mapper(verbose=verbose)
        self.mapper_ = mapper.add_dataset(*self.datasets)
        self.mappings_ = self.mapper_.get_mappings()

        # At this point we should remove labels from the mapping which do not exist in all datasets.
        # Keep in mind that the Mapper will have produced labels for ALL neurons in the datasets, not
        # just the pool of neurons we're looking at here.
        # Basically need to avoid that e.g. "AVLP1234->AVLP2345" shows up as 50 synapses for one dataset
        # but as 0 for another dataset simply because either of the two types is not present in
        # the `dataset.neurons` (i.e. we didn't select the exact same pool of neurons in all datasets).

        # 1. Collect all labels per dataset
        labels_per_dataset = [
            {self.mappings_.get(i, i) for i in ds.neurons} for ds in self.datasets
        ]
        # 2. Intersect
        labels_keep = set.intersection(*labels_per_dataset)
        # 3. Filter mappings
        self.mappings_filtered_ = {
            k: v for k, v in self.mappings_.items() if v in labels_keep
        }

        printv(
            f"{len(self.mappings_filtered_):,} of all {len(self.mappings_):,} ({len(self.mappings_filtered_)/len(self.mappings_):.2%}) cross-matched labels present in all datasets.",
            verbose=verbose,
            flush=True,
        )

        printv("Combining adjacencies... ", verbose=verbose, flush=True)

        # Apply the mappings to the individual datasets
        adjacencies = []
        for ds in self.datasets:
            # Add types based on the mappings
            adj = _add_types(
                ds.adj_.copy(),
                types=self.mappings_filtered_,
                col=("pre", "post"),
                sides=None,
                sides_rel=False,
            )

            # Drop untyped connections
            total_weights = adj.weight.sum()
            total_cn = adj.shape[0]
            adj = adj[
                adj.pre.isin(self.mappings_filtered_.values())
                & adj.post.isin(self.mappings_filtered_.values())
            ]
            printv(
                f"{ds.label}: {len(adj):,} of {total_cn:,} connections containing {adj.weight.sum()/total_weights:.2%} of synapses cross-matched.",
                verbose=verbose,
            )

            # TODOs:
            # -[] normalised edge weights

            # Group by pre and post
            adj = adj.groupby(["pre", "post"]).weight.sum()

            # Rename the weight column to `{label}`
            adj.name = f"{ds.label}"

            adjacencies.append(adj)

        # Join the adjancencies on their index
        self.adj_ = pd.concat(adjacencies, axis=1)

        printv(
            "All done! Results are available as `.adj_` and `.adj_norm` properties.",
            verbose=verbose,
        )
        return self

    @req_compile
    def plot_all(self):
        """Run all plots."""
        for f in dir(self):
            if f.startswith("plot_") and callable(getattr(self, f)) and f != "plot_all":
                print(f"Running {f}")
                getattr(self, f)()

    @req_compile
    def plot_correlation_matrix(
        self, ax=None, square=True, annot=True, cmap="coolwarm", triangle='lower'
    ):
        """Plot correlation matrix of edge weights."""
        # TODO:
        # - add other metrics such as cosine similarity
        if triangle == 'lower':
            mask = np.triu(np.ones_like(self.adj_.corr(), dtype=bool), k=1)
        elif triangle == 'upper':
            mask = np.tril(np.ones_like(self.adj_.corr(), dtype=bool), k=-1)
        else:
            mask=None

        ax = sns.heatmap(
            # N.B. that we need to fill NaNs with 0 here, otherwise those will be
            # ignored by the correlation calculation
            self.adj_.fillna(0).corr(),
            vmin=-1,
            center=0,
            vmax=1,
            cmap=cmap,
            annot=annot,
            cbar=False,
            mask=mask,
            ax=ax,
            square=square,
        )

        return ax

    @req_compile
    def plot_frac_found(self):
        """Plot fraction of edges present in 1-N datasets."""
        # Count the number of datasets in which each edge is present
        counts = self.adj_.notnull().sum(axis=1).value_counts().to_frame()
        ax = counts.plot.barh(color="k")

        ax.set_ylabel("found in N datasets")
        ax.set_xlabel("# of edges")
        ax.get_legend().remove()

        # Add percentages
        counts_perc = counts / counts.sum()
        offset = counts.max() * 0.025
        for i in range(len(counts)):
            ax.text(
                counts.iloc[i, 0] - offset,
                i,
                f"{counts_perc.iloc[i, 0]:.2%}",
                va="center",
                ha="right",
                color="w",
            )

        return ax

    @req_compile
    def plot_frac_found_pairwise(self):
        """Plot the fraction of edges found in each pairwise comparison."""
        # Two rows figure: top = barplot, bottom = scatterplot
        fig, axes = plt.subplots(
            2, 1, figsize=(5, 5), sharex=True, height_ratios=(3, 1)
        )

        # Generate the counts
        counts = self.adj_.notnull().value_counts().to_frame()
        indices = counts.reset_index(drop=False).drop("count", axis=1)
        # Sort such that the number of datasets goes from high to low (left to right)
        srt = np.argsort(indices.sum(axis=1).values)[::-1]
        indices = indices.iloc[srt]
        counts = counts.iloc[srt]

        # Plot the bar plot
        ax = counts.plot.bar(color="k", width=0.95, ax=axes[0])

        # Add fractions
        offset = counts.values.max() * 0.001
        for i, (idx, row) in enumerate(counts.iterrows()):
            axes[0].text(
                i,
                row.values[0] - offset,
                f"{row.values[0]/counts.values.sum():.1%}",
                ha="center",
                color="w",
                rotation=90,
                va="top",
            )

        # Plot the scatter plot indicating the pairwise combinations
        x, y = np.where(indices)
        axes[1].scatter(x, y, color="k")

        for i, _x in enumerate(np.unique(x)):
            this_y = y[x == _x]
            axes[1].plot([_x, _x], [min(this_y), max(this_y)], "k", lw=0.5)

        axes[1].set_yticks(range(len(indices.columns)))
        axes[1].set_yticklabels(indices.columns)

        # Some formatting cleanup
        axes[0].get_legend().remove()
        axes[1].set_xticks([])

        for spine in axes[1].spines.values():
            spine.set_visible(False)

        # Remove minor ticks
        axes[0].xaxis.set_tick_params(which="minor", size=0)
        axes[1].xaxis.set_tick_params(which="minor", size=0)

        axes[0].set_ylabel("# of edges")

        plt.tight_layout()
        return axes

    @req_compile
    def plot_pairwise_edgeweights(self, subset=None, threshold=50, quantiles=(.05, .25, .75, .95), fillna=True, **kwargs):
        """Plot pairwise edge weights.

        The envelopes represent the 50th percentile.

        Parameters
        ----------
        subset :    list of str, optional
                    Subset of datasets to plot. Default to all.
        threshold : int, optional
                    Threshold for edge weights to plot. Typically, the lower
                    range of edge weights is more interesting since the data
                    tends to get very sparse at higher weights (which causes scraggly
                    lines). Default to 50.
        quantiles : iterable, optional
                    Which quantiles [0-100] to plot as envelopes.
        fillna :    bool, optional
                    Fill NaN values (i.e. edges missing in one of the datasets) with 0. If False,
                    edges that are not present in both datasets of a given pairwise comparison are
                    ignored. Default to True.
        **kwargs
                    Keyword arguments are passed through to seaborn.Lineplot

        """
        fig, ax = plt.subplots(figsize=(5, 5))

        if subset is None:
            subset = self.adj_.columns.values

        if not isinstance(subset, (list, np.ndarray)):
            raise TypeError(f"Expected `subset` to be list or array, got {type(subset)}")

        if len(subset) < 2:
            raise ValueError("Need at least two datasets in `subset` to compare.")

        if isinstance(quantiles, (int, float)):
            quantiles = [quantiles]

        # Plot pairwise comparisons
        for c1, c2 in combinations(subset, 2):
            this = self.adj_[self.adj_[c1] <= threshold]
            if fillna:
                this = this.fillna(0)
            y = this.groupby(c1)[c2].mean()
            x = y.index
            ax.plot(x, y, label=f"{c1} vs {c2}", **kwargs)
            # Add quantile envelopes
            if quantiles is not None:
                for q in quantiles:
                    assert q >= 0 and q <= 1, f"Quantile {q} is not in [0, 1]"
                    y2 = this.groupby(c1)[c2].quantile(q)
                    ax.fill_between(x, y2, y, alpha=0.2, color=ax.get_lines()[-1].get_color())

        # Add x=y line
        ax.plot(
            ax.get_xlim(), ax.get_xlim(), color="lightgrey", linestyle="--", zorder=-10
        )

        # Make sure x and y axis are the same
        ax.set_yticks(ax.get_xticks())
        ax.set_ylim(ax.get_xlim())

        ax.legend()
        ax.get_legend().set_title("x- vs y-axis")

        ax.set_xlabel("edge weight")
        ax.set_ylabel("edge weight")

        return ax

    @req_compile
    def plot_edge_weight_histogram(self):
        """Plot edge weight distributions.

        Importantly, this will plot the distribution of edge weights
        before and after normalisation, and before and after collapsing
        by cell type.

        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].set_title("neuron-to-neuron (all)")
        axes[1].set_title("type-to-type (cross-matched only)")

        for i, ds in enumerate(self.datasets):
            # Plot the before edge
            sns.histplot(
                data=ds.adj_,
                x="weight",
                log_scale=True,
                element="step",
                fill=False,
                # stat="density",
                cumulative=True,
                ax=axes[0],
                label=ds.label,
            )

            sns.histplot(
                data=self.adj_,
                x=ds.label,
                log_scale=True,
                element="step",
                fill=False,
                # stat="density",
                cumulative=True,
                ax=axes[1],
                label=ds.label,
            )

        for ax in axes:
            ax.set_xlabel("edge weight")

        axes[-1].legend()

    @req_compile
    def plot_edge_prob(self, threshold=50):
        """Plot probability to find edge in N datasets as function of its weight.

        Parameters
        ----------
        threshold : int, optional
                    Threshold for edge weights to plot. Typically, the lower
                    range of edge weights is more interesting since the data
                    tends to get very sparse at higher weights (which causes scraggly
                    lines). Default to 50.

        """
        # Collect the data
        x = []
        ys = [[] for _ in range(self.adj_.shape[1])]
        for i in range(1, threshold + 1):
            # Get edge that occur in {i} datasets
            # N.B: here, we're asking "for a given edge, is ANY of the weights equal to {i}?"
            this = (self.adj_ == i).any(axis=1)
            # Count the number of datasets the weight={i} edges are present in
            n_ds = self.adj_[this].notnull().sum(axis=1).values

            # Tally up the number of times we have edges in 1-N datasets
            x.append(i)
            for y, count in zip(*np.unique(n_ds, return_counts=True)):
                ys[y - 1].append(count)
        x = np.array(x)
        ys = np.array(ys)

        # Normalise such that each bin (i.e. each weight from 1 to threshold) sums to 1
        ys_norm = ys / ys.sum(axis=0)

        # Plot!
        fig, ax = plt.subplots(figsize=(4, 4))
        for i in range(ys.shape[0]):
            ax.plot(x, ys_norm[i], label=f"{i + 1} datasets")

        ax.legend()
        ax.set_xlabel('edge weight')
        ax.set_ylabel('probability to find edge in N datasets')

        return ax
