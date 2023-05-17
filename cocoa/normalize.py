import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcl
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

from .datasets.core import DataSet

"""
Code Example:

# Step 1: Define datasets
>>> import cocoa as co
>>> ds1 = co.datasets.FlyWireRHS(ids=[1213, 12321])
>>> ds2 = co.datasets.FlyWireLHS(ids=[1213, 12321])

# Step 2: Plot
>>> co.cosine_plot(ds1, ds2, ...)

# Alternatively if there is a preset
>>> co.cosine_plot({'FAFB_RHS': [1213, 12321],
...                 'FAFB_LHS': [1211, 22321],
...                 'hemibrain_RHS': [3212, 32313]})

>>> # `edges_...` is a dataframe from some other source
>>> ds1, ds2 = co.DataSet('RHS'), co.DataSet('LHS')
>>> ds1.add_edges(edges_left)
>>> ds2.add_edges(edges_right)
>>> # Add (optional) common labels
>>> ds1.add_labels(labels_dict1)
>>> ds2.add_labels(labels_dict2)
>>> # Combine
>>> cb = co.Normalizer(ds1, ds2)
>>> # Alternatively: cb = ds1 + ds2
>>> # Get some stats
>>> cb.report()
Normalizer containing 2 datasets: "RHS", "LHS"

"RHS" contains 4231 edges for 142 neurons.
"LHS" contains 4532 edges for 140 neurons.

"RHS" and "LHS" have 50 shared labels.

Other methods:

Normalizer.label_summary()  # for each shared label get a summary
Normalizer.coverage()  # report a coverage for each neuron
Normalizer.coverage_plot()
Normalizer.connectivity_vector(use_labels=True)
Normalizer.connectivity_distance()
Normalizer.clustermap(interactive=True)
Normalizer.dendrogram()

For precomputed co-clustering (e.g. from NBLAST scores) use `co.Cluster`

For graph matching use `co.GraphMatcher([])`

"""

__all__ = ["Normalizer"]


class Normalizer:
    """Normalizes and combine datasets."""

    def __init__(self, *datasets):
        self._datasets = []
        self.add_dataset(datasets)

    def __repr__(self):
        return f"Normalizer <datasets={len(self.datasets)}>"

    def __len__(self):
        return len(self._datasets)

    @property
    def datasets(self):
        return self._datasets

    def add_dataset(self, ds):
        """Add dataset(s)."""
        if isinstance(ds, (list, tuple)):
            for d in ds:
                self.add_dataset(d)
        else:
            if not isinstance(ds, DataSet):
                raise TypeError(f'Expected dataset(s), got "{type(ds)}"')
            self._datasets.append(ds)

    def compile(self, metric="cosine", force_recompile=False):
        """Compile a combined connectivity vector."""
        if len(self) <= 1:
            raise ValueError("Normalizer requires >=2 datasets")

        # First compile datasets if necessary
        for i, ds in enumerate(self.datasets):
            if not hasattr(ds, "edges_") or force_recompile:
                print(
                    f'Compiling connectivity for "{ds.label}"'
                    f"({ds.type}) [{i+1}/{len(self.datasets)}]"
                )
                ds.compile()

        # Extract labels
        all_labels = set()
        for ds in self.datasets:
            up = ds.edges_.loc[ds.edges_.post.isin(ds.neurons)]
            down = ds.edges_.loc[ds.edges_.pre.isin(ds.neurons)]
            all_labels |= set(up.pre.values.astype(str))
            all_labels |= set(down.post.values.astype(str))

        # Find any cell types we need to collapse
        collapse_types = {
            c: cc for cc in list(all_labels) for c in cc.split(",") if ("," in cc)
        }
        for ds in self.datasets:
            edges = ds.edges_.copy()
            if len(collapse_types):
                is_up = edges.post.isin(ds.neurons)
                is_down = edges.pre.isin(ds.neurons)
                edges.loc[is_up, "pre"] = edges.loc[is_up, "pre"].map(
                    lambda x: collapse_types.get(x, x)
                )
                edges.loc[is_down, "post"] = edges.loc[is_down, "post"].map(
                    lambda x: collapse_types.get(x, x)
                )

                ds.edges_proc_ = edges.groupby(
                    ["pre", "post"], as_index=False
                ).weight.sum()
            else:
                ds.edges_proc_ = edges

        # Find labels that exist in all datasets
        in_all = set(self.datasets[0].edges_.pre.unique().tolist()) | set(
            self.datasets[0].edges_.post.unique().tolist()
        )
        for ds in self.datasets[1:]:
            in_all = in_all & (
                set(ds.edges_.pre.unique().tolist())
                | set(ds.edges_.post.unique().tolist())
            )
        in_all = list(in_all)

        # Subset edge lists to these labels
        for ds in self.datasets:
            edges = ds.edges_proc_
            is_up = edges.post.isin(ds.neurons)
            up_shared = edges.pre.isin(in_all)

            is_down = edges.pre.isin(ds.neurons)
            down_shared = edges.post.isin(in_all)

            ds.edges_proc_ = ds.edges_proc_.loc[
                (is_up & up_shared) | (is_down & down_shared)
            ]

        # Generate observation vectors
        adjacencies = []
        sources = []
        labels = []
        for ds in self.datasets:
            # Group the edge list
            adj = ds.edges_proc_.groupby(["pre", "post"]).weight.sum().unstack()
            # Get downstream adjacency (rows = queries, columns = shared targets)
            down = adj.reindex(index=ds.neurons, columns=in_all)
            # Get upstream adjacency (rows = shared inputs, columns = queries)
            up = adj.reindex(columns=ds.neurons, index=in_all)
            adjacencies.append(pd.concat((down, up.T), axis=1).fillna(0))
            sources += [ds.label] * adjacencies[-1].shape[0]
            labels += ds.get_labels(ds.neurons).tolist()
        self.vect_ = pd.concat(adjacencies, axis=0).astype(np.int32)
        self.vect_sources_ = np.array(sources)
        self.vect_labels = np.array(labels)

        # Calculate fraction of connectivity used for the observation vector
        syn_counts_before = {}
        for ds in self.datasets:
            syn_counts_before.update(ds.syn_counts)

        syn_counts_after = self.vect_.sum(axis=1)
        self.cn_frac_ = syn_counts_after / syn_counts_after.index.map(syn_counts_before)

        print(f"Using on average {self.cn_frac_.mean():.1%} of neurons' synapses.")
        print(f"Worst case is keeping {self.cn_frac_.min():.1%} of its synapses.")

        if metric == "Euclidean":
            dists = squareform(pdist(self.vect_))
        elif metric == "cosine":
            # Turn cosine similarity into distance and round to make sure we
            # have zeros in the diagonal
            dists = (1 - cosine_similarity(self.vect_)).round(6)

        # Change columns to "type, ds" (index remains just "id")
        self.dists_ = pd.DataFrame(
            dists,
            index=self.vect_.index,
            columns=[f"{l}_{ds}" for l, ds in zip(labels, sources)],
        )

        print("Done")

    def extract_clusters(self, N, **kwargs):
        """Extract clusters from clustermap or distance matrix.

        Parameters
        ----------
        N :     int
                Number of clusters to make.
        **kwargs
                Keyword arguments passed to `linkage()` if input is a distance
                matrix.

        Returns
        -------
        list of arrays

        """
        if not hasattr(self, 'dists_'):
            self.compile()

        x = self.dists_.copy()
        if x.values[0, 0] >= 0.999:
            x = 1 - x
        defaults = dict(method='ward')
        defaults.update(kwargs)
        Z = linkage(squareform(x.values), **defaults)
        labels = x.index.values

        cl = cut_tree(Z, n_clusters=N).flatten()

        return [labels[cl == i] for i in np.unique(cl)]

    def plot_clustermap(self):
        """Plot connectivity distance as cluster heatmap.

        Returns
        -------
        cm :        sns.clustermap
                    The distances are available via ``cm.data``

        """
        if not hasattr(self, 'dists_'):
            self.compile()

        dists = self.dists_
        Z = linkage(squareform(dists.values), method='ward')

        row_colors = [_percent_to_color(v) for v in self.cn_frac_.values]

        cm = sns.clustermap(dists, cbar_pos=None, cmap='Greys_r',
                            row_colors=row_colors,
                            row_linkage=Z, col_linkage=Z)
        ax = cm.ax_heatmap
        ix = cm.dendrogram_row.reordered_ind
        ax.set_xticks(np.arange(len(dists)) + .5)
        ax.set_yticks(np.arange(len(dists)) + .5)
        ax.set_xticklabels(dists.columns[ix], fontsize=4)
        ax.set_yticklabels(dists.index[ix], fontsize=4)

        # Bring back the scale for one of the dendrograms
        ax_d = cm.ax_col_dendrogram
        ax_d.set_axis_on()
        ax_d.spines['left'].set_visible(True)
        ylim = ax_d.get_ylim()
        if ylim[1] <= .1:
            interval = .05
            rnd = 2
        elif ylim[1] <= .3:
            interval = .1
            rnd = 1
        elif ylim[1] <= 2:
            interval = .2
            rnd = 1
        elif ylim[1] <= 5:
            interval = 1
            rnd = 0
        else:
            interval = 5
            rnd = 0
        ticks = np.arange(0, round(ylim[1], 1) + interval, interval)
        ax_d.set_yticks(ticks)
        ax_d.set_yticklabels(ticks.round(rnd), size=5)
        ax_d.set_ylim(ylim)

        plt.tight_layout()

        return cm


def _percent_to_color(x):
    """Take fraction and turn into color category."""
    if x < .1:
        c = 'red'
    elif x < .3:
        c = 'orange'
    elif x < .5:
        c = 'yellow'
    else:
        c = 'g'
    return mcl.to_rgb(c)