import re
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcl
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

# Fastcluster seems to be ~2X faster than scipy
from fastcluster import linkage
from scipy.cluster.hierarchy import cut_tree, leaves_list
from scipy.spatial.distance import squareform

from .datasets import FlyWire, Hemibrain, MaleCNS
from .datasets.core import DataSet
from .cluster_utils import extract_homogeneous_clusters, is_good
from .utils import make_iterable, req_compile, printv
from .distance import calculate_distance

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
>>> cb = co.Clustering(ds1, ds2)
>>> # Alternatively: cb = ds1 + ds2
>>> # Get some stats
>>> cb.report()
Clustering containing 2 datasets: "RHS", "LHS"

"RHS" contains 4231 edges for 142 neurons.
"LHS" contains 4532 edges for 140 neurons.

"RHS" and "LHS" have 50 shared labels.

Other methods:

Clustering.label_summary()  # for each shared label get a summary
Clustering.coverage()  # report a coverage for each neuron
Clustering.coverage_plot()
Clustering.connectivity_vector(use_labels=True)
Clustering.connectivity_distance()
Clustering.clustermap(interactive=True)
Clustering.dendrogram()

For precomputed co-clustering (e.g. from NBLAST scores) use `co.Cluster`

For graph matching use `co.GraphMatcher([])`

"""

__all__ = ["Clustering", "generate_clustering"]


CLUSTER_DEFAULTS = dict(method="ward")
DISTS_DTYPE = np.float32
VECT_DTYPE = np.uint16


class Clustering:
    """Normalizes, combines and co-clusters datasets."""

    def __init__(self, *datasets):
        self._datasets = []
        self.add_dataset(datasets)

    def __repr__(self):
        return f"Clustering <datasets={len(self.datasets)}>"

    def __len__(self):
        """The number of datasets in the Clustering."""
        return len(self._datasets)

    @property
    def datasets(self):
        """Datasets in this Clustering."""
        return self._datasets

    def copy(self):
        """Return copy of clustering."""
        cl = Clustering()
        cl._datasets = list(self.datasets)
        for prop in ("dists_", "cn_frac_", "vect", "vect_sources_", "vect_labels_"):
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
                    f"Clustering already contains a dataset with label '{ds.label}'."
                )
            self._datasets.append(ds)

        return self

    def test_labels(self, subset=None, progress=True, **kwargs):
        """Test impact of removal of individual labels.

        Parameters
        ----------
        subset :    str | list thereof, optional
                    If provided will only test a subset of the labels. Uses
                    regex!
        **kwargs
                    Keyword arguments are passed through to `.compile()`.

        Returns
        -------
        pandas.DataFrame
                    A DataFrame containing a row for each tested label and:
                     1. `difference` is the absolute difference in distances
                        between the full and the distance matrix after removing
                        given label
                     2. `mantel_r` is the Mantel correlation coefficient

        """
        import mantel

        # Make sure we aren't verbose unless asked for
        kwargs["verbose"] = kwargs.get("verbose", False)

        # Make sure we have a baseline compilation
        self.compile(**kwargs)

        if subset is None:
            subset = self.vect_.columns.unique()

        # Keep the original, full distances as reference
        org = self.dists_.copy()

        # Get the number of synapses for each label
        n_syn = (
            self.vect_.sum(axis=0)
            .reset_index(drop=False)
            .groupby("index")
            .sum()
            .iloc[:, 0]
            .to_dict()
        )

        results = []
        for la in tqdm(subset, disable=not progress, desc="Testing labels"):
            kwargs["exclude_labels"] = la
            self.compile(**kwargs)

            diff = np.abs(org.values - self.dists_.values).sum()
            man = mantel.test(org.values, self.dists_.values, perms=1000)
            results.append([la, diff, man.r])

        results = pd.DataFrame(results, columns=["label", "difference", "mantel_r"])
        results["n_syn"] = results.label.map(n_syn)
        return results

    def compile(
        self,
        join="existing",
        metric="cosine",
        force_recompile=False,
        exclude_labels=None,
        include_labels=None,
        cn_frac_threshold=None,
        augment=None,
        verbose=True,
    ):
        """Compile combined connectivity vector and calculate distance matrix.

        Parameters
        ----------
        join :      "inner" | "outer" | "existing"
                    How to combine the dataset connectivity vectors:
                      - "existing" (default) will check if a label exists in
                        theory and use it even if it's not present in the
                        connectivity vectors of all datasets
                      - "inner" will get the intersection of all labels across
                        the connectivity vectors
                      - "outer" will use all available labels
        metric :    "cosine" | "Euclidean"
                    Metric to use for distance calculations.
        exclude_labels : str | list of str, optional
                    If provided will exclude given labels from the observation
                    vector. This uses regex!
        include_labels : str | list of str, optional
                    If provided will only include given labels from the
                    observation vector. This uses regex!
        force_recompile : bool
                    If True, will recompile connectivity vectors for each data
                    set even if they already exist.
        augment :   DataFrame, optional
                    An second distance matrix (e.g. from NBLAST) that will be
                    used to augment the connectivity-based scores. Index and
                    columns must contain all the IDs in this clustering.

        Returns
        -------
        self

        """
        if len(self) <= 1:
            raise ValueError("Clustering requires >=2 datasets")

        assert metric in ("cosine", "Euclidean")
        assert join in ("inner", "outer", "existing")

        if include_labels is not None and exclude_labels is not None:
            raise ValueError("Can't provide both `include_labels` and `exclude_labels`")

        if not isinstance(augment, (pd.DataFrame, type(None))):
            raise TypeError(f'`augment` must be DataFrame, got "{type(augment)}"')

        all_ids = np.concatenate([ds.neurons for ds in self.datasets])
        if len(all_ids) > len(list(set(all_ids))):
            print("Warning: Looks the clustering contains non-unique IDs!")

        # First compile datasets if necessary
        for i, ds in enumerate(self.datasets):
            if not hasattr(ds, "edges_") or force_recompile:
                printv(
                    f'Compiling connectivity vector for "{ds.label}" '
                    f"({ds.type}) [{i+1}/{len(self.datasets)}]",
                    verbose=verbose,
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
        if join == "inner":
            to_use = set(self.datasets[0].edges_.pre.unique().tolist()) | set(
                self.datasets[0].edges_.post.unique().tolist()
            )
            for ds in self.datasets[1:]:
                to_use = to_use & (
                    set(ds.edges_.pre.unique().tolist())
                    | set(ds.edges_.post.unique().tolist())
                )
            to_use = list(to_use)
        elif join in ("outer", "existing"):
            # Get all labels
            to_use = set(self.datasets[0].edges_.pre.unique().tolist()) | set(
                self.datasets[0].edges_.post.unique().tolist()
            )
            for ds in self.datasets[1:]:
                to_use = to_use | set(ds.edges_.pre.unique().tolist())
                to_use = to_use | set(ds.edges_.post.unique().tolist())
            to_use = list(to_use)
            if join == "existing":
                exists = np.ones(len(to_use), dtype=bool)
                for ds in self.datasets:
                    exists[~ds.label_exists(to_use)] = False
                to_use = np.array(to_use)[exists]

        # Exclude labels
        if exclude_labels is not None:
            to_exclude = set()
            for la in make_iterable(exclude_labels):
                to_exclude |= {t for t in to_use if re.match(la, t)}
            if to_exclude:
                printv(
                    f"Excluding {len(to_exclude)} of {len(to_use)} labels: "
                    f"{to_exclude}",
                    verbose=verbose,
                )
                to_use = to_use[~np.isin(to_use, list(to_exclude))]  # keep the list()

        # Include labels
        if include_labels is not None:
            to_include = set()
            for la in make_iterable(include_labels):
                to_include |= {t for t in to_use if re.match(la, t)}
            if to_include:
                printv(
                    f"Including {len(to_include)} of {len(to_use)} labels: "
                    f"{to_include}",
                    verbose=verbose,
                )
                to_use = to_use[np.isin(to_use, list(to_include))]

        # Subset edge lists to these labels
        for ds in self.datasets:
            edges = ds.edges_proc_
            is_up = edges.post.isin(ds.neurons)
            up_shared = edges.pre.isin(to_use)

            is_down = edges.pre.isin(ds.neurons)
            down_shared = edges.post.isin(to_use)

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
            down = adj.reindex(index=ds.neurons, columns=to_use)
            # Get upstream adjacency (rows = shared inputs, columns = queries)
            up = adj.reindex(columns=ds.neurons, index=to_use)
            adjacencies.append(pd.concat((down, up.T), axis=1).fillna(0))
            sources += [ds.label] * adjacencies[-1].shape[0]
            labels += ds.get_labels(ds.neurons).tolist()
        self.vect_ = pd.concat(adjacencies, axis=0).astype(VECT_DTYPE)
        self.vect_sources_ = np.array(sources)
        self.vect_labels_ = np.array(labels)

        printv(
            f"Generated a {self.vect_.shape[0]:,} by {self.vect_.shape[1]:,} observation vector.",
            verbose=verbose,
        )

        # Calculate fraction of connectivity used for the observation vector
        syn_counts_before = {}
        for ds in self.datasets:
            syn_counts_before.update(ds.syn_counts)

        syn_counts_after = self.vect_.sum(axis=1)
        self.cn_frac_ = syn_counts_after / syn_counts_after.index.map(syn_counts_before)

        printv(
            f"Using on average {self.cn_frac_.mean():.1%} of neurons' synapses.",
            verbose=verbose,
        )
        printv(
            f"Worst case is keeping {self.cn_frac_.min():.1%} of its synapses.",
            verbose=verbose,
        )

        if cn_frac_threshold is not None:
            keep = (self.cn_frac_.fillna(0) > cn_frac_threshold).values
            printv(
                f"Dropping {(~keep).sum():,} neurons with <= {cn_frac_threshold} "
                "kept connectivity.",
                verbose=verbose,
            )
            self.vect_ = self.vect_.iloc[keep]
            self.vect_sources_ = self.vect_sources_[keep]
            self.vect_labels_ = self.vect_labels_[keep]

        # Calculate distances
        self.dists_ = calculate_distance(
            self.vect_, augment=augment, metric=metric, verbose=verbose
        )
        self.dists_.columns = [
            f"{l}_{ds}" for l, ds in zip(self.vect_labels_, self.vect_sources_)
        ]

        printv("All done.", verbose=verbose)
        return self

    @req_compile
    def to_table(self, clusters=None, link_method="ward", orient="neurons"):
        """Generate a table in the same the order as dendrogram.

        Parameters
        ----------
        clusters :    iterable, optional
                      Clusters as provided by `extract_clusters`. Must be
                      membership, i.e. `[0, 0, 0, 1, 2, 1, 0, ...]`.
        link_method : str
                      Linkage method for sorting neurons.
        orient :      "neurons" | "clusters
                      Determines output:
                        - for "neurons" each row will be a neuron
                        - for "clusters" each row will be a ``cluster``

        Returns
        -------
        DataFrame

        """
        assert orient in (
            "neurons",
            "clusters",
        ), 'orient must be "clusters" or "neurons"'

        if orient == "clusters" and clusters is None:
            raise ValueError('Must provide `clusters` when `orient="clusters"`')

        # Turn similarity into distances
        x = self.dists_
        if x.values[0, 0] >= 0.999:
            x = 1 - x

        # Generate linkage and extract order
        Z = linkage(
            squareform(self.dists_.values, checks=False),
            method=link_method,
            preserve_input=False,
        )
        leafs = leaves_list(Z)

        # Generate table
        table = pd.DataFrame()
        table["id"] = self.dists_.index.values[leafs]

        # Add labels
        labels = {}
        for ds in self.datasets:
            labels.update(dict(zip(ds.neurons, ds.get_labels(ds.neurons))))
        table["label"] = table.id.map(labels).astype(str)
        # Neurons without an actual type will show up with their own ID as label
        # Here we set these to None
        table.loc[table.label == table.id.astype(str), "label"] = None

        # Add a column for the dataset
        ds = {i: ds.label for ds in self.datasets for i in ds.neurons}
        table["dataset"] = table.id.map(ds)

        # Add fraction of connectivity used
        table["cn_frac_used"] = table.id.map(self.cn_frac_.to_dict())

        # Order in the dendrogram
        table["dend_ix"] = table.index

        # Last but not least: add clusters (if provided)
        if clusters is not None:
            if not isinstance(clusters, (np.ndarray, list)):
                raise TypeError(
                    "Expected `clusters` to be list or array, got "
                    f'"{type(clusters)}".'
                )
            clusters = np.asarray(clusters)
            if clusters.ndim != 1:
                raise ValueError("`clusters` must be a flat list or array")
            if len(clusters) != len(table):
                raise ValueError(f"Got {len(clusters)} for {len(table)} rows")

            table["cluster"] = clusters[leafs]

        return table

    @req_compile
    def extract_clusters(self, N, out="membership", **kwargs):
        """Extract clusters.

        Parameters
        ----------
        N :     int
                Number of clusters to make.
        out :   'ids' | 'membership' | 'labels'
                Determines the output format:
                    - `ids` returns lists of neuron IDs
                    - `membership` returns a cluster ID for each neuron
                    - `labels` returns lists of neuron labels
        **kwargs
                Keyword arguments passed to `linkage()`.

        Returns
        -------
        See `out` parameter.

        """
        # Turn similarity into distances
        x = self.dists_
        if x.values[0, 0] >= 0.999:
            x = 1 - x
        defaults = CLUSTER_DEFAULTS.copy()
        defaults.update(kwargs)
        Z = linkage(
            squareform(x.values, checks=False), preserve_input=False, **defaults
        )

        cl = cut_tree(Z, n_clusters=N).flatten()
        if out == "membership":
            return cl
        elif out == "ids":
            return [self.dists_.index.values[cl == i] for i in np.unique(cl)]
        elif out == "labels":
            return [self.dists_.columns.values[cl == i] for i in np.unique(cl)]
        else:
            raise ValueError(f'Unknown output format "{out}"')

    @req_compile
    def extract_homogeneous_clusters(
        self,
        out="membership",
        eval_func=is_good,
        max_dist=None,
        min_dist=None,
        min_dist_diff=None,
        link_method="ward",
        verbose=False,
    ):
        """Extract homogenous clusters from clustermap or distance matrix.

        Parameters
        ----------
        out :           'ids' | 'membership' | 'labels'
                        Determines the output format:
                         - `ids` returns lists of neuron IDs
                         - `membership` returns a cluster ID for each neuron
                         - `labels` returns lists of neuron labels
        eval_func :     callable
                        Must accept two positional arguments:
                         1. A numpy array of label counts (e.g. `[1, 1, 2]`)
                         2. An integer describing how many unique labels we expect
                        Must return True if cluster composition is acceptable and
                        False if it isn't.
        min/max_dist :  float
                        Use this to set a range of between-cluster distances at
                        which we are allowed to make clusters.
        link_method :   str
                        Method to use for generating the linkage.

        Returns
        -------
        See `out` parameter.

        """

        if len(np.unique(self.vect_sources_)) != len(self):
            print("Warning: it appears that dataset labels are not unique")

        cl = extract_homogeneous_clusters(
            self.dists_,
            self.vect_sources_,
            eval_func=eval_func,
            link_method=link_method,
            max_dist=max_dist,
            min_dist=min_dist,
            min_dist_diff=min_dist_diff,
            verbose=verbose,
        )
        if out == "membership":
            return cl
        elif out == "ids":
            return [self.dists_.index.values[cl == i] for i in np.unique(cl)]
        elif out == "labels":
            return [self.dists_.columns.values[cl == i] for i in np.unique(cl)]
        else:
            raise ValueError(f'Unknown output format "{out}"')

    @req_compile
    def plot_dendrogram(
        self,
    ):
        """Plot dendrogram."""
        pass

    @req_compile
    def plot_clustermap(
        self, x_labels="{label}_{dataset}", y_labels="{id}", fontsize=4, **kwargs
    ):
        """Plot connectivity distance as cluster heatmap.

        Parameters
        ----------
        x/y_labels :    str
                        Formatting for tick labels on x- and y-axis, respectively.
                        Possible values are "id", "label" and "dataset".
        fontsize :      int | float | None
                        Fontsize for tick labels. If `None`, will remove labels.
        **kwargs 
                        Keyword arguments are passed to seaborn.clustermap
                        
        Returns
        -------
        cm :            sns.clustermap
                        The distances are available via ``cm.data``

        """
        dists = self.dists_
        Z = linkage(
            squareform(dists.values, checks=False),
            preserve_input=False,
            **CLUSTER_DEFAULTS,
        )

        # We seem to sometimes get negative cluster distances which dendrogram()
        # does not like - perhaps something to do with neurons not having any
        # shared connectivity? Anywho: let's set distances to zero
        Z[Z[:, 2] < 0, 2] = 0

        row_colors = [_percent_to_color(v) for v in self.cn_frac_.values]

        ds_dict = {i: ds.label for ds in self.datasets for i in ds.neurons}
        ds_cmap = dict(
            zip(
                [ds.label for ds in self.datasets],
                sns.color_palette("muted", len(self.datasets)),
            )
        )
        col_colors = [ds_cmap.get(ds_dict.get(i), "k") for i in dists.index]

        label_dict = {}
        for ds in self.datasets:
            label_dict.update(dict(zip(ds.neurons, ds.get_labels(ds.neurons))))

        cm = sns.clustermap(
            dists,
            cbar_pos=None,
            cmap="Greys_r",
            row_colors=row_colors,
            col_colors=col_colors,
            row_linkage=Z,
            col_linkage=Z,
            **kwargs
        )
        ax = cm.ax_heatmap
        ix = cm.dendrogram_row.reordered_ind
        ax.set_xticks(np.arange(len(dists)) + 0.5)
        ax.set_yticks(np.arange(len(dists)) + 0.5)

        if fontsize is None:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            xl = [
                x_labels.format(dataset=ds_dict[i], label=label_dict[i], id=i)
                for i in dists.index.values[ix]
            ]
            yl = [
                y_labels.format(dataset=ds_dict[i], label=label_dict[i], id=i)
                for i in dists.index.values[ix]
            ]
            ax.set_xticklabels(xl, fontsize=fontsize)
            ax.set_yticklabels(yl, fontsize=fontsize)

        # Bring back the scale for one of the dendrograms
        ax_d = cm.ax_col_dendrogram
        ax_d.set_axis_on()
        ax_d.spines["left"].set_visible(True)
        ylim = ax_d.get_ylim()
        if ylim[1] <= 0.1:
            interval = 0.05
            rnd = 2
        elif ylim[1] <= 0.3:
            interval = 0.1
            rnd = 1
        elif ylim[1] <= 2:
            interval = 0.2
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

    @req_compile
    def plot_cn_frac(self, split=True, bins=None):
        """Plot fraction of connectivity used.

        Parameters
        ----------
        split :     bool
                    If True, will split datasets into individual graphs.
        bins
        """
        if isinstance(bins, type(None)):
            bins = bins = np.arange(0, 1.05, 0.05)

        if split:
            fig, axes = plt.subplots(nrows=len(self), sharex=True)

            for i, ds in enumerate(self.datasets):
                this = self.cn_frac_.loc[ds.neurons]
                axes[i].hist(
                    this.values,
                    bins=bins,
                    color="k",
                    density=True,
                    width=(bins[1] - bins[0]) * 0.95,
                )
                axes[i].set_ylabel(ds.label)
        else:
            fig, ax = plt.subplots()
            ax.hist(
                self.cn_frac_.values,
                bins=bins,
                color="k",
                width=(bins[1] - bins[0]) * 0.95,
            )
            axes = [ax]

        axes[-1].set_xlabel("fraction of connectivity used")

        sns.despine(trim=True)
        plt.tight_layout()

        return axes

    @req_compile
    def interactive_dendrogram(
        self, clusters=None, labels=None, link_method="ward", open=True, **kwargs
    ):
        """Make an interactive dendrogram using plotly dash.

        Parameters
        ----------

        """

        from .app import interactive_dendrogram

        app = interactive_dendrogram(
            self.dists_,
            clusters=clusters,
            labels=labels,
            symbols=self.vect_sources_,
            marks=self.dists_.index.astype(str),
            linkage_method=link_method,
        )

        defaults = dict(debug=False, port="8051")
        defaults.update(kwargs)

        app.run_server(**defaults)

        if open:
            pass

        return app

    def remove_low_cn(self, threshold=0, recompile=False, inplace=False):
        """Drop neurons with less than `threshold` connectivity going into
        the clustering.

        Parameters
        ----------
        threshold :     float [0-1]
                        Connectivity fraction threshold for dropping neurons.
        recompile :     bool
                        For very large clusterings it can be faster to just
                        recompile without the offending neurons instead of
                        making copies of all the data.
        inplace :       bool
                        If True, will drop neurons inplace. If False, will
                        return a copy.

        """
        if not hasattr(self, "dists_"):
            raise ValueError("Must run .compile() first.")

        cl = self
        if not inplace:
            cl = cl.copy()

        to_drop = (cl.cn_frac_.fillna(0) <= threshold).values

        print(
            f"Dropping {to_drop.sum():,} ({to_drop.sum()/to_drop.shape[0]:.1%}) "
            "neurons from the clustering.",
            flush=True,
        )

        cl.dists_ = cl.dists_.loc[~to_drop, ~to_drop]
        cl.vect_ = cl.vect_.loc[~to_drop]
        cl.vect_sources_ = cl.vect_sources_[~to_drop]
        cl.vect_labels_ = cl.vect_labels_[~to_drop]
        cl.cn_frac_ = cl.cn_frac_[~to_drop]

        return cl

    def split(self, x):
        """Split this clustering.

        Parameters
        ----------
        x :     int | iterable
                How to split:
                 - `integer` will split into N clusters by cutting the dendrogram
                 - `iterable` must provide a label for each neuron

        Returns
        -------
        list
                List of Clusterings.

        """
        if not hasattr(self, "dists_"):
            raise ValueError("Must run .compile() first.")

        if isinstance(x, int):
            x = self.extract_clusters(x)

        if not isinstance(x, (list, np.ndarray)):
            raise TypeError(f'`x` must be integer or iterable, got "{type(x)}".')

        if len(x) != len(self.dists_):
            raise ValueError(f"Got {len(x)} labels for {len(self.dists_)} neurons.")

        for l in x:
            pass


def _percent_to_color(x):
    """Take fraction and turn into color category."""
    if x < 0.1:
        c = "red"
    elif x < 0.3:
        c = "orange"
    elif x < 0.5:
        c = "yellow"
    else:
        c = "g"
    return mcl.to_rgb(c)


def generate_clustering(
    fw=None,
    hb=None,
    mcns=None,
    split_lr=True,
    live_annot=False,
    upstream=True,
    downstream=True,
    fw_cn_file=None,
    fw_materialization=783,
):
    """Shortcut for generating a clustering on the pre-defined datasets.

    Parameters
    ----------
    fw :        str | int | list thereof
                FlyWire root ID(s) or cell type(s). Will automatically be split
                into left and right. See also `split_lr` parameter.
    hb :        str | int | list thereof
                Hemibrain body ID(s) or cell type(s). Will automatically be split
                into left and right. See also `split_lr` parameter.
    mcns :      str | int | list thereof
                MaleCNS body ID(s) or cell type(s). Will automatically be split
                into left and right. See also `split_lr` parameter.
    split_lr :  bool
                If True, will split IDs into left and right automatically.
    live_annot : bool
                Whether to use live annotations. This requires access to SeatTable.
    upstream :  bool
                Whether to use input connectivity.
    downstream : bool
                Whether to use output connectivity.
    fw_cn_file : str
                Path to FlyWire edge list.
    fw_materialization : int
                Materialization to use for FlyWire. Must match `fw_cn_file` if
                that is provided.

    """
    datasets = []

    if fw is not None:
        # Use the dataset to parse `fw` into root IDs
        fw = FlyWire(
            live_annot=live_annot,
            upstream=upstream,
            downstream=downstream,
            label="FW",
            cn_file=fw_cn_file,
            materialization=fw_materialization,
        ).add_neurons(fw)
        # Now split into left/right
        if split_lr:
            fw_ann = fw.get_annotations()
            is_left = np.isin(
                fw.neurons, fw_ann[fw_ann.side == "left"].root_id.astype(int)
            )
            fw_left = FlyWire(
                live_annot=live_annot,
                upstream=upstream,
                downstream=downstream,
                label="FwL",
                cn_file=fw_cn_file,
                materialization=fw_materialization,
            ).add_neurons(np.array(fw.neurons)[is_left])
            fw_right = FlyWire(
                live_annot=live_annot,
                upstream=upstream,
                downstream=downstream,
                label="FwR",
                cn_file=fw_cn_file,
                materialization=fw_materialization,
            ).add_neurons(np.array(fw.neurons)[~is_left])

            if len(fw_left.neurons):
                datasets.append(fw_left)
            if len(fw_right.neurons):
                datasets.append(fw_right)
        elif len(fw.neurons):
            datasets.append(fw)

    if hb is not None:
        # Use the dataset to parse `hb` into body IDs
        hb = Hemibrain(
            live_annot=live_annot,
            upstream=upstream,
            downstream=downstream,
            label="HB",
        ).add_neurons(hb)

        # Now split into left/right
        if split_lr:
            hb_ann = hb.get_annotations()
            is_left = np.isin(
                hb.neurons, hb_ann[hb_ann.side == "left"].bodyId.astype(int)
            )

            if any(is_left):
                datasets.append(
                    Hemibrain(
                        live_annot=live_annot,
                        upstream=upstream,
                        downstream=downstream,
                        label="HbL",
                    ).add_neurons(np.array(hb.neurons)[is_left])
                )

            if any(~is_left):
                datasets.append(
                    Hemibrain(
                        live_annot=live_annot,
                        upstream=upstream,
                        downstream=downstream,
                        label="HbR",
                    ).add_neurons(np.array(hb.neurons)[~is_left])
                )
        elif len(hb.neurons):
            datasets.append(hb)

    if mcns is not None:
        # Use the dataset to parse `mcns` into body IDs
        mcns = MaleCNS(
            upstream=upstream, downstream=downstream, label="McnsL"
        ).add_neurons(mcns)

        # Now split into left/right
        if split_lr:
            mcns_ann = mcns.get_annotations()
            is_left = np.isin(
                mcns.neurons, mcns_ann[mcns_ann.soma_side.isin(["left", "L"])].bodyid.astype(int)
            )
            mcns_left = MaleCNS(
                upstream=upstream, downstream=downstream, label="McnsL"
            ).add_neurons(np.array(mcns.neurons)[is_left])
            mcns_right = MaleCNS(
                upstream=upstream, downstream=downstream, label="McnsR"
            ).add_neurons(np.array(mcns.neurons)[~is_left])

            if len(mcns_left.neurons):
                datasets.append(mcns_left)
            if len(mcns_right.neurons):
                datasets.append(mcns_right)
        elif len(mcns.neurons):
            datasets.append(mcns)

    if not len(datasets):
        raise ValueError("Must provide IDs for at least one dataset")

    return Clustering(datasets)
