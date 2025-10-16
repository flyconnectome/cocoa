import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.sparse import coo_array

from ...utils import printv


class EffectiveConnectivity:
    """Linear model of the effective connectivity.

    This class implements a simple linear model of the effective connectivity as
    sequence of matrix-vector multiplications. The effective connectivity is defined
    as the product of the adjacency matrix and the current activity vector.

    Parameters
    ----------
    adjacency :     pd.DataFrame | np.ndarray | scipy.sparse.coo_array
                    Normalized (M x M) adjacency matrix, (N, 3) weighted edge list
                    or sparse array. Note that internally, the adjacency is converted
                    to a sparse matrix in COO format.
    labels :        iterable
                    List of labels for the nodes in the network.
    max_iter :      int
                    Maximum number of iterations to run the effective connectivity.

    """

    def __init__(self, adjacency, labels=None, verbose=True, progress=True):
        self.adjacency = adjacency

        if labels is not None:
            self.labels = labels  # overwrites labels from adjacency

        self.verbose = verbose
        self.progress = progress
        self._current_activity = None

    @property
    def shape(self):
        return self.adjacency.shape

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency):
        # Convert pandas DataFrame to numpy array
        if isinstance(adjacency, pd.DataFrame):
            # If this is a (N, N) matrix
            if adjacency.shape[0] == adjacency.shape[1]:
                adjacency = adjacency.values
                self.labels = adjacency.columns
            # If this is a (N, 2) or (N, 3) edge list
            elif adjacency.shape[1] in (2, 3):
                id2ix = {
                    fw_id: ix
                    for ix, fw_id in enumerate(np.unique(adjacency.values.flatten()))
                }
                ix2id = {ix: fw_id for fw_id, ix in id2ix.items()}
                self.labels = [ix2id[ix] for ix in range(len(id2ix))]

                sources = adjacency.iloc[:, 0].map(id2ix)
                targets = adjacency.iloc[:, 1].map(id2ix)

                if adjacency.shape[1] == 2:
                    weights = np.ones(adjacency.shape[0])
                else:
                    weights = adjacency.iloc[:, 2].values

                adjacency = coo_array(
                    (weights, (sources, targets)), shape=(len(id2ix), len(id2ix))
                )
            else:
                raise ValueError(
                    "Adjacency matrix must be square, or (N, 2) or (N, 3) edge list. "
                    f"Got shape {adjacency.shape}. Consider using EffectiveConnectivity.from_pandas_edgelist instead."
                )

        # Convert numpy array to scipy.sparse.coo_array
        if isinstance(adjacency, np.ndarray):
            assert adjacency.ndim == 2

            # If this is a (N, N) matrix
            if adjacency.shape[0] == adjacency.shape[1]:
                adjacency = coo_array(adjacency)
            else:
                raise ValueError(f"Invalid adjacency matrix shape: {adjacency.shape}")

            self.labels = np.arange(adjacency.shape[0])

        # At this point, adjacency should be a scipy.sparse.coo_array
        assert isinstance(adjacency, coo_array)
        self._adjacency = adjacency

    @property
    def activation_0(self):
        return self._activation_0

    @activation_0.setter
    def activation_0(self, value):
        assert isinstance(value, np.ndarray)
        assert value.ndim == 1
        assert value.shape[0] == self.adjacency.shape[0]
        self._activation_0 = value

    def __iter__(self):
        return self

    def __next__(self):
        return self._advance()

    def _advance(self):
        """Advance the effective connectivity."""
        if self._current_activity is None:
            self._current_activity = self.activation_0

        self._current_activity = self.adjacency.T.dot(self._current_activity)

        return self._current_activity

    @classmethod
    def from_pandas_edgelist(cls, edges, source_col="pre", target_col="post", weight_col="weight"):
        """Create effective connectivity model from pandas DataFrame.

        Parameters
        ----------
        edges :     pd.DataFrame
                    DataFrame with columns.
        source_col : str
                    Column name for source nodes.
        target_col : str
                    Column name for target nodes.
        weight_col : str
                    Column name for edge weights.

        """
        # Map node IDs to indices
        id2ix = {fw_id: ix for ix, fw_id in enumerate(np.unique(edges[[source_col, target_col]].values.flatten()))}
        ix2id = {ix: fw_id for fw_id, ix in id2ix.items()}
        labels = [ix2id[ix] for ix in range(len(id2ix))]

        adjacency = coo_array(
            (edges[weight_col], (edges[source_col].map(id2ix), edges[target_col].map(id2ix))),
            shape=(len(id2ix), len(id2ix))
        )

        return cls(adjacency, labels=labels)

    def compile(self, activation_0, stop="auto", max_iter=1000):
        """Run the effective connectivity.

        Parameters
        ----------
        activation_0 : np.ndarray
                    Initial activation vector.
        stop :      int | str
                    If int, run the effective connectivity for `stop` iterations. If 'auto',
                    run until convergence.
        max_iter :  int
                    Maximum number of iterations to run the effective connectivity.

        Returns
        -------
        self
                    Results of the effective connectivity are stored as `self.activity_`

        """
        self.activation_0 = activation_0
        self.activity_ = [self.activation_0]

        with tqdm(total=max_iter, disable=not self.progress, leave=False) as pbar:
            while True:
                self.activity_.append(self._advance())
                pbar.update(1)

                if stop == "auto":
                    if np.allclose(self.activity_[-1], self.activity_[-2]):
                        printv(
                            f"Model converged after {len(self.activity_) - 1} iterations",
                            verbose=self.verbose,
                        )
                        break
                elif len(self.activity_) >= stop:
                    printv(
                        f"Model stopped after {len(self.activity_) - 1} iterations",
                        verbose=self.verbose,
                    )
                    break

                if len(self.activity_) >= max_iter:
                    printv(
                        f"Model reached maximum number of iterations ({max_iter})",
                        verbose=self.verbose,
                    )
                    break

        self.activity_ = pd.DataFrame(self.activity_).T

        return self

    def plot_heatmap(self, group=True, N=10, steps=10, figsize=(20, 5), **kwargs):
        """Plot heatmap of activity.

        Parameters
        ----------
        group :     bool
                    If True, group neurons by label.
        N :         int
                    Number of neurons to plot.
        steps :     int
                    Number of steps to plot.
        figsize :   tuple
                    Figure size.
        **kwargs
                    Additional keyword arguments for seaborn.heatmap.

        Returns
        -------
        ax
                    Matplotlib axis.
        """
        act = pd.DataFrame(self.activity_.iloc[:, :steps])

        if group and self.labels is not None:
            act = act.groupby(self.labels).mean()

        # Sort by mean activity and take top N
        act = act.loc[act.mean(axis=1).sort_values(ascending=False).index].iloc[:N]

        fig, ax = plt.subplots(figsize=figsize)

        defaults = dict(cmap='coolwarm', cbar=False, square=True)
        defaults.update(kwargs)

        sns.heatmap(act, ax=ax, **defaults)

        ax.set_xlabel('Step')

        plt.tight_layout()

        return ax
