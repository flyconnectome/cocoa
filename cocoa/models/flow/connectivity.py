import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.sparse import coo_array

from .utils import printv


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
    max_iter :      int
                    Maximum number of iterations to run the effective connectivity.

    """

    def __init__(self, adjacency, verbose=True, progress=True):
        self.adjacency = adjacency
        self.verbose = verbose
        self.progress = progress
        self._current_activity = None

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency):
        # Convert pandas DataFrame to numpy array
        if isinstance(pd.DataFrame):
            adjacency = adjacency.values

        # Convert numpy array to scipy.sparse.coo_array
        if isinstance(adjacency, np.ndarray):
            assert adjacency.ndim == 2

            # If this is a (N, N) matrix
            if adjacency.shape[0] == adjacency.shape[1]:
                adjacency = coo_array(adjacency)
            # If this is a (N, 2) edge list
            elif adjacency.shape[1] == 2:
                adjacency = coo_array(
                    (adjacency[:, 0], adjacency[:, 1], np.ones(adjacency.shape[0]))
                )
            elif adjacency.shape[1] == 3:
                adjacency = coo_array(
                    (adjacency[:, 0], adjacency[:, 1], adjacency[:, 2])
                )
            else:
                raise ValueError(f"Invalid adjacency matrix shape: {adjacency.shape}")

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
            self._current_activity = self.activation

        self._current_activity = self.adjacency.dot(self._current_activity)

        return self._current_activity

    def compile(self, activation_0, stop='auto', max_iter=1000):
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

                if stop == 'auto':
                    if np.allclose(self.activity_[-1], self.activity_[-2]):
                        printv(f"Model converged after {len(self.activity_) - 1} iterations", verbose=self.verbose)
                        break
                elif len(self.activity_) >= stop:
                    printv(f"Model stopped after {len(self.activity_) - 1} iterations", verbose=self.verbose)
                    break

                if len(self.activity_) >= max_iter:
                    printv(f"Model reached maximum number of iterations ({max_iter})", verbose=self.verbose)
                    break

        return self