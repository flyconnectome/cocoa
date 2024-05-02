import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from ..utils import printv
from ..distance import calculate_distance


class DataSet(ABC):
    def __init__(self, label):
        self.label = label
        self.neurons = np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return len(self.neurons)

    def __repr__(self):
        return f"class {self.type} <label={self.label};neurons={len(self.neurons)}>"

    @property
    def type(self):
        return str(type(self))[:-2].split(".")[-1]

    @property
    def syn_counts(self):
        if not hasattr(self, "edges_"):
            raise ValueError("Must first compile connectivity")
        up = (
            self.edges_[self.edges_.post.isin(self.neurons)]
            .groupby("post")
            .weight.sum()
            .to_dict()
        )
        down = (
            self.edges_[self.edges_.pre.isin(self.neurons)]
            .groupby("pre")
            .weight.sum()
            .to_dict()
        )
        return {n: up.get(n, 0) + down.get(n, 0) for n in self.neurons}

    def add_neurons(self, x, **kwargs):
        """Add neurons to dataset.

        Parameters
        ----------
        x :     str | int | list thereof
                Something that can be parsed into IDs. Details depend on the
                dataset.

        """
        new_neurons = self._add_neurons(x, **kwargs)

        if not len(new_neurons):
            print(f'No neurons matching "{x}" found.')

        self.neurons = np.unique(
            np.append(self.neurons, new_neurons)
        )
        return self

    def get_ngl_scene(self):
        return NotImplementedError

    @abstractmethod
    def _add_neurons(self, x, **kwargs):
        """Turn `x` into IDs."""
        pass

    @abstractmethod
    def get_labels(self, x, **kwargs):
        """Get label for ID `x`."""
        pass

    @abstractmethod
    def compile(self):
        """Compile connectivity vector."""
        pass

    def connectivity_dist(
        self,
        metric="cosine",
        force_recompile=False,
        augment=None,
        labelled_only=True,
        verbose=True,
    ):
        """Calculate cosine distance for neurons in this dataset.

        Parameters
        ---------
        """
        if not hasattr(self, "edges_") or force_recompile:
            printv(
                f'Compiling connectivity vector for "{self.label}" ({self.type})',
                verbose=verbose,
            )
            self.compile()
        edges = self.edges_.copy()

        # Could set this to only typed neurons
        to_use = list(set(edges[["pre", "post"]].values.flatten().tolist()))
        to_use = np.array(to_use)[self.label_exists(to_use)]

        is_up = edges.post.isin(self.neurons)
        up_shared = edges.pre.isin(to_use)

        is_down = edges.pre.isin(self.neurons)
        down_shared = edges.post.isin(to_use)

        edges = edges.loc[(is_up & up_shared) | (is_down & down_shared)]

        adj = edges.groupby(["pre", "post"]).weight.sum().unstack()
        # Get downstream adjacency (rows = queries, columns = shared targets)
        down = adj.reindex(index=self.neurons, columns=to_use)
        # Get upstream adjacency (rows = shared inputs, columns = queries)
        up = adj.reindex(columns=self.neurons, index=to_use)

        self.vect_ = pd.concat((down, up.T), axis=1).fillna(0)

        # Calculate fraction of connectivity used for the observation vector
        syn_counts_after = self.vect_.sum(axis=1)
        self.cn_frac_ = syn_counts_after / syn_counts_after.index.map(self.syn_counts)

        printv(
            f"Using on average {self.cn_frac_.mean():.1%} of neurons' synapses.",
            verbose=verbose,
        )
        printv(
            f"Worst case is keeping {self.cn_frac_.min():.1%} of its synapses.",
            verbose=verbose,
        )

        self.dists_ = calculate_distance(
            self.vect_, metric=metric, verbose=verbose, augment=augment
        )
        printv("All Done.", verbose=verbose)

        return self
