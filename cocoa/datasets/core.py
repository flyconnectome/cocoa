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
        props = f"label={self.label};neurons={len(self.neurons)}"
        for prop in ("meta_source", ):
            if hasattr(self, prop):
                props += f";{prop}={getattr(self, prop)}"
        return f"class {self.type} <{props}>"

    @property
    def type(self):
        return str(type(self))[:-2].split(".")[-1]

    @property
    def syn_counts(self):
        """Dictionary of synapse counts for neurons in this dataset."""
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

    @property
    def neuroglancer_source(self):
        """Neuroglancer source for this dataset."""
        if not hasattr(self, "_neuroglancer_source"):
            raise ValueError("No neuroglancer source defined for this dataset.")
        return self._neuroglancer_source

    @neuroglancer_source.setter
    def neuroglancer_source(self, value):
        self._neuroglancer_source = value

    @abstractmethod
    def add_neurons(self, x, **kwargs):
        pass

    @abstractmethod
    def _parse_ids(self, x, **kwargs):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def get_sides(self, x):
        pass

    def split_sides(self):
        """Split neurons into left and right datasets."""
        if not len(self.neurons):
            raise ValueError("No neurons in dataset.")

        sides = self.get_sides(self.neurons)
        if sides is None:
            raise ValueError("No side information available for this dataset.")

        datasets = []
        for s in np.unique(sides):
            ds = self.copy()
            ds.label = f"{self.label}_{s.lower()}"
            ds.neurons = self.neurons[sides == s]
            datasets.append(ds)

        return datasets

    def drop_neurons(self, x, **kwargs):
        """Drop neurons from dataset.

        Parameters
        ----------
        x :     str | int | list thereof
                Something that can be parsed into IDs. Details depend on the
                dataset.
        **kwargs
                Keyword arguments are passed to `_parse_ids`.

        """
        if not len(self.neurons):
            return self

        to_drop = self._parse_ids(x, **kwargs)
        self.neurons = np.setdiff1d(self.neurons, to_drop)
        return self

    def get_ngl_scene(self):
        return NotImplementedError

    @abstractmethod
    def get_labels(self, x, **kwargs):
        """Get label for ID `x`."""
        pass

    @abstractmethod
    def get_annotations(self, **kwargs):
        """Get annotations for neurons."""
        pass

    @abstractmethod
    def get_all_neurons(self):
        """Get all neurons in dataset."""
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
        drop_unlabeled=True,
        verbose=True,
    ):
        """Calculate cosine distance for neurons in this dataset.

        Parameters
        ----------
        metric :            str
                            Distance metric to use. Default is "cosine".
        force_recompile :   bool
                            Whether to recompile the connectivity vector.
        augment :           str | None
                            Augment the connectivity vector with additional
                            information. Default is None.
        drop_unlabeled :    bool
                            Only relevant if the `self.use_types=True`:
                            Whether to drop connections to/from unlabeled neurons
                            before calculating distances. Default is True.
        verbose :           bool
                            Whether to print progress. Default is True.

        Returns
        -------
        self :              DataSet
                            The dataset with compiled distance matrix as `dists_` attribute.

        """
        if not hasattr(self, "edges_") or force_recompile:
            printv(
                f'Compiling connectivity vector for "{self.label}" ({self.type}: {len(self)} neurons).',
                verbose=verbose,
            )
            self.compile()
        edges = self.edges_.copy()

        # Compile up- and downstream connectivity
        to_use = list(set(edges[["pre", "post"]].values.flatten().tolist()))
        if self.use_types and drop_unlabeled:
            to_use = np.array(to_use)[self.label_exists(to_use)]

            is_up = edges.post.isin(self.neurons)
            is_down = edges.pre.isin(self.neurons)

            up_shared = edges.pre.isin(to_use)
            down_shared = edges.post.isin(to_use)

            edges = edges.loc[(is_up & up_shared) | (is_down & down_shared)]

        adj = edges.groupby(["pre", "post"]).weight.sum().unstack()
        # Get downstream adjacency (rows = queries, columns = shared targets)
        down = adj.reindex(index=self.neurons, columns=to_use)
        # Get upstream adjacency (rows = shared inputs, columns = queries)
        up = adj.reindex(columns=self.neurons, index=to_use)

        self.vect_ = pd.concat((down, up.T), axis=1).fillna(0).astype(np.uint32)

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
