import numpy as np

from abc import ABC, abstractmethod


class DataSet(ABC):
    def __init__(self, label):
        self.label = label
        self.neurons = np.zeros((0,), dtype=np.int64)

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
        self.neurons = np.unique(
            np.append(self.neurons, self._add_neurons(x, **kwargs))
        )
        return self

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
