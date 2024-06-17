from .core import DataSet

from abc import ABC, abstractproperty


class NeuprintDataSet(DataSet, ABC):
    """Base class for datasets that use the neuprint API."""

    _roi_col = "roi"

    def __init__(self, label):
        super().__init__(label)

    @abstractproperty
    def neuprint_client(self):
        pass

    def get_roi_completeness(self):
        """Get ROI completeness for all neurons in this dataset."""
        return self.neuprint_client.fetch_roi_completeness()

    def get_meshes(self, x):
        """Fetch meshes for given IDs.

        Parameters
        ----------
        x :         int | list | np.ndarray
                    Body IDs to fetch meshes for.

        """

        import navis.interfaces.neuprint as neu

        return neu.fetch_mesh_neuron(x, client=self.neuprint_client)
