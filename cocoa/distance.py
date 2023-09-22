import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_array
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from .utils import printv

DISTS_DTYPE = np.float32
VECT_DTYPE = np.uint16


def calculate_distance(vect, augment=None, metric="cosine", verbose=True):
    assert metric in ("cosine", "Euclidean")

    printv(f"Calculating {metric} distances... ", verbose=verbose, end="", flush=True)
    if metric == "Euclidean":
        dists = (
            squareform(pdist(vect, checks=False))
            .astype(DISTS_DTYPE, copy=False)
            .round(6)
        )
    elif metric == "cosine":
        # Note that we're converting to sparse array here. That's because
        # the vector (if it's large) is typically sparse (<1% filled)
        # and cosine_similarity is much much faster on sparse arrays
        dists = cosine_similarity(coo_array(vect), dense_output=True)

        # Turn into distance inplace
        np.subtract(1, dists, out=dists)

        # Make sure diagonal is actually zero
        np.fill_diagonal(dists, 0)
    printv("Done.", verbose=verbose)

    # Change columns to "type, ds" (index remains just "id")
    dists = pd.DataFrame(
        dists,
        index=vect.index,
        columns=vect.index,
    )

    if augment is not None:
        miss = dists.index[~np.isin(dists.index, augment.index)]
        if any(miss):
            raise ValueError(
                f"{len(miss)} IDs are missing from the " "augmentation matrix."
            )
        if round(augment.values[0, 0], 2) != 0:
            print(
                "Looks like the augmentation matrix contains similarities? "
                "Will invert those for you."
            )
            augment = 1 - augment
        printv(
            "Augmenting connectivity distances... ",
            end="",
            flush=True,
            verbose=verbose,
        )
        dists = (dists + augment.loc[dists.index, dists.index].values) / 2
        printv("Done", verbose=verbose)

    return dists


def cosine_similarity(X, dense_output=False, dtype=np.float32):
    """Modified from sklearn.metrics.cosine_similarity to forgoe some
    unnecessary tests and to allow casting the datatype going into the
    dot product function.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Input data.

    Returns
    -------
    kernel matrix : ndarray of shape (n_samples_X, n_samples_Y)
        Returns the cosine similarity between samples in X and Y.

    """
    X_normalized = normalize(X, copy=True).astype(dtype, copy=False)

    K = safe_sparse_dot(X_normalized, X_normalized.T, dense_output=dense_output)

    return K
