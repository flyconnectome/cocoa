import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse import coo_array
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

from .utils import printv

DISTS_DTYPE = np.float32
VECT_DTYPE = np.uint16


def calculate_distance(vect, augment=None, metric="cosine", n_batches=None, verbose=True):
    assert metric in ("cosine", "Euclidean")

    printv(f"Calculating {metric} distances... ", verbose=verbose, end="", flush=True)

    if n_batches and n_batches > 1:
        # Split the data
        dists = np.zeros((vect.shape[0], vect.shape[0]), dtype=DISTS_DTYPE)

        # Split `vect` into `n_batches` parts
        batches = np.array_split(vect, n_batches, axis=0)

        row_ix = 0
        col_ix = 0
        printv("\n", verbose=verbose, end="")
        for i, batch1 in enumerate(batches):
            printv(f"  Batch {i+1}/{n_batches}... ", verbose=verbose, end="", flush=True)
            for j, batch2 in enumerate(batches):
                if metric == "Euclidean":
                    this_dists = cdist(batch1, batch2).astype(DISTS_DTYPE, copy=False).round(6)
                elif metric == "cosine":
                    this_dists = cosine_similarity(coo_array(batch1), coo_array(batch2), dense_output=True)
                    # Turn into distance inplace
                    np.subtract(1, this_dists, out=this_dists)

                # Fill the distance matrix
                dists[
                    row_ix : row_ix + this_dists.shape[0],
                    col_ix : col_ix + this_dists.shape[1],
                ] = this_dists

                col_ix += this_dists.shape[1]
            row_ix += this_dists.shape[0]
            col_ix = 0
            printv("Done.", verbose=verbose)

        np.fill_diagonal(dists, 0)
        printv("All Done.", verbose=verbose)
    elif metric == "Euclidean":
        dists = (
            squareform(pdist(vect, checks=False))
            .astype(DISTS_DTYPE, copy=False)
            .round(6)
        )
        printv("Done.", verbose=verbose)
    elif metric == "cosine":
        # Note that we're converting to sparse array here. That's because
        # the vector (if it's large) is typically sparse (<1% filled)
        # and cosine_similarity is much much faster on sparse arrays
        dists = cosine_similarity(coo_array(vect), dense_output=True)

        # The resulting distance `dists` seems to be fairly dense (even
        # though the input was sparse), so no point in keeping it as sparse
        # array. I've also looked into clipping values below e.g. 0.001 but
        # even that does not typically help much.

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


def cosine_similarity(X, Y=None, dense_output=False, dtype=np.float32):
    """Modified from sklearn.metrics.cosine_similarity to forgoe some
    unnecessary tests and to allow (down)casting the datatype going into the
    dot product function.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
        Input data.
    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features)
        Input data. If ``None``, the output will be the pairwise
        similarities of `X`.

    Returns
    -------
    kernel matrix : ndarray of shape (n_samples_X, n_samples_X)
        Returns the cosine similarity between samples in X.

    """
    X_normalized = normalize(X, copy=True).astype(dtype, copy=False)

    if Y is not None:
        Y_normalized = normalize(Y, copy=True).astype(dtype, copy=False)
    else:
        Y_normalized = X_normalized


    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

    return K


    return K
