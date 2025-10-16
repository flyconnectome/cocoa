import numpy as np
import pandas as pd

from numba import njit
from sklearn import preprocessing


@njit(debug=False)
def get_row(v, row, n):
    """Extract a single row from a condensed distance matrix."""
    values = np.zeros(n, dtype=v.dtype)
    for j in range(n):
        if row == j:
            continue

        if row < j:
            ix = n * row - row * (row + 1) // 2 + j - 1 - row
        else:
            ix = n * j - j * (j + 1) // 2 + row - 1 - j
        values[j] = v[ix]
    return values


def silhoutte_samples_knn(X, labels, k=10):
    """Calculate the per-sample silhouette scores using only the k-nearest neighbors.

    Parameters
    ----------
    X :         (N, N) or (N * (N - 1) / 2, ) numpy array
                Distance matrix between samples. Can be squareform or condensed.
    labels :    (N, ) numpy array
                Labels for each sample.
    k :         int
                Number of neighbors to consider.

    Returns
    -------
    S :         ndarray
                Silhouette scores where 1 indicates that the sample is well-clustered,
                -1 indicates that the sample is likely in the wrong cluster, and 0
                indicates that the sample is on the border between two clusters.

    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if X.ndim == 2:
        assert X.shape[0] == X.shape[1]
        N = X.shape[0]
    elif X.ndim == 1:
        N = int((1 + np.sqrt(1 + 8 * X.shape[0])) / 2)
    else:
        raise ValueError("Invalid shape for X:", X.shape)

    if N != len(labels):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of samples ({N})"
        )

    # Turn labels into contiguous integers
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_int = le.transform(labels)

    return _silhoutte_samples_knn(X, labels_int, N, k)


@njit(debug=False)
def _silhoutte_samples_knn(X, labels_int, N, k=10):
    # Prepare the output array
    scores = np.zeros(N, dtype=X.dtype)

    # Iterate over all rows
    for i in range(N):
        # Get this row
        if X.ndim == 2:
            row = X[i]
        else:
            # Get the full row out of the condensed distance matrix
            row = get_row(X, i, N)

        # Get the top K neighbors
        nn = np.argsort(row)[1 : k + 1]

        # Get the distances
        d = row[nn]

        # Get the labels
        la = labels_int[nn]

        # Get distance to same labels
        same_label = la == labels_int[i]

        if not np.any(same_label):
            scores[i] = -1.0
            continue
        elif np.all(same_label):
            scores[i] = 1.0
            continue

        # Get the label for the closest cluster
        for ol in la:
            if ol != labels_int[i]:
                break

        d_same = d[same_label].mean()
        d_other = d[la == ol].mean()
        scores[i] = (d_other - d_same) / max(d_other, d_same)

    return scores
