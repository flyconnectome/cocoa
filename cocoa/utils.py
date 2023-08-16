import six
import functools

import numpy as np
import pandas as pd

from collections.abc import Iterable


def printv(*args, verbose=True, **kwargs):
    """Thin wrapper around print function."""
    if verbose:
        print(*args, **kwargs)


def check_frame(x, required_cols=None, dtypes=None):
    """Check dataframe."""
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f'Expected pandas.DataFrame, got "{type(x)}"')

    if not isinstance(required_cols, type(None)):
        if isinstance(required_cols, str):
            required_cols = [required_cols]
        for c in required_cols:
            if c not in x.columns:
                raise ValueError(f"DataFrame has to have a {c} column")

    if not isinstance(dtypes, type(None)):
        for c, types in dtypes.items():
            if not isinstance(types, (tuple, list)):
                types = (types,)

            if c not in x.columns:
                raise ValueError(f"DataFrame has to have a {c} column")
            if x[c].dtype not in types:
                raise ValueError(
                    f'Column {c} is expected to be of type "{types}" '
                    f"got {x[c].dtype}"
                )


def make_iterable(x, force_type=None) -> np.ndarray:
    """Force input into a numpy array.

    For dicts, keys will be turned into array.
    """
    if not isinstance(x, Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, (dict, set)):
        x = list(x)

    return np.asarray(x, dtype=force_type)


def req_compile(func):
    """Check if we need to compile connectivity."""
    @functools.wraps(func)
    def inner(*args, **kwargs):
        if not hasattr(args[0], "dists_"):
            args[0].compile()
        return func(*args, **kwargs)
    return inner