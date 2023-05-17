import pandas as pd


def check_frame(x, required_cols=None, dtypes=None):
    """Check dataframe."""
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f'Expected pandas.DataFrame, got "{type(x)}"')

    if not isinstance(required_cols, type(None)):
        if isinstance(required_cols, str):
            required_cols = [required_cols]
        for c in required_cols:
            if c not in x.columns:
                raise ValueError(f'DataFrame has to have a {c} column')

    if not isinstance(dtypes, type(None)):
        for c, types in dtypes.items():
            if not isinstance(types, (tuple, list)):
                types = (types, )

            if c not in x.columns:
                raise ValueError(f'DataFrame has to have a {c} column')
            if x[c].dtype not in types:
                raise ValueError(f'Column {c} is expected to be of type "{types}" '
                                 f'got {x[c].dtype}')
