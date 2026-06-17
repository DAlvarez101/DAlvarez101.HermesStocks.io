"""Time-based train/validation/test splitting utilities."""
from typing import Tuple

import pandas as pd


def time_based_split(
    df: pd.DataFrame, train_end: str, val_end: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/validation/test sets by time.

    The index of ``df`` may be date strings or a ``DatetimeIndex``.

    - Train: dates ``<= train_end``
    - Validation: dates ``> train_end`` and ``<= val_end``
    - Test: dates ``> val_end``

    Parameters
    ----------
    df:
        Input dataframe with a date-like index.
    train_end:
        Inclusive end date for the training set, as an ISO-like date string.
    val_end:
        Inclusive end date for the validation set, as an ISO-like date string.

    Returns
    -------
    Tuple of ``(train_df, val_df, test_df)``.
    """
    cutoff_train = pd.Timestamp(train_end)
    cutoff_val = pd.Timestamp(val_end)

    # Normalize the index to Timestamps for robust comparison.
    index_as_ts = pd.to_datetime(df.index)

    train_mask = index_as_ts <= cutoff_train
    val_mask = (index_as_ts > cutoff_train) & (index_as_ts <= cutoff_val)
    test_mask = index_as_ts > cutoff_val

    return df[train_mask], df[val_mask], df[test_mask]
