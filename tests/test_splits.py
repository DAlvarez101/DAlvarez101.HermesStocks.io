"""Tests for time-based dataset splits."""
import pandas as pd
import pytest

from dfw_temp_model.training.splits import time_based_split


def test_time_based_split_dates():
    """Simple case with 4 date strings across 2022-2025."""
    df = pd.DataFrame(
        {"value": [1, 2, 3, 4]},
        index=["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"],
    )
    train, val, test = time_based_split(df, train_end="2022-12-31", val_end="2023-12-31")

    assert len(train) == 1
    assert train.index[0] == "2022-01-01"

    assert len(val) == 1
    assert val.index[0] == "2023-01-01"

    assert len(test) == 2
    assert list(test.index) == ["2024-01-01", "2025-01-01"]


def test_time_based_split_with_datetime_index():
    """Split works when the index is a DatetimeIndex."""
    dates = pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"])
    df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=dates)
    train, val, test = time_based_split(df, train_end="2022-12-31", val_end="2023-12-31")

    assert len(train) == 1
    assert train.index[0] == pd.Timestamp("2022-01-01")

    assert len(val) == 1
    assert val.index[0] == pd.Timestamp("2023-01-01")

    assert len(test) == 2
    assert list(test.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")]
