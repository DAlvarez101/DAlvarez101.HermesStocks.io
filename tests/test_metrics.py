import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.evaluation.metrics import (
    bucket_hit_rate,
    evaluate_correction,
    mae,
    rmse,
)


def test_rmse_perfect_predictions():
    obs = pd.Series([1.0, 2.0, 3.0, 4.0])
    pred = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert rmse(obs, pred) == pytest.approx(0.0)


def test_rmse_known_value():
    obs = pd.Series([0.0, 0.0, 0.0])
    pred = pd.Series([3.0, 4.0, 0.0])
    expected = np.sqrt((9 + 16 + 0) / 3)
    assert rmse(obs, pred) == pytest.approx(expected)


def test_rmse_ignores_nan_pairwise():
    obs = pd.Series([0.0, 5.0, np.nan])
    pred = pd.Series([3.0, np.nan, np.nan])
    # Only first observation contributes.
    expected = np.sqrt((0 - 3) ** 2 / 1)
    assert rmse(obs, pred) == pytest.approx(expected)


def test_mae_perfect_predictions():
    obs = pd.Series([1.0, 2.0, 3.0])
    pred = pd.Series([1.0, 2.0, 3.0])
    assert mae(obs, pred) == pytest.approx(0.0)


def test_mae_known_value():
    obs = pd.Series([0.0, 0.0, 0.0])
    pred = pd.Series([3.0, -4.0, 0.0])
    expected = (3 + 4 + 0) / 3
    assert mae(obs, pred) == pytest.approx(expected)


def test_mae_ignores_nan_pairwise():
    obs = pd.Series([0.0, np.nan, 10.0])
    pred = pd.Series([2.0, 5.0, np.nan])
    expected = abs(0 - 2) / 1
    assert mae(obs, pred) == pytest.approx(expected)


def test_bucket_hit_rate_perfect():
    obs = pd.Series([1.0, 2.0, 3.0])
    pred = pd.Series([1.2, 1.8, 3.4])
    assert bucket_hit_rate(obs, pred, bucket_width=1.0) == pytest.approx(1.0)


def test_bucket_hit_rate_zero():
    obs = pd.Series([0.0, 2.0, 4.0])
    pred = pd.Series([1.0, 1.0, 5.0])
    # Rounded obs: 0, 2, 4; rounded pred: 1, 1, 5 -> no matches
    assert bucket_hit_rate(obs, pred, bucket_width=1.0) == pytest.approx(0.0)


def test_bucket_hit_rate_known_value():
    obs = pd.Series([0.4, 1.6, 2.4, 3.6])
    pred = pd.Series([0.6, 1.4, 2.6, 3.4])
    # Rounded obs: 0, 2, 2, 4; rounded pred: 1, 1, 3, 3 -> matches at 0 of 4
    assert bucket_hit_rate(obs, pred, bucket_width=1.0) == pytest.approx(0.0)


def test_bucket_hit_rate_half():
    obs = pd.Series([0.0, 1.0, 2.0, 3.0])
    pred = pd.Series([0.0, 2.0, 2.0, 4.0])
    # Matches at index 0 and 2 -> 2/4
    assert bucket_hit_rate(obs, pred, bucket_width=1.0) == pytest.approx(0.5)


def test_bucket_hit_rate_custom_width():
    obs = pd.Series([0.0, 2.0, 4.0])
    pred = pd.Series([0.9, 2.1, 5.0])
    # Width 2: round(0/2)=0 vs 0; round(2/2)=1 vs 1; round(4/2)=2 vs 2 -> all match
    assert bucket_hit_rate(obs, pred, bucket_width=2.0) == pytest.approx(1.0)


def test_bucket_hit_rate_ignores_nan_pairwise():
    obs = pd.Series([0.0, 1.0, np.nan])
    pred = pd.Series([0.0, np.nan, np.nan])
    expected = 1 / 1
    assert bucket_hit_rate(obs, pred, bucket_width=1.0) == pytest.approx(expected)


def test_evaluate_correction_structure():
    obs = pd.Series([10.0, 12.0, 14.0])
    fcst = pd.Series([11.0, 11.0, 11.0])
    corrected = pd.Series([10.0, 12.0, 14.0])
    result = evaluate_correction(obs, fcst, corrected)
    assert set(result.keys()) == {
        "raw_rmse",
        "raw_mae",
        "corrected_rmse",
        "corrected_mae",
        "rmse_improvement",
        "bucket_hit_rate",
    }


def test_evaluate_correction_perfect_correction():
    obs = pd.Series([10.0, 12.0, 14.0])
    fcst = pd.Series([11.0, 11.0, 11.0])
    corrected = pd.Series([10.0, 12.0, 14.0])
    result = evaluate_correction(obs, fcst, corrected)
    assert result["raw_rmse"] > 0
    assert result["corrected_rmse"] == pytest.approx(0.0)
    assert result["corrected_mae"] == pytest.approx(0.0)
    assert result["rmse_improvement"] == pytest.approx(1.0)
    assert result["bucket_hit_rate"] == pytest.approx(1.0)


def test_evaluate_correction_no_improvement():
    obs = pd.Series([10.0, 12.0, 14.0])
    fcst = pd.Series([11.0, 11.0, 11.0])
    corrected = fcst.copy()
    result = evaluate_correction(obs, fcst, corrected)
    assert result["raw_rmse"] == pytest.approx(result["corrected_rmse"])
    assert result["rmse_improvement"] == pytest.approx(0.0)
