"""Tests for the rolling bias corrector."""
import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.blending.bias import (
    compute_rolling_bias,
    apply_bias_correction,
    compute_trend_correction,
)


def test_compute_rolling_bias_basic():
    """Bias = observed - forecast, returned as a rolling exponentially-weighted mean."""
    obs = pd.DataFrame({
        "valid_hour": pd.to_datetime([
            "2026-06-17T18:00:00Z",
            "2026-06-17T19:00:00Z",
            "2026-06-17T20:00:00Z",
        ], utc=True),
        "tmpf_obs": [89.0, 90.0, 91.0],
    })
    fcst = pd.DataFrame({
        "valid_hour": pd.to_datetime([
            "2026-06-17T18:00:00Z",
            "2026-06-17T19:00:00Z",
            "2026-06-17T20:00:00Z",
        ], utc=True),
        "tmpf_fcst": [88.0, 89.0, 90.0],
    })
    result = compute_rolling_bias(obs, fcst, halflife_hours=6.0)
    assert "bias" in result.columns
    assert "n_matches" in result.columns
    assert result["bias"].iloc[-1] == pytest.approx(1.0, abs=0.01)
    assert result["n_matches"].iloc[-1] == 3


def test_compute_rolling_bias_empty():
    """No overlaps -> empty result with correct columns."""
    obs = pd.DataFrame({"valid_hour": pd.to_datetime([], utc=True), "tmpf_obs": []})
    fcst = pd.DataFrame({"valid_hour": pd.to_datetime([], utc=True), "tmpf_fcst": []})
    result = compute_rolling_bias(obs, fcst, halflife_hours=6.0)
    assert result.empty
    assert "bias" in result.columns
    assert "bias_std" in result.columns


def test_apply_bias_correction():
    """Corrected = raw forecast + rolling bias."""
    forecast = pd.DataFrame({
        "valid_dt": pd.to_datetime([
            "2026-06-17T21:00:00Z",
            "2026-06-17T22:00:00Z",
            "2026-06-17T23:00:00Z",
        ], utc=True),
        "tmpf": [88.0, 87.0, 86.0],
        "forecast_hour": [1, 2, 3],
    })
    bias_df = pd.DataFrame({
        "valid_hour": [pd.Timestamp("2026-06-17T20:00:00Z", tz="UTC")],
        "bias": [1.5],
        "bias_std": [0.5],
        "n_matches": [5],
    })
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=1.0)
    assert "tmpf_corrected" in result.columns
    assert "uncertainty_low" in result.columns
    assert "uncertainty_high" in result.columns
    assert result["tmpf_corrected"].iloc[0] == pytest.approx(89.5, abs=0.01)
    assert result["uncertainty_low"].iloc[0] == pytest.approx(89.0, abs=0.01)
    assert result["uncertainty_high"].iloc[0] == pytest.approx(90.0, abs=0.01)


def test_apply_bias_correction_no_bias():
    """If no bias data, corrected = raw, uncertainty from raw spread."""
    forecast = pd.DataFrame({
        "valid_dt": pd.to_datetime(["2026-06-17T21:00:00Z"], utc=True),
        "tmpf": [88.0],
        "forecast_hour": [1],
    })
    bias_df = pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=1.0)
    assert result["tmpf_corrected"].iloc[0] == pytest.approx(88.0, abs=0.01)
    assert not np.isnan(result["uncertainty_low"].iloc[0])


def test_compute_trend_correction_warming():
    """Trend correction is positive when newer cycles are warmer."""
    cycles_df = pd.DataFrame({
        "valid_dt": ["2026-06-17T10:00:00Z"] * 3,
        "init_dt": [
            "2026-06-17T04:00:00Z",
            "2026-06-17T06:00:00Z",
            "2026-06-17T08:00:00Z",
        ],
        "tmpf": [78.0, 79.0, 80.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert "valid_dt" in result.columns
    assert "trend_correction" in result.columns
    # Slope is -0.5 deg F per hour of age (warming as age decreases).
    # correction = -slope * 0.15 = 0.5 * 0.15 = 0.075
    assert result["trend_correction"].iloc[0] > 0
    assert result["n_cycles"].iloc[0] == 3


def test_compute_trend_correction_cooling():
    """Trend correction is negative when newer cycles are cooler."""
    cycles_df = pd.DataFrame({
        "valid_dt": ["2026-06-17T10:00:00Z"] * 3,
        "init_dt": [
            "2026-06-17T04:00:00Z",
            "2026-06-17T06:00:00Z",
            "2026-06-17T08:00:00Z",
        ],
        "tmpf": [82.0, 81.0, 80.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert result["trend_correction"].iloc[0] < 0


def test_compute_trend_correction_single_cycle():
    """Only one cycle available -> zero trend correction."""
    cycles_df = pd.DataFrame({
        "valid_dt": ["2026-06-17T10:00:00Z"],
        "init_dt": ["2026-06-17T08:00:00Z"],
        "tmpf": [80.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert result["trend_correction"].iloc[0] == pytest.approx(0.0, abs=0.01)


def test_compute_trend_correction_multiple_valid_hours():
    """Trend correction computed independently for each valid hour."""
    cycles_df = pd.DataFrame({
        "valid_dt": [
            "2026-06-17T10:00:00Z", "2026-06-17T10:00:00Z", "2026-06-17T10:00:00Z",
            "2026-06-17T11:00:00Z", "2026-06-17T11:00:00Z", "2026-06-17T11:00:00Z",
        ],
        "init_dt": [
            "2026-06-17T04:00:00Z", "2026-06-17T06:00:00Z", "2026-06-17T08:00:00Z",
            "2026-06-17T04:00:00Z", "2026-06-17T06:00:00Z", "2026-06-17T08:00:00Z",
        ],
        "tmpf": [78.0, 79.0, 80.0, 76.0, 75.0, 74.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert len(result) == 2
    h10 = result[result["valid_dt"] == "2026-06-17T10:00:00+00:00"]
    assert h10["trend_correction"].iloc[0] > 0
    h11 = result[result["valid_dt"] == "2026-06-17T11:00:00+00:00"]
    assert h11["trend_correction"].iloc[0] < 0