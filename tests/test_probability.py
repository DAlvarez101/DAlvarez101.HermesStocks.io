"""Tests for bucket probability engine."""
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from dfw_temp_model.blending.probability import (
    daily_high_distribution,
    compute_bucket_probabilities,
)


def test_daily_high_distribution_basic():
    """Extract the daily high from a blended forecast and model it as Gaussian."""
    forecast = pd.DataFrame({
        "valid_dt": pd.to_datetime([
            "2026-06-20T05:00:00+00:00",  # midnight CT
            "2026-06-20T10:00:00+00:00",  # 5am CT
            "2026-06-20T17:00:00+00:00",  # noon CT
            "2026-06-20T20:00:00+00:00",  # 3pm CT  <- daily high
            "2026-06-20T23:00:00+00:00",  # 6pm CT
            "2026-06-21T03:00:00+00:00",  # 10pm CT
            "2026-06-21T05:00:00+00:00",  # midnight CT
        ], utc=True),
        "forecast_hour": [1, 6, 13, 16, 19, 23, 25],
        "tmpf_blended": [72.0, 75.0, 88.0, 95.0, 93.0, 85.0, 78.0],
        "tmpf_near_resolution": [72.0, 75.0, 88.0, 95.0, 93.0, 85.0, 78.0],
    })
    result = daily_high_distribution(
        forecast,
        target_date="2026-06-20",
        timezone="America/Chicago",
    )
    assert result is not None
    assert result["daily_high_forecast"] == 95.0
    assert result["mu"] == 95.0
    assert result["sigma"] > 0


def test_daily_high_distribution_multi_model_constrains_sigma():
    """When multiple models have data at the high hour, sigma should be
    constrained by model agreement, not just the horizon cap."""
    forecast = pd.DataFrame({
        "valid_dt": pd.to_datetime([
            "2026-06-20T05:00:00+00:00",  # midnight CT
            "2026-06-20T20:00:00+00:00",  # 3pm CT <- daily high
            "2026-06-20T23:00:00+00:00",  # 6pm CT
        ], utc=True),
        "forecast_hour": [1, 16, 19],
        "tmpf_blended": [72.0, 95.0, 93.0],
        "tmpf_near_resolution": [72.0, 95.0, 93.0],
        "model_spread": [1.0, 1.5, 2.0],
        "tmpf_hrrr": [72.0, 95.0, 93.0],
        "tmpf_nam": [71.5, 94.5, 92.5],
    })
    result = daily_high_distribution(
        forecast,
        target_date="2026-06-20",
        timezone="America/Chicago",
    )
    assert result is not None
    # Without n_models: effective_sigma(16, spread=1.5) = sqrt(3.2^2 + 0.75^2) = 3.29
    # With n_models=2: spread_ceiling = max(0.75, 2.0) = 2.0
    # sigma = min(3.29, 2.0) = 2.0
    assert result["sigma"] < 3.2, f"Sigma should be constrained by model agreement, got {result['sigma']}"
    assert result["sigma"] >= 2.0 - 1e-10


def test_compute_bucket_probabilities_basic():
    """Compute P(bucket) for a set of 2F buckets."""
    result = compute_bucket_probabilities(mu=95.0, sigma=2.0, bucket_width=2.0)
    assert isinstance(result, list)
    assert len(result) > 0
    total = sum(b["probability"] for b in result)
    assert 0.99 < total < 1.01


def test_bucket_probabilities_peak_near_mu():
    """The highest probability bucket should be near mu."""
    result = compute_bucket_probabilities(mu=95.0, sigma=2.0, bucket_width=2.0)
    max_bucket = max(result, key=lambda b: b["probability"])
    assert abs(max_bucket["lower"] - 95.0) <= 2.0