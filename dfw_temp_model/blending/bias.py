"""Rolling bias correction for model-agnostic forecast blending.

Computes the exponentially-weighted rolling mean of (observed - forecast)
at each valid hour, then applies that bias as an additive correction to
future forecast hours. The bias is provider-specific: HRRR has its own
bias, GFS would have its own, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _floor_to_hour(ts: pd.Series) -> pd.Series:
    """Floor a datetime Series to the top of the hour."""
    return pd.to_datetime(ts, utc=True).dt.floor("h")


def compute_rolling_bias(
    obs_df: pd.DataFrame,
    fcst_df: pd.DataFrame,
    halflife_hours: float = 6.0,
) -> pd.DataFrame:
    """Compute the rolling bias (observed - forecast) from matched hours.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Must have columns ``valid_hour`` (datetime, UTC) and ``tmpf_obs`` (float).
    fcst_df : pd.DataFrame
        Must have columns ``valid_hour`` (datetime, UTC) and ``tmpf_fcst`` (float).
    halflife_hours : float
        Half-life of the exponential decay in hours. Recent observations
        weigh more. Default 6 hours means an observation 6 hours old has
        half the weight of the current one.

    Returns
    -------
    pd.DataFrame
        Columns: ``valid_hour`` (datetime UTC), ``bias`` (float, the EWMA of
        obs - fcst), ``bias_std`` (float, rolling std for uncertainty),
        ``n_matches`` (int, cumulative count of matched hours).
    """
    if obs_df.empty or fcst_df.empty:
        return pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])

    obs = obs_df.copy()
    fcst = fcst_df.copy()
    obs["valid_hour"] = _floor_to_hour(obs["valid_hour"])
    fcst["valid_hour"] = _floor_to_hour(fcst["valid_hour"])

    # Merge on valid_hour (many-to-one if multiple cycles match the same obs hour)
    merged = obs.merge(fcst, on="valid_hour", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])

    # If multiple forecast cycles match the same obs hour, take the mean.
    merged["error"] = merged["tmpf_obs"] - merged["tmpf_fcst"]
    hourly = merged.groupby("valid_hour").agg(
        error_mean=("error", "mean"),
        error_std=("error", "std"),
        n=("error", "count"),
    ).reset_index()
    hourly = hourly.sort_values("valid_hour")

    # Exponentially-weighted moving average of the bias.
    # span = 2 * halflife (in number of samples, assuming ~1-hour spacing)
    span = max(1, int(2 * halflife_hours))

    hourly["bias"] = hourly["error_mean"].ewm(
        span=span, adjust=False, min_periods=1
    ).mean()
    # Rolling std (expanding, with at least 2 samples)
    hourly["bias_std"] = hourly["error_std"].fillna(0.0)
    # If only 1 sample, use a default uncertainty of 1.0 deg F
    hourly.loc[hourly["n"] == 1, "bias_std"] = 1.0
    hourly["n_matches"] = hourly["n"].cumsum()

    return hourly[["valid_hour", "bias", "bias_std", "n_matches"]]


def apply_bias_correction(
    forecast: pd.DataFrame,
    bias_df: pd.DataFrame,
    uncertainty_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Apply the latest rolling bias to a forecast and add uncertainty bands.

    Parameters
    ----------
    forecast : pd.DataFrame
        Must have ``valid_dt`` (datetime UTC) and ``tmpf`` (float).
    bias_df : pd.DataFrame
        Output of ``compute_rolling_bias``. The *latest* bias value is used
        as a constant correction for all future forecast hours.
    uncertainty_multiplier : float
        Multiplier for the bias_std to form the uncertainty band. 1.0 = +/- 1 sigma.

    Returns
    -------
    pd.DataFrame
        Copy of ``forecast`` with added columns: ``tmpf_corrected``,
        ``uncertainty_low``, ``uncertainty_high``, ``bias_applied``.
    """
    result = forecast.copy()
    result["valid_dt"] = pd.to_datetime(result["valid_dt"], utc=True)

    if bias_df.empty:
        # No bias data: corrected = raw, default uncertainty
        result["tmpf_corrected"] = result["tmpf"]
        result["bias_applied"] = 0.0
        default_unc = 2.0  # 2 deg F default when we have no bias estimate
        result["uncertainty_low"] = result["tmpf_corrected"] - default_unc
        result["uncertainty_high"] = result["tmpf_corrected"] + default_unc
        return result

    # Use the latest bias value as a constant for all future hours.
    latest = bias_df.iloc[-1]
    bias = float(latest["bias"])
    bias_std = float(latest["bias_std"]) * uncertainty_multiplier

    result["tmpf_corrected"] = result["tmpf"] + bias
    result["bias_applied"] = bias
    result["uncertainty_low"] = result["tmpf_corrected"] - bias_std
    result["uncertainty_high"] = result["tmpf_corrected"] + bias_std
    return result