"""Convert blended temperature forecasts into Polymarket-style bucket probabilities.

The pipeline:
1. Extract the daily high from the blended forecast for the target climate day.
2. Model the daily high as a Gaussian distribution N(mu, sigma).
3. Compute P(bucket) = CDF(upper) - CDF(lower) for each temperature bucket.

This is the foundation for comparing our model probabilities against
Polymarket market prices to find mispriced buckets.
"""
from __future__ import annotations

import sqlite3
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from scipy.stats import norm

from dfw_temp_model.blending.multi import multi_model_forecast
from dfw_temp_model.blending.near_resolution import near_resolution_adjust
from dfw_temp_model.blending.sigma import (
    sigma_for_forecast_hour,
    effective_sigma,
    shrink_sigma_for_observations,
)
from dfw_temp_model.blending.trend import observed_temperature_trend


def daily_high_distribution(
    forecast: pd.DataFrame,
    target_date: str,
    timezone: str = "America/Chicago",
) -> Optional[dict]:
    """Extract the daily high from a blended forecast and model it as Gaussian.

    Parameters
    ----------
    forecast : pd.DataFrame
        Must have ``valid_dt`` (datetime UTC) and ``tmpf_blended`` (float).
        If ``tmpf_near_resolution`` exists, it is used instead of
        ``tmpf_blended`` for the daily high (near-resolution is closer to
        reality in the final hours).
    target_date : str
        The climate day in ``YYYY-MM-DD`` format. This is the LOCAL date
        (Pacific Time for KLAX), not the UTC date.
    timezone : str
        Timezone name for the climate day boundary (default Pacific Time).

    Returns
    -------
    dict or None
        ``{"mu": float, "sigma": float, "daily_high_forecast": float,
          "high_forecast_hour": float, "n_hours_in_day": int}``
        or None if no forecast hours fall within the climate day.
    """
    if forecast.empty:
        return None

    tz = ZoneInfo(timezone)
    target_day = pd.Timestamp(target_date, tz=tz)
    day_start_utc = target_day.astimezone(ZoneInfo("UTC"))
    day_end_utc = (target_day + pd.Timedelta(days=1)).astimezone(ZoneInfo("UTC"))

    df = forecast.copy()
    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)

    # Filter to climate day: [day_start_utc, day_end_utc)
    mask = (df["valid_dt"] >= day_start_utc) & (df["valid_dt"] < day_end_utc)
    day_rows = df[mask].copy()
    if day_rows.empty:
        return None

    # Use near_resolution if available, else blended
    if "tmpf_near_resolution" in day_rows.columns:
        temp_col = "tmpf_near_resolution"
    else:
        temp_col = "tmpf_blended"

    # Find the daily high (max temperature in the climate day)
    high_idx = day_rows[temp_col].idxmax()
    daily_high = float(day_rows.loc[high_idx, temp_col])
    high_fhr = float(day_rows.loc[high_idx, "forecast_hour"])
    high_valid_dt = day_rows.loc[high_idx, "valid_dt"]

    # Sigma from the forecast hour of the daily high.
    # If model_spread is available at the high hour, incorporate it.
    # When multiple models have data (n_models >= 2), their agreement
    # constrains sigma — preventing the horizon cap from inflating
    # uncertainty when models are tightly clustered.
    if "model_spread" in day_rows.columns:
        spread_at_high = float(day_rows.loc[high_idx, "model_spread"])
        # Count models with non-null data at the high hour
        model_cols = [c for c in day_rows.columns if c.startswith("tmpf_") and c != "tmpf_blended" and c != "tmpf_near_resolution"]
        n_models_at_high = sum(
            1 for c in model_cols
            if pd.notna(day_rows.loc[high_idx, c])
        )
        sigma = effective_sigma(
            high_fhr, model_spread=spread_at_high, n_models=n_models_at_high
        )
    else:
        sigma = sigma_for_forecast_hour(high_fhr)

    # Coverage: how many hours of the 24-hour climate day have forecast data
    # (24 hours = full coverage; fewer = the high might be missed)
    n_hours_in_day = len(day_rows)
    coverage_pct = (n_hours_in_day / 24.0) * 100.0

    # Identify which temperature source was used
    temp_source = "near-resolution" if temp_col == "tmpf_near_resolution" else "blended"

    return {
        "mu": daily_high,
        "sigma": sigma,
        "daily_high_forecast": daily_high,
        "high_forecast_hour": high_fhr,
        "high_valid_dt": high_valid_dt,
        "n_hours_in_day": n_hours_in_day,
        "coverage_pct": coverage_pct,
        "temp_source": temp_source,
    }


def _truncated_bucket_prob(
    lower: float,
    upper: float,
    mu: float,
    sigma: float,
    observed_high: float,
) -> float:
    """P(H in [lower, upper) | H >= observed_high) for H ~ N(mu, sigma).

    Uses the left-truncated Gaussian: the daily high is monotonically
    non-decreasing, so once we've observed a max of *observed_high*,
    every outcome below it is impossible.  The remaining probability
    is renormalised by the survival function 1 - CDF(observed_high).

    Three cases:
      1. upper <= observed_high  ->  bucket is dead, P = 0
      2. lower < observed_high < upper  ->  partial bucket,
         P = [CDF(upper) - CDF(observed_high)] / S
      3. lower >= observed_high  ->  fully alive,
         P = [CDF(upper) - CDF(lower)] / S

    where S = 1 - CDF(observed_high) is the truncation normaliser.
    """
    if sigma <= 0:
        sigma = 0.1
    cdf_obs = norm.cdf(observed_high, loc=mu, scale=sigma)
    survival = 1.0 - cdf_obs
    if survival <= 1e-10:
        # Observed high is so far above mu that there's essentially no
        # probability mass left — everything is already resolved.
        # Return 0 for all buckets; the caller should handle this.
        return 0.0

    l_eff = max(lower, observed_high)
    if l_eff >= upper:
        return 0.0
    return float((norm.cdf(upper, loc=mu, scale=sigma) -
                  norm.cdf(l_eff, loc=mu, scale=sigma)) / survival)


def compute_bucket_probabilities(
    mu: float,
    sigma: float,
    bucket_width: float = 2.0,
    min_temp: float = 50.0,
    max_temp: float = 100.0,
    bucket_offset: float = 0.0,
    observed_high: float | None = None,
) -> list[dict]:
    """Compute the probability of the daily high falling in each temperature bucket.

    Buckets are aligned on multiples of *bucket_width*, shifted by
    *bucket_offset*. Edge buckets use sentinel values (-999, 999) to
    represent "below X" and "Y or above".

    Parameters
    ----------
    mu : float
        Mean of the daily high distribution (bias-corrected forecast high).
    sigma : float
        Standard deviation of the daily high distribution.
    bucket_width : float
        Width of each temperature bucket in degrees F (default 2.0).
    min_temp : float
        Lower bound for the bucket range. Below this, an edge bucket
        "below min_temp" is created (default 50°F).
    max_temp : float
        Upper bound for the bucket range. Above this, an edge bucket
        "max_temp or above" is created (default 100°F).
    bucket_offset : float
        Shift applied to the alignment grid (default 0.0). For example,
        with ``bucket_width=2`` and ``bucket_offset=1``, buckets start
        at odd integers: [55,57), [57,59), [59,61), ... labelled "55-56°F",
        "57-58°F", etc. This matches the Polymarket convention. With
        ``bucket_offset=0`` (default), buckets are even-aligned:
        [56,58), [58,60), ... labelled "56-57°F", "58-59°F", etc.
    observed_high : float, optional
        If provided, the maximum temperature observed **so far** in the
        climate day.  Because the daily high is monotonically
        non-decreasing, any bucket whose upper bound is at or below this
        value is impossible and receives zero probability.  The remaining
        probabilities are renormalised via a left-truncated Gaussian
        (dividing by the survival function 1 - CDF(observed_high)).
        Pass ``None`` (default) for future days or when no live
        observations are available.

    Returns
    -------
    list[dict]
        Each dict has ``lower``, ``upper``, ``label``, ``probability``.
        Probabilities sum to ~1.0.

    Notes
    -----
    The *label* shows the consecutive integer values the bucket contains
    (e.g. "67-68°F" for a 2°F bucket [67, 69)), matching the Polymarket
    convention. For 1°F buckets the label is just the single integer.
    The mathematical boundaries (``lower``, ``upper``) remain the actual
    CDF integration limits.
    """
    if sigma <= 0:
        sigma = 0.1  # avoid division by zero

    # Snap min_temp to the offset-aligned grid so interior buckets are
    # consistent regardless of the caller's exact min_temp.
    # e.g. bucket_width=2, offset=1 -> grid points at 51,53,55,...
    #      bucket_width=2, offset=0 -> grid points at 50,52,54,...
    grid_start = min_temp
    if bucket_width > 0:
        n_steps = int((min_temp - bucket_offset) // bucket_width)
        grid_start = bucket_offset + n_steps * bucket_width
        if grid_start < min_temp:
            grid_start += bucket_width

    # Determine the truncation point (if any).
    truncate_at = observed_high if observed_high is not None else None

    buckets = []

    # Edge bucket: "below grid_start"
    if truncate_at is not None and truncate_at >= grid_start:
        p_below = 0.0
    elif truncate_at is not None:
        # Partial: some of the "below" bucket is still alive
        p_below = _truncated_bucket_prob(
            -1e9, grid_start, mu, sigma, truncate_at,
        )
    else:
        p_below = float(norm.cdf(grid_start, loc=mu, scale=sigma))
    buckets.append({
        "lower": -999,
        "upper": grid_start,
        "label": f"{grid_start - 1:.0f}°F or below",
        "probability": p_below,
    })

    # Interior buckets
    lower = grid_start
    while lower < max_temp:
        upper = lower + bucket_width
        if truncate_at is not None:
            p = _truncated_bucket_prob(lower, upper, mu, sigma, truncate_at)
        else:
            p = float(norm.cdf(upper, loc=mu, scale=sigma) -
                      norm.cdf(lower, loc=mu, scale=sigma))
        if bucket_width == 1.0:
            label = f"{lower:.0f}°F"
        else:
            # Show consecutive integers: [67,69) -> "67-68°F"
            label = f"{lower:.0f}-{upper - 1:.0f}°F"
        buckets.append({
            "lower": lower,
            "upper": upper,
            "label": label,
            "probability": max(p, 0.0),  # clamp to non-negative
        })
        lower = upper

    # Edge bucket: "max_temp or above" (snap max_temp to grid too)
    grid_end = lower  # last upper boundary reached by the while loop
    # If the loop overshot max_temp, the last bucket still covers up to grid_end
    if truncate_at is not None:
        p_above = _truncated_bucket_prob(
            grid_end, 1e9, mu, sigma, truncate_at,
        )
    else:
        p_above = float(1.0 - norm.cdf(grid_end, loc=mu, scale=sigma))
    buckets.append({
        "lower": grid_end,
        "upper": 999,
        "label": f"{grid_end:.0f}°F or above",
        "probability": max(p_above, 0.0),
    })

    return buckets


def forecast_bucket_probabilities(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    timezone: str = "America/Chicago",
    bucket_width: float = 2.0,
    min_temp: float = 50.0,
    max_temp: float = 100.0,
    bucket_offset: float = 0.0,
    init_dt: str | None = None,
) -> dict | None:
    """Full pipeline: DB -> blended forecast -> daily high -> bucket probabilities.

    When live observations exist for the target climate day, the running
    maximum observed temperature is used to **left-truncate** the Gaussian
    distribution: buckets below the observed high are impossible (the
    daily high is monotonically non-decreasing) and receive zero
    probability.  The remaining probabilities are renormalised via the
    truncated Gaussian survival function.  This corrects the systematic
    cool-bias / over-dispersion that occurs when the model allocates
    probability to outcomes that have already been invalidated.

    Parameters
    ----------
    conn : sqlite3.Connection
    station : str
        ICAO code.
    target_date : str
        Climate day in ``YYYY-MM-DD`` format (LOCAL date, not UTC).
    timezone : str
        Timezone for the climate day boundary (default Pacific Time).
    bucket_width : float
        Width of each temperature bucket in degrees F (default 2.0).
    min_temp, max_temp : float
        Edge bucket boundaries (default 50-100°F).
    bucket_offset : float
        Shift applied to the alignment grid (default 0.0). Use 1.0 for
        odd-aligned buckets (matching Polymarket convention).
    init_dt : str, optional
        Specific model cycle to use. If None, uses the latest available.

    Returns
    -------
    dict or None
        ``{"buckets": list[dict], "mu": float, "sigma": float,
           "daily_high_forecast": float, "high_forecast_hour": float,
           "n_hours_in_day": int, "observed_high": float or None}``
        or None if no forecast data is available for the target date.
    """
    # Get HRRR forecast (bias-corrected; ECMWF removed 2026-06-23)
    blended = multi_model_forecast(conn, station, init_dt=init_dt)
    if blended.empty:
        return None

    # Load observations for near-resolution adjustment
    obs_df = pd.read_sql_query(
        "SELECT valid, tmpf, source FROM metar_observations "
        "WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn, params=[station],
    )
    if not obs_df.empty:
        obs_df["valid"] = pd.to_datetime(obs_df["valid"], utc=True)

    # Apply near-resolution adjustment
    blended = near_resolution_adjust(blended, obs_df, station)

    # Override past hours with actual observations (ground truth).
    # Only when using the latest cycle (init_dt=None) — for historical
    # cycle lookbacks, we want to see what the model predicted, not what
    # actually happened.
    if init_dt is None and not obs_df.empty and not blended.empty:
        blended = _override_past_hours_with_obs(blended, obs_df, target_date, timezone)

    # Extract daily high distribution
    dist = daily_high_distribution(blended, target_date, timezone)
    if dist is None:
        return None

    # Shrink sigma based on how much time has elapsed in the climate day.
    # More observations (later latest-obs time) = less remaining uncertainty.
    # Only apply for the latest cycle (init_dt=None) — historical lookbacks
    # should show the model's raw uncertainty, not conditioned on obs.
    hours_elapsed = _hours_elapsed_in_climate_day(conn, station, target_date, timezone)
    sigma_before_shrink = dist["sigma"]
    if init_dt is None:
        sigma_shrunk = shrink_sigma_for_observations(
            sigma_before_shrink, hours_elapsed,
        )
    else:
        sigma_shrunk = sigma_before_shrink

    # Get the running max observed temperature for this climate day.
    # If observations exist, we left-truncate the Gaussian at this value
    # because the daily high is monotonically non-decreasing — buckets
    # below the observed max are already impossible.
    # Only apply truncation for the latest cycle (init_dt=None) — historical
    # lookbacks should show what the model predicted, not conditioned on what
    # actually happened.
    if init_dt is None:
        observed_high = observed_high_so_far(conn, station, target_date, timezone)
    else:
        observed_high = None

    # Compute the forecast temperature floor for the climate day.
    # The daily high is the MAX of all hourly temperatures in the climate day,
    # so it is >= every individual hour's temperature. The first hour (12am)
    # is a valid mathematical floor: the daily high cannot be below the 12am
    # temperature. When live observations exist, observed_high (running max of
    # actual obs) takes precedence. When no observations exist yet (future
    # days or early in the current day), the 12am forecast temperature serves
    # as a proxy floor, preventing the Gaussian from allocating probability to
    # buckets below the forecast 12am temp.
    # Skip for historical lookbacks (init_dt is not None) — same rationale as
    # observed_high: show what the model predicted, not conditioned on reality.
    if init_dt is None:
        forecast_floor = _forecast_floor_for_climate_day(blended, target_date, timezone)
    else:
        forecast_floor = None
    # Use the higher of observed_high and forecast_floor as the truncation point.
    # If both are None, no truncation is applied.
    truncation_point = None
    if observed_high is not None and forecast_floor is not None:
        truncation_point = max(observed_high, forecast_floor)
    elif observed_high is not None:
        truncation_point = observed_high
    elif forecast_floor is not None:
        truncation_point = forecast_floor

    # Determine if the target date is today (the in-progress climate day).
    # Trend shift should only be applied when the target date matches today
    # and observations exist within the target climate day. For future days,
    # the trend from today's observations has no physical relevance.
    tz = ZoneInfo(timezone)
    now_pt = pd.Timestamp.now(tz=tz).normalize()
    target_day_ts = pd.Timestamp(target_date, tz=tz).normalize()
    is_today = (now_pt == target_day_ts)

    # Observed temperature trend (rate-of-change inertia) correction.
    # If the temperature is rising/cooling faster than the model expects,
    # apply a conservative shift to mu.  Only when live obs exist AND
    # the target date is today (in-progress climate day with observations).
    # Uses OLS regression over a 3h window so 5-minute obs give a robust slope.
    # The trend is extrapolated over a FIXED 3h horizon (not to the model's
    # predicted high time) to avoid circular reasoning — see trend.py docstring.
    trend_shift = 0.0
    trend_F_per_hour = 0.0
    if not obs_df.empty and is_today and init_dt is None:
        trend_info = observed_temperature_trend(
            obs_df,
            trend_window_hours=3.0,
            extrapolation_hours=3.0,
            trend_weight=0.15,
            max_shift=3.0,
        )
        if trend_info is not None:
            trend_shift = trend_info["trend_shift"]
            trend_F_per_hour = trend_info["trend_F_per_hour"]

    mu_adjusted = dist["mu"] + trend_shift

    # Compute bucket probabilities (with truncation if truncation_point is not None)
    buckets = compute_bucket_probabilities(
        mu=mu_adjusted,
        sigma=sigma_shrunk,
        bucket_width=bucket_width,
        min_temp=min_temp,
        max_temp=max_temp,
        bucket_offset=bucket_offset,
        observed_high=truncation_point,
    )

    return {
        "buckets": buckets,
        "mu": mu_adjusted,
        "mu_before_trend": dist["mu"],
        "sigma": sigma_shrunk,
        "sigma_before_shrink": sigma_before_shrink,
        "hours_elapsed": hours_elapsed,
        "trend_shift": trend_shift,
        "trend_F_per_hour": trend_F_per_hour,
        "daily_high_forecast": dist["daily_high_forecast"],
        "high_forecast_hour": dist["high_forecast_hour"],
        "high_valid_dt": dist["high_valid_dt"],
        "n_hours_in_day": dist["n_hours_in_day"],
        "coverage_pct": dist["coverage_pct"],
        "temp_source": dist["temp_source"],
        "observed_high": observed_high,
        "forecast_floor": forecast_floor,
        "truncation_point": truncation_point,
    }


def _forecast_floor_for_climate_day(
    blended: pd.DataFrame,
    target_date: str,
    timezone: str = "America/Chicago",
) -> float | None:
    """Return the 12am (midnight) forecast temperature for a climate day.

    The daily high is the maximum of all hourly temperatures in the climate
    day, so it is mathematically >= the 12am temperature. This makes the 12am
    forecast temp a valid floor for the daily high distribution, even before
    any live observations are available. When the 12am hour has already been
    observed, ``observed_high_so_far`` takes precedence (it will be >= the
    12am temp because it's a running max).

    Uses the near-resolution adjusted temperature if available (closest to
    reality), otherwise the blended forecast.

    Returns None if no forecast data covers the 12am hour of the target day.
    """
    if blended.empty:
        return None

    tz = ZoneInfo(timezone)
    target_day = pd.Timestamp(target_date, tz=tz)
    day_start_utc = target_day.astimezone(ZoneInfo("UTC"))

    df = blended.copy()
    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)

    # Find the forecast row closest to 12am (climate day start)
    mask = (df["valid_dt"] >= day_start_utc) & (df["valid_dt"] < day_start_utc + pd.Timedelta(hours=1))
    midnight_rows = df[mask]
    if midnight_rows.empty:
        # If no exact 12am row, use the closest row within the first 2 hours
        mask = (df["valid_dt"] >= day_start_utc) & (df["valid_dt"] < day_start_utc + pd.Timedelta(hours=2))
        midnight_rows = df[mask]
    if midnight_rows.empty:
        return None

    # Use near_resolution if available, else blended
    if "tmpf_near_resolution" in midnight_rows.columns:
        temp_col = "tmpf_near_resolution"
    else:
        temp_col = "tmpf_blended"

    return float(midnight_rows.iloc[0][temp_col])


def observed_high_so_far(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    timezone: str = "America/Chicago",
) -> float | None:
    """Return the maximum observed temperature **so far** in a climate day.

    Unlike :func:`actual_daily_high`, which is only meaningful after the
    day is complete, this function is designed for **in-progress** days.
    It returns the running max of all observations from the start of the
    climate day up to the latest available observation.

    For a future date with no observations, returns ``None``.
    For a completed day, returns the same value as ``actual_daily_high``.
    """
    tz = ZoneInfo(timezone)
    target_day = pd.Timestamp(target_date, tz=tz)
    day_start_utc = target_day.astimezone(ZoneInfo("UTC"))
    day_end_utc = (target_day + pd.Timedelta(days=1)).astimezone(ZoneInfo("UTC"))

    df = pd.read_sql_query(
        "SELECT valid, tmpf FROM metar_observations "
        "WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn, params=[station],
    )
    if df.empty:
        return None

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    mask = (df["valid"] >= day_start_utc) & (df["valid"] < day_end_utc)
    day_obs = df[mask]
    if day_obs.empty:
        return None

    return float(day_obs["tmpf"].max())


def _hours_elapsed_in_climate_day(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    timezone: str = "America/Chicago",
) -> float:
    """Return hours elapsed from climate-day start to the latest observation.

    Uses the timestamp of the most recent observation within the climate
    day, relative to the day's start (local midnight).  Returns a float
    to support 5-minute observation granularity.  Capped at 24.0.
    Returns 0.0 if no observations exist for the day.
    """
    tz = ZoneInfo(timezone)
    target_day = pd.Timestamp(target_date, tz=tz)
    day_start_utc = target_day.astimezone(ZoneInfo("UTC"))
    day_end_utc = (target_day + pd.Timedelta(days=1)).astimezone(ZoneInfo("UTC"))

    row = conn.execute(
        "SELECT MAX(valid) FROM metar_observations "
        "WHERE station = ? AND tmpf IS NOT NULL "
        "AND valid >= ? AND valid < ?",
        (station, day_start_utc.isoformat(), day_end_utc.isoformat()),
    ).fetchone()

    if row is None or row[0] is None:
        return 0.0

    latest = pd.to_datetime(row[0], utc=True)
    elapsed = (latest - day_start_utc).total_seconds() / 3600.0
    return max(0.0, min(24.0, elapsed))


def _override_past_hours_with_obs(
    blended: pd.DataFrame,
    obs_df: pd.DataFrame,
    target_date: str,
    timezone: str = "America/Chicago",
) -> pd.DataFrame:
    """Replace forecast temperatures with observations for hours that passed.

    For every forecast row whose valid time is at or before the latest
    observation, replace ``tmpf_blended`` and ``tmpf_near_resolution``
    with the maximum observed temperature in that hour.  This makes the
    daily high distribution reflect ground truth for settled hours,
    preventing stale model forecasts from inflating the predicted high.

    Also INSERTS missing past hours from observations — if the latest
    HRRR cycle starts at 3 PM PT but the climate day started at midnight,
    the hours from midnight to 3 PM are added as observation rows so
    the daily high distribution has full coverage.

    Only overrides hours within the target climate day.  Uses hourly
    max (not instantaneous obs) to match the forecast granularity and
    avoid noise from individual METAR spikes.
    """
    tz = ZoneInfo(timezone)
    target_day = pd.Timestamp(target_date, tz=tz)
    day_start_utc = target_day.astimezone(ZoneInfo("UTC"))
    day_end_utc = (target_day + pd.Timedelta(days=1)).astimezone(ZoneInfo("UTC"))

    result = blended.copy()
    result["valid_dt"] = pd.to_datetime(result["valid_dt"], utc=True)

    obs = obs_df.copy()
    obs["valid"] = pd.to_datetime(obs["valid"], utc=True)
    obs = obs.dropna(subset=["tmpf"])

    # Only use observations within the climate day
    day_obs = obs[(obs["valid"] >= day_start_utc) & (obs["valid"] < day_end_utc)]
    if day_obs.empty:
        return result

    # Floor both to the hour for matching
    day_obs = day_obs.copy()
    day_obs["valid_hour"] = day_obs["valid"].dt.floor("h")
    hourly_max = day_obs.groupby("valid_hour")["tmpf"].max()

    latest_obs_time = day_obs["valid"].max()

    # Override existing forecast rows with observations for past hours
    for idx, row in result.iterrows():
        vdt = row["valid_dt"]
        if vdt < day_start_utc or vdt >= day_end_utc:
            continue
        if vdt > latest_obs_time:
            continue

        vdt_hour = vdt.floor("h")
        if vdt_hour in hourly_max.index:
            obs_temp = float(hourly_max[vdt_hour])
            result.at[idx, "tmpf_blended"] = obs_temp
            result.at[idx, "tmpf_hrrr"] = obs_temp
            if "tmpf_near_resolution" in result.columns:
                result.at[idx, "tmpf_near_resolution"] = obs_temp
            if "model_spread" in result.columns:
                result.at[idx, "model_spread"] = 0.0

    # INSERT missing past hours from observations that have no forecast row
    existing_hours = set(result["valid_dt"].dt.floor("h"))
    missing_hours = [h for h in hourly_max.index if h not in existing_hours]
    if missing_hours:
        new_rows = []
        for h in missing_hours:
            obs_temp = float(hourly_max[h])
            new_row = {
                "valid_dt": h,
                "tmpf_blended": obs_temp,
                "tmpf_hrrr": obs_temp,
                "model_spread": 0.0,
            }
            if "tmpf_near_resolution" in result.columns:
                new_row["tmpf_near_resolution"] = obs_temp
            # Try to carry forward forecast_hour from nearest existing row
            if "forecast_hour" in result.columns:
                # Find the nearest existing row before this hour
                before_rows = result[result["valid_dt"] <= h]
                if not before_rows.empty:
                    nearest = before_rows.iloc[-1]
                    new_row["forecast_hour"] = nearest.get("forecast_hour", 0)
                else:
                    new_row["forecast_hour"] = 0
            new_rows.append(new_row)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            result = pd.concat([result, new_df], ignore_index=True)
            result = result.sort_values("valid_dt").reset_index(drop=True)

    return result


def actual_daily_high(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    timezone: str = "America/Chicago",
) -> float | None:
    """Extract the actual daily high from METAR observations for a climate day.

    Uses the same climate-day definition as ``daily_high_distribution``:
    midnight-to-midnight LOCAL time.

    Parameters
    ----------
    conn : sqlite3.Connection
    station : str
        ICAO code.
    target_date : str
        Climate day in ``YYYY-MM-DD`` format (LOCAL date).
    timezone : str
        Timezone for the climate day boundary.

    Returns
    -------
    float or None
        The maximum observed temperature in the climate day, or None
        if no observations exist for that day.
    """
    tz = ZoneInfo(timezone)
    target_day = pd.Timestamp(target_date, tz=tz)
    day_start_utc = target_day.astimezone(ZoneInfo("UTC"))
    day_end_utc = (target_day + pd.Timedelta(days=1)).astimezone(ZoneInfo("UTC"))

    df = pd.read_sql_query(
        "SELECT valid, tmpf FROM metar_observations "
        "WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn, params=[station],
    )
    if df.empty:
        return None

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    mask = (df["valid"] >= day_start_utc) & (df["valid"] < day_end_utc)
    day_obs = df[mask]
    if day_obs.empty:
        return None

    return float(day_obs["tmpf"].max())