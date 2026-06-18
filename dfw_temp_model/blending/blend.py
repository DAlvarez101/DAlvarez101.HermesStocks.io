"""Orchestrator: read DB, match METAR to forecast, compute bias, correct.

This is the top-level entry point for the blending pipeline. It is
provider-agnostic: pass in any ForecastProvider and it will read that
provider's forecasts from the DB, match against METAR observations,
compute the rolling bias, and return a corrected forecast.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from dfw_temp_model.blending.bias import (
    apply_bias_correction,
    compute_rolling_bias,
    compute_trend_correction,
)
from dfw_temp_model.blending.providers import ForecastProvider


def _load_metar_for_station(conn: sqlite3.Connection, station: str) -> pd.DataFrame:
    """Load METAR observations for a station, floored to the hour."""
    df = pd.read_sql_query(
        """
        SELECT valid, tmpf
        FROM metar_observations
        WHERE station = ?
        ORDER BY valid
        """,
        conn,
        params=[station],
    )
    if df.empty:
        return pd.DataFrame(columns=["valid_hour", "tmpf_obs"])
    df["valid_hour"] = pd.to_datetime(df["valid"], utc=True).dt.floor("h")
    # If multiple obs in the same hour, take the latest one
    df = df.sort_values("valid").groupby("valid_hour").tail(1)
    df = df.rename(columns={"tmpf": "tmpf_obs"})
    return df[["valid_hour", "tmpf_obs"]]


def _load_forecast_for_matching(
    conn: sqlite3.Connection,
    provider: ForecastProvider,
    station: str,
    cycles: list[str],
) -> pd.DataFrame:
    """Load forecast rows from multiple cycles for bias matching.

    We use all recent cycles (not just the latest) so the bias estimate
    benefits from more data points. Each cycle contributes its own
    (forecast, observation) pairs at matching valid hours.
    """
    frames = []
    for init_dt in cycles:
        df = provider.fetch_forecast(conn, station, init_dt)
        if df.empty:
            continue
        df["valid_hour"] = pd.to_datetime(df["valid_dt"], utc=True).dt.floor("h")
        df = df.rename(columns={"tmpf": "tmpf_fcst"})
        frames.append(df[["valid_hour", "tmpf_fcst"]])
    if not frames:
        return pd.DataFrame(columns=["valid_hour", "tmpf_fcst"])
    return pd.concat(frames, ignore_index=True)


def _load_all_cycles_for_trend(
    conn: sqlite3.Connection,
    provider: ForecastProvider,
    station: str,
    cycles: list[str],
) -> pd.DataFrame:
    """Load forecast rows from multiple cycles for trend computation.

    Returns a DataFrame with valid_dt, init_dt, and tmpf columns.
    Unlike _load_forecast_for_matching, this keeps init_dt so we can
    compute per-cycle trends at each valid hour.
    """
    frames = []
    for init_dt in cycles:
        df = provider.fetch_forecast(conn, station, init_dt)
        if df.empty:
            continue
        frames.append(df[["valid_dt", "init_dt", "tmpf"]])
    if not frames:
        return pd.DataFrame(columns=["valid_dt", "init_dt", "tmpf"])
    return pd.concat(frames, ignore_index=True)


def blended_forecast(
    conn: sqlite3.Connection,
    station: str,
    provider: ForecastProvider,
    init_dt: str | None = None,
    halflife_hours: float = 6.0,
    uncertainty_multiplier: float = 1.0,
    trend_weight: float = 0.0,
) -> pd.DataFrame:
    """Compute a bias-corrected forecast for a station.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection to the weather DB.
    station : str
        ICAO code (e.g. ``"KDAL"``).
    provider : ForecastProvider
        The model provider (HRRR, GFS, etc.).
    init_dt : str, optional
        The model cycle to correct. If None, uses the latest complete cycle.
    halflife_hours : float
        Half-life of the exponential bias decay. Recent observations
        matter more.
    uncertainty_multiplier : float
        Multiplier for the bias std to form the uncertainty band.
    trend_weight : float
        Fraction of the model trend slope to apply as an additional
        correction. 0.0 = no trend correction (default). 0.15 = slight
        pull toward the direction recent HRRR cycles are trending.

    Returns
    -------
    pd.DataFrame
        One row per forecast hour with columns: ``valid_dt``, ``tmpf``
        (raw), ``tmpf_corrected``, ``uncertainty_low``, ``uncertainty_high``,
        ``forecast_hour``, ``bias_applied``, ``trend_correction``,
        ``tmpf_trend_adjusted``, ``init_dt``.
    """
    # Determine which cycle to correct
    if init_dt is None:
        cycles = provider.recent_cycles(conn, station, min_hours=18)
        if not cycles:
            return pd.DataFrame()
        init_dt = cycles[0]  # newest first

    # Load the forecast to correct
    forecast = provider.fetch_forecast(conn, station, init_dt)
    if forecast.empty:
        return pd.DataFrame()

    # Load all recent cycles for bias matching (more data = better bias)
    all_cycles = provider.recent_cycles(conn, station, min_hours=1)
    if not all_cycles:
        all_cycles = [init_dt]

    # Load METAR observations
    obs_df = _load_metar_for_station(conn, station)

    # Load all forecast data for matching (from all recent cycles)
    fcst_for_matching = _load_forecast_for_matching(conn, provider, station, all_cycles)

    # Compute rolling bias
    bias_df = compute_rolling_bias(obs_df, fcst_for_matching, halflife_hours=halflife_hours)

    # Apply bias correction
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=uncertainty_multiplier)

    # Compute and apply trend correction if requested
    if trend_weight > 0.0 and len(all_cycles) > 1:
        trend_cycles = _load_all_cycles_for_trend(conn, provider, station, all_cycles)
        trend_df = compute_trend_correction(trend_cycles, init_dt, trend_weight=trend_weight)

        if not trend_df.empty:
            trend_df["valid_dt"] = pd.to_datetime(trend_df["valid_dt"], utc=True)
            result["valid_dt"] = pd.to_datetime(result["valid_dt"], utc=True)
            result = result.merge(
                trend_df[["valid_dt", "trend_correction", "n_cycles"]],
                on="valid_dt",
                how="left",
            )
            result["trend_correction"] = result["trend_correction"].fillna(0.0)
            result["tmpf_trend_adjusted"] = result["tmpf_corrected"] + result["trend_correction"]
        else:
            result["trend_correction"] = 0.0
            result["tmpf_trend_adjusted"] = result["tmpf_corrected"]
    else:
        result["trend_correction"] = 0.0
        result["tmpf_trend_adjusted"] = result["tmpf_corrected"]

    return result


def list_recent_cycles(
    conn: sqlite3.Connection,
    station: str,
    provider: ForecastProvider,
    min_hours: int = 18,
) -> list[str]:
    """Convenience wrapper: list available complete forecast cycles."""
    return provider.recent_cycles(conn, station, min_hours=min_hours)