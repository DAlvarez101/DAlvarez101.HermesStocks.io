"""Multi-model forecast blending with bias correction.

Loads HRRR, NAM, NBM, and ECMWF-IFS forecasts from the Wethr.net API
(each with its own source tag in hrrr_forecasts) and applies independent
rolling bias correction to each model. The blended forecast uses
lead-time-dependent inverse-MAE weighting: at each forecast hour, each
model's weight is 1/(MAE + epsilon), normalized across models that have
data at that hour. This gives HRRR more weight at short range (where
its 3km resolution and hourly cycling produce lower MAE) and shifts
weight to global models at longer horizons where HRRR has no data.

Model spread (max - min across models at each hour) feeds into
effective_sigma() via quadrature, inflating uncertainty when models
disagree.

ECMWF was originally removed because Open-Meteo snapped KLAX to an
inland grid point ~9 miles from the coast, running ~10F warmer. Wethr.net
serves station-level data, solving the grid mismatch. Validation confirmed
0.00F MAE between Wethr HRRR and GRIB2 extraction at KLAX.

NAM replaced GFS on 2026-06-27. NAM is NCEP's regional mesoscale model
(~12 km resolution) with hourly output to 48h — higher spatial resolution
than GFS (~13 km global) and better suited for short-range coastal
temperature forecasting at KLAX. Both run every 6h on Wethr.net.

Model weighting follows NOAA's NBM v5.0 MAE weighting methodology and
the inverse-variance weighting literature (Sun et al. 2017).
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from dfw_temp_model.blending.bias import apply_bias_correction, compute_rolling_bias
from dfw_temp_model.blending.providers import HRRRProvider
from dfw_temp_model.blending.wethr_provider import WethrProvider


# Weight assigned to a model at a forecast hour where it has no MAE data
# (insufficient matchup observations). Must be < 1.0 so unverified models
# don't dilute verified ones equally. 0.3 gives verified models ~3x more
# influence than unverified at hours where MAE data is sparse.
UNVERIFIED_FALLBACK_WEIGHT = 0.3

# Model configurations: (model_name, source_tag, min_hours)
# HRRR is the primary short-range model (hourly, 18h horizon).
# NAM is NCEP's regional mesoscale model (hourly, 48h horizon, runs every 6h).
# NBM is NOAA's blended model (hourly, 36h+).
# ECMWF-IFS is the best global model (every 12h, 65h horizon).
MULTI_MODELS = [
    WethrProvider("HRRR", "wethr", min_hours=18),
    WethrProvider("NAM", "wethr-nam", min_hours=18),
    WethrProvider("NBM", "wethr-nbm", min_hours=18),
    WethrProvider("ECMWF-IFS", "wethr-ecmwf", min_hours=18),
]

# Map source tags to output column names.
# tmpf_ecmwf keeps its name for backwards compatibility with downstream code.
SOURCE_TO_COL = {
    "wethr": "tmpf_hrrr",
    "wethr-nam": "tmpf_nam",
    "wethr-nbm": "tmpf_nbm",
    "wethr-ecmwf": "tmpf_ecmwf",
}


def _load_all_forecasts_for_bias(
    conn: sqlite3.Connection,
    provider,
    station: str,
) -> pd.DataFrame:
    """Load ALL forecast rows from this provider for bias computation.

    Using only the latest cycle gives very few matched hours (e.g. a just-
    started 12Z cycle overlaps with only 1 obs hour), producing a
    wildly unstable bias from a single data point. Using all cycles gives
    hundreds of matched hours and a stable, meaningful bias estimate.

    For each valid_hour, the most recent cycle's forecast is used (latest
    init_dt), since that's the most skillful forecast for that hour.
    """
    all_rows = pd.read_sql_query(
        "SELECT init_dt, valid_dt, tmpf FROM hrrr_forecasts "
        "WHERE station = ? AND source = ? ORDER BY valid_dt",
        conn, params=[station, provider.SOURCE],
    )
    if all_rows.empty:
        return pd.DataFrame()

    all_rows["valid_hour"] = pd.to_datetime(all_rows["valid_dt"], utc=True).dt.floor("h")
    all_rows["init_dt_ts"] = pd.to_datetime(all_rows["init_dt"], utc=True)

    # Keep the latest cycle's forecast for each valid_hour
    latest = all_rows.sort_values("init_dt_ts").drop_duplicates("valid_hour", keep="last")
    return latest[["valid_hour", "tmpf"]].rename(columns={"tmpf": "tmpf_fcst"})


def _load_model_forecast(
    conn: sqlite3.Connection,
    provider,
    station: str,
    init_dt: str,
) -> pd.DataFrame:
    """Load, bias-correct, and label a single model's forecast.

    Bias is computed using ALL of this provider's forecast cycles in the DB
    (not just the latest cycle) to ensure enough matched observation hours
    for a stable bias estimate. See _load_all_forecasts_for_bias.

    Returns an empty DataFrame if the provider has no data.
    """
    raw = provider.fetch_forecast(conn, station, init_dt)
    if raw.empty:
        return pd.DataFrame()

    # Build obs df for bias correction
    obs_df = pd.read_sql_query(
        "SELECT valid AS valid_hour, tmpf AS tmpf_obs FROM metar_observations "
        "WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn, params=[station],
    )
    obs_df["valid_hour"] = pd.to_datetime(obs_df["valid_hour"], utc=True).dt.floor("h")

    # Use ALL forecast cycles for bias computation (not just the target cycle)
    all_fcst_df = _load_all_forecasts_for_bias(conn, provider, station)
    if all_fcst_df.empty:
        result = raw.copy()
        result["tmpf_corrected"] = result["tmpf"]
        return result

    # Compute error = obs - fcst at each matched hour using a lookup table.
    fcst_lookup = all_fcst_df.groupby("valid_hour")["tmpf_fcst"].mean()
    obs_df["error"] = obs_df.apply(
        lambda row: row["tmpf_obs"] - fcst_lookup.get(row["valid_hour"], row["tmpf_obs"])
        if row["valid_hour"] in fcst_lookup.index else None,
        axis=1,
    )
    # Drop unmatched hours before bias computation
    obs_matched = obs_df.dropna(subset=["error"])

    try:
        bias_df = compute_rolling_bias(obs_matched, all_fcst_df)
        corrected = apply_bias_correction(raw, bias_df)
        return corrected
    except Exception:
        # Fallback: no bias correction
        result = raw.copy()
        result["tmpf_corrected"] = result["tmpf"]
        return result


def _compute_lead_time_weights(
    conn: sqlite3.Connection,
    providers: list,
    station: str,
    epsilon: float = 0.5,
    min_matchups: int = 5,
) -> dict:
    """Compute lead-time-dependent inverse-MAE weights for each model.

    For each model, at each forecast hour, computes the mean absolute error
    (MAE) of forecasts vs observations. The weight is
    1 / (MAE + epsilon), normalized across all models that have data at
    that forecast hour.

    This naturally gives more weight to models with lower error at each
    lead time. Models with no data at a given hour get zero weight there,
    and the normalization redistributes their share to the remaining models.
    No hardcoded lead-time thresholds needed.

    The epsilon floor (0.5F) prevents a single model from getting infinite
    weight when it happens to have MAE=0 at some hour.

    Returns
    -------
    dict
        {source_tag: {forecast_hour: weight}} where weights at each hour
        sum to ~1.0 across available models.
    """
    obs_df = pd.read_sql_query(
        "SELECT valid AS valid_hour, tmpf AS tmpf_obs FROM metar_observations "
        "WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn, params=[station],
    )
    if obs_df.empty:
        return {}
    obs_df["valid_hour"] = pd.to_datetime(obs_df["valid_hour"], utc=True).dt.floor("h")

    # For each model, compute per-forecast-hour MAE
    mae_by_model = {}

    for provider in providers:
        all_rows = pd.read_sql_query(
            "SELECT init_dt, valid_dt, forecast_hour, tmpf FROM hrrr_forecasts "
            "WHERE station = ? AND source = ? ORDER BY valid_dt",
            conn, params=[station, provider.SOURCE],
        )
        if all_rows.empty:
            continue

        all_rows["valid_hour"] = pd.to_datetime(all_rows["valid_dt"], utc=True).dt.floor("h")
        all_rows["init_dt_ts"] = pd.to_datetime(all_rows["init_dt"], utc=True)

        # Keep latest cycle's forecast for each valid_hour
        latest = all_rows.sort_values("init_dt_ts").drop_duplicates("valid_hour", keep="last")

        # Match against observations
        merged = latest.merge(obs_df, on="valid_hour", how="inner")
        if len(merged) < min_matchups:
            # Not enough data for stable MAE -- use default MAE of 2.0F
            mae_by_model[provider.SOURCE] = {}
            continue

        # Compute absolute error
        merged["abs_error"] = (merged["tmpf"] - merged["tmpf_obs"]).abs()

        # Group by forecast hour and compute mean absolute error
        merged["fhr_int"] = merged["forecast_hour"].round().astype(int)
        mae_per_hour = merged.groupby("fhr_int")["abs_error"].mean()

        mae_by_model[provider.SOURCE] = mae_per_hour.to_dict()

    # Compute inverse-MAE weights per forecast hour
    weights_by_model = {source: {} for source in mae_by_model}

    # Get all forecast hours where at least one model has data
    all_hours = set()
    for mae_dict in mae_by_model.values():
        all_hours.update(mae_dict.keys())

    for fhr in sorted(all_hours):
        # Compute inverse-MAE for each model at this hour
        inv_maes = {}
        for source, mae_dict in mae_by_model.items():
            if fhr in mae_dict:
                mae = mae_dict[fhr]
                inv_maes[source] = 1.0 / (mae + epsilon)
            # else: model has no MAE data at this hour

        # Also include models with insufficient matchup data (neutral weight)
        # but only if they have forecasts at this hour in the DB
        for source in mae_by_model:
            if source not in inv_maes and not mae_by_model[source]:
                # This model had insufficient matchups -- check if it has
                # forecasts at this hour
                has_data = conn.execute(
                    "SELECT COUNT(*) FROM hrrr_forecasts "
                    "WHERE station = ? AND source = ? AND forecast_hour = ?",
                    [station, source, fhr],
                ).fetchone()[0]
                if has_data > 0:
                    inv_maes[source] = 1.0 / (2.0 + epsilon)

        # Normalize
        total_inv = sum(inv_maes.values())
        if total_inv > 0:
            for source, inv_mae in inv_maes.items():
                weights_by_model[source][fhr] = inv_mae / total_inv

    return weights_by_model


def multi_model_forecast(
    conn: sqlite3.Connection,
    station: str,
    init_dt: str | None = None,
    hrrr_weight: float = 1.0,
    ecmwf_weight: float = 0.0,
) -> pd.DataFrame:
    """Load, bias-correct, and blend multiple model forecasts for a station.

    Loads all available Wethr.net models (HRRR, NAM, NBM, ECMWF-IFS),
    applies independent rolling bias correction to each, then computes
    a correlation-weighted blend. Model spread (max - min across models)
    is returned for sigma inflation via effective_sigma().

    The bias correction and sigma mechanics are unchanged from the
    single-model design -- each model gets its own EWMA bias, and
    model_spread feeds into effective_sigma() via quadrature exactly
    as the sigma module was designed to handle.

    Parameters
    ----------
    conn : sqlite3.Connection
    station : str
        ICAO code.
    init_dt : str, optional
        Model cycle to use. If None, uses the latest available cycle
        for each model independently.
    hrrr_weight, ecmwf_weight : float
        Ignored (kept for signature compatibility). Weights are now
        data-driven (correlation-based).

    Returns
    -------
    pd.DataFrame
        One row per forecast hour with columns:
        ``valid_dt``, ``forecast_hour``, ``tmpf_hrrr``, ``tmpf_nam``,
        ``tmpf_nbm``, ``tmpf_ecmwf``, ``tmpf_blended``, ``model_spread``,
        ``uncertainty_low``, ``uncertainty_high``.
        ``tmpf_ecmwf`` column name is kept for backwards compatibility
        but now contains ECMWF-IFS data from Wethr.net (not Open-Meteo).
        Models without data have None in their column.
    """
    # Try to load each model
    model_dfs = {}  # source_tag -> bias-corrected DataFrame

    for provider in MULTI_MODELS:
        # Determine which cycle to use for this model
        if init_dt is not None:
            # Check if this init_dt exists for this model
            check = provider.fetch_forecast(conn, station, init_dt)
            if check.empty:
                cycles = provider.recent_cycles(conn, station)
                model_init = cycles[0] if cycles else None
            else:
                model_init = init_dt
        else:
            cycles = provider.recent_cycles(conn, station)
            model_init = cycles[0] if cycles else None

        if model_init is None:
            continue

        try:
            df = _load_model_forecast(conn, provider, station, model_init)
            if not df.empty:
                df = df[["valid_dt", "forecast_hour", "tmpf_corrected"]].rename(
                    columns={"tmpf_corrected": "tmpf"}
                )
                df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
                model_dfs[provider.SOURCE] = df
        except Exception:
            continue

    if not model_dfs:
        return pd.DataFrame()

    # Use HRRR as the base (shortest horizon but highest skill at short range)
    # If HRRR is not available, use whichever model has the most rows
    base_source = "wethr" if "wethr" in model_dfs else max(model_dfs, key=lambda s: len(model_dfs[s]))
    base_df = model_dfs[base_source].copy()

    # Merge all models on valid_dt (outer join to keep all hours from all models)
    result = base_df[["valid_dt", "forecast_hour"]].copy()
    temp_cols = []

    for source_tag, df in model_dfs.items():
        col_name = SOURCE_TO_COL.get(source_tag, f"tmpf_{source_tag}")
        result = result.merge(
            df[["valid_dt", "tmpf"]].rename(columns={"tmpf": col_name}),
            on="valid_dt", how="outer",
        )
        temp_cols.append(col_name)

    # Fill forecast_hour for rows that come from non-base models (outer join
    # leaves NaN for forecast_hour when the base model doesn't have that hour).
    # Compute from valid_dt relative to the base model's init_dt.
    if not result.empty:
        base_init_dt = model_dfs[base_source]["valid_dt"].min() - pd.Timedelta(hours=1)
        # Actually, forecast_hour = valid_dt - init_dt. The base model's init_dt
        # is the first valid_dt minus 1 hour (f1 = first hour after init).
        # Simpler: fill NaN forecast_hour with the hour offset from the earliest
        # non-NaN forecast_hour's valid_dt.
        known = result["forecast_hour"].notna()
        if known.any() and not known.all():
            # Find the base init time: valid_dt - forecast_hour for known rows
            base_init = result.loc[known, "valid_dt"].iloc[0] - pd.Timedelta(
                hours=int(result.loc[known, "forecast_hour"].iloc[0])
            )
            # Fill NaN forecast_hours from valid_dt - base_init
            mask = result["forecast_hour"].isna()
            result.loc[mask, "forecast_hour"] = (
                result.loc[mask, "valid_dt"] - base_init
            ).dt.total_seconds() / 3600.0

    # Compute model_spread = max - min across available models at each hour.
    # This feeds into effective_sigma() via quadrature: when models disagree,
    # uncertainty is inflated. This is exactly what effective_sigma was
    # designed to do -- it was just always getting 0 before.
    temp_data = result[temp_cols].apply(pd.to_numeric, errors="coerce")
    result["model_spread"] = temp_data.max(axis=1) - temp_data.min(axis=1)
    result["model_spread"] = result["model_spread"].fillna(0.0)

    # Compute lead-time-dependent inverse-MAE weights
    available_providers = [p for p in MULTI_MODELS if p.SOURCE in model_dfs]
    lt_weights = _compute_lead_time_weights(conn, available_providers, station)

    # Compute blended forecast = weighted average using per-hour weights.
    # At hours where a model has no data, it is excluded from the average.
    result["tmpf_blended"] = 0.0
    total_weight_per_row = pd.Series(0.0, index=result.index)

    for source_tag in model_dfs:
        col_name = SOURCE_TO_COL.get(source_tag, f"tmpf_{source_tag}")
        vals = pd.to_numeric(result[col_name], errors="coerce")
        mask = vals.notna()

        # Get per-hour weights for this model
        model_weights = lt_weights.get(source_tag, {})

        for idx in result.index[mask]:
            fhr = int(round(result.loc[idx, "forecast_hour"]))
            w = model_weights.get(fhr, 0.0)
            if w > 0:
                result.loc[idx, "tmpf_blended"] += vals[idx] * w
                total_weight_per_row.loc[idx] += w
            else:
                # No per-hour weight available for this model at this hour.
                # Use a reduced fallback weight so the model still contributes
                # but doesn't dilute verified models equally.
                w_fallback = UNVERIFIED_FALLBACK_WEIGHT
                result.loc[idx, "tmpf_blended"] += vals[idx] * w_fallback
                total_weight_per_row.loc[idx] += w_fallback

    # Normalize per-row: divide by the sum of weights that actually contributed
    has_data = total_weight_per_row > 0
    result.loc[has_data, "tmpf_blended"] = result.loc[has_data, "tmpf_blended"] / total_weight_per_row[has_data]

    if not has_data.any():
        # Fallback: equal weight for all available models (should not happen
        # if model_dfs is non-empty, but guard against division by zero)
        n_models = len(model_dfs)
        for source_tag in model_dfs:
            col_name = SOURCE_TO_COL.get(source_tag, f"tmpf_{source_tag}")
            vals = pd.to_numeric(result[col_name], errors="coerce").fillna(0)
            result["tmpf_blended"] += vals / n_models

    # Uncertainty bands (from bias correction, if available)
    result["uncertainty_low"] = result["tmpf_blended"]
    result["uncertainty_high"] = result["tmpf_blended"]

    # Sort and return
    result = result.sort_values("valid_dt").reset_index(drop=True)

    # Ensure all expected columns exist (fill missing with None)
    for col in ["tmpf_hrrr", "tmpf_nam", "tmpf_nbm", "tmpf_ecmwf"]:
        if col not in result.columns:
            result[col] = None

    return result[["valid_dt", "forecast_hour", "tmpf_hrrr", "tmpf_nam",
                   "tmpf_nbm", "tmpf_ecmwf", "tmpf_blended",
                   "model_spread", "uncertainty_low", "uncertainty_high"]]