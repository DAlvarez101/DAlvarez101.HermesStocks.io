"""Wind-advection-weighted temperature residual model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dfw_temp_model.features.geometry import smallest_angle_diff


def upwind_weight(
    bearing_from_target: float,
    wind_dir: float,
    half_width: float = 45.0,
    boost: float = 3.0,
) -> float:
    """Return a multiplicative weight that boosts neighbors lying upwind of the target.

    Parameters
    ----------
    bearing_from_target : float
        Bearing from the target station to the neighbor, in degrees (0° = north).
    wind_dir : float
        Wind direction, i.e. direction the wind is coming from, in degrees.
    half_width : float, optional
        Half-width of the upwind cone in degrees. Outside this cone the weight is 1.0.
    boost : float, optional
        Multiplicative boost at the center of the cone (when bearing equals wind_dir).

    Returns
    -------
    float
        A value in ``[1.0, boost]`` when the neighbor lies within the upwind cone,
        otherwise ``1.0``.
    """
    diff = smallest_angle_diff(bearing_from_target, wind_dir)
    if diff > half_width:
        return 1.0
    return 1.0 + (boost - 1.0) * math.cos(math.radians(diff * (90.0 / half_width)))


def advection_predict(
    residuals: pd.DataFrame,
    geom: pd.DataFrame,
    wind_df: pd.DataFrame,
    target_col: str = "KDFW",
    p: float = 2.0,
    half_width: float = 45.0,
    boost: float = 3.0,
    l_adv_km: float = 50.0,
) -> pd.DataFrame:
    """Predict the target station residual using wind-advection-weighted neighbors.

    For each date in ``residuals``, neighbor weights combine inverse-distance,
    an upwind cone boost, and an advection distance decay. The weighted mean of
    neighbor residuals is returned as the predicted correction.

    Parameters
    ----------
    residuals : pd.DataFrame
        DataFrame with one column per station residual (including ``target_col``).
        Rows are indexed by date / forecast valid time.
    geom : pd.DataFrame
        Geometry table with a column ``icao`` and columns ``dist_km`` and
        ``bearing_from_target_deg``.
    wind_df : pd.DataFrame
        DataFrame indexed by date containing ``wind_dir_deg`` and ``wind_speed_kts``.
    target_col : str, optional
        Name of the target station column in ``residuals``.
    p : float, optional
        Inverse-distance exponent.
    half_width : float, optional
        Half-width of the upwind cone in degrees.
    boost : float, optional
        Maximum multiplicative upwind boost at the center of the cone.
    l_adv_km : float, optional
        Advection decay length in kilometers; weights are multiplied by
        ``exp(-dist_km / l_adv_km)``.

    Returns
    -------
    pd.DataFrame
        Columns ``target_col`` (observed residual), ``predicted_residual``,
        and ``corrected_residual`` (the predicted correction).
    """
    if "icao" in geom.columns:
        geom_index = geom.set_index("icao")
    else:
        geom_index = geom.copy()

    neighbor_cols = [c for c in residuals.columns if c != target_col]
    if not neighbor_cols:
        raise ValueError(f"No neighbor residual columns available for target {target_col!r}")

    geom_index = geom_index.loc[[c for c in neighbor_cols if c in geom_index.index]]
    missing_geom = set(neighbor_cols) - set(geom_index.index)
    if missing_geom:
        raise ValueError(f"Missing geometry for neighbors: {sorted(missing_geom)}")

    dists = geom_index.loc[neighbor_cols, "dist_km"].values
    bearings = geom_index.loc[neighbor_cols, "bearing_from_target_deg"].values

    if (dists <= 0).any():
        raise ValueError("All neighbor distances must be positive")

    out = residuals[[target_col]].copy()
    predictions = []

    common_dates = residuals.index.intersection(wind_df.index)
    if len(common_dates) != len(residuals.index):
        missing_dates = residuals.index.difference(wind_df.index)
        raise ValueError(
            f"Wind data missing for {len(missing_dates)} residual date(s): "
            f"{list(missing_dates[:5])}{' ...' if len(missing_dates) > 5 else ''}"
        )

    for date in residuals.index:
        wind_dir = float(wind_df.loc[date, "wind_dir_deg"])

        idw = 1.0 / (dists ** p)
        uw = np.array(
            [upwind_weight(b, wind_dir, half_width, boost) for b in bearings]
        )
        adv_decay = np.exp(-dists / l_adv_km)
        weights = idw * uw * adv_decay
        weights = weights / weights.sum()

        neighbor_residuals = residuals.loc[date, neighbor_cols].fillna(0).values
        pred = float(neighbor_residuals @ weights)
        predictions.append(pred)

    out["predicted_residual"] = predictions
    out["corrected_residual"] = out["predicted_residual"]
    return out
