"""Wind-advection-weighted temperature residual model."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dfw_temp_model.features.geometry import smallest_angle_diff
from dfw_temp_model.models.baseline import inverse_distance_predict


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


def _fit_plane_gradient(
    residuals: pd.DataFrame,
    geom: pd.DataFrame,
    target_col: str = "KDFW",
) -> tuple[float, float]:
    """Fit a plane r(x, y) = a*x + b*y + c to neighbor residuals and return (dx, dy).

    The gradient magnitude is ``sqrt(a**2 + b**2)`` in residual units per km.
    Local coordinates are taken from ``x_m``/``y_m`` when present; otherwise
    they are reconstructed from ``dist_km`` and ``bearing_from_target_deg``.
    """
    if "icao" in geom.columns:
        geom_index = geom.set_index("icao")
    else:
        geom_index = geom.copy()

    neighbor_cols = [c for c in residuals.columns if c != target_col]
    geom_index = geom_index.loc[
        [c for c in neighbor_cols if c in geom_index.index]
    ]
    neighbor_cols = [c for c in neighbor_cols if c in geom_index.index]

    if "x_m" in geom_index.columns and "y_m" in geom_index.columns:
        x = geom_index.loc[neighbor_cols, "x_m"].values / 1000.0
        y = geom_index.loc[neighbor_cols, "y_m"].values / 1000.0
    else:
        dists = geom_index.loc[neighbor_cols, "dist_km"].values
        bearings = np.radians(
            geom_index.loc[neighbor_cols, "bearing_from_target_deg"].values
        )
        # bearing 0° = north, clockwise. x = east = sin(bearing), y = north = cos(bearing).
        x = dists * np.sin(bearings)
        y = dists * np.cos(bearings)

    r = residuals.loc[residuals.index[0], neighbor_cols].fillna(0).values

    # Ordinary least-squares plane fit: design matrix [x, y, 1].
    A = np.column_stack([x, y, np.ones_like(x)])
    coeff, *_ = np.linalg.lstsq(A, r, rcond=None)
    return float(coeff[0]), float(coeff[1])


def detect_front_day(
    residuals: pd.DataFrame,
    geom: pd.DataFrame,
    gradient_threshold: float = 0.05,
    target_col: str = "KDFW",
) -> pd.Series:
    """Flag dates where the residual gradient exceeds ``gradient_threshold`` °F/km.

    For each date, a plane is fit to neighbor residuals as a function of local
    easting/northing (in km). The gradient magnitude of that plane is compared
    to the threshold and the result is returned as a boolean Series indexed by
    date.

    Parameters
    ----------
    residuals : pd.DataFrame
        DataFrame with one column per station residual (including ``target_col``).
        Rows are indexed by date / forecast valid time.
    geom : pd.DataFrame
        Geometry table with a column ``icao`` and columns ``x_m``, ``y_m``,
        ``dist_km`` and ``bearing_from_target_deg``.
    gradient_threshold : float, optional
        Threshold in °F/km. A date is flagged when the gradient magnitude is
        strictly greater than this value.
    target_col : str, optional
        Name of the target station column in ``residuals``.

    Returns
    -------
    pd.Series
        Boolean series indexed by ``residuals.index``.
    """
    flags = []
    for date in residuals.index:
        dx, dy = _fit_plane_gradient(
            residuals.loc[[date]], geom, target_col=target_col
        )
        gradient_magnitude = math.hypot(dx, dy)
        flags.append(gradient_magnitude > gradient_threshold)
    return pd.Series(flags, index=residuals.index, dtype=bool)


def advection_predict_with_fronts(
    residuals: pd.DataFrame,
    geom: pd.DataFrame,
    wind_df: pd.DataFrame,
    target_col: str = "KDFW",
    p: float = 2.0,
    half_width: float = 45.0,
    boost: float = 3.0,
    l_adv_km: float = 50.0,
    front_params: dict | None = None,
) -> pd.DataFrame:
    """Predict target residual, using a fallback for days with detected fronts.

    Non-front days use the normal ``advection_predict``. On front days the
    wind direction is not trusted; the predicted residual is replaced by a
    simple neighbor mean (``front_fallback='mean'``) or by the IDW baseline
    (``front_fallback='idw'``). The output includes a ``front_day`` flag and a
    ``front_uncertainty_multiplier`` column for downstream use.

    Parameters
    ----------
    residuals, geom, wind_df, target_col, p, half_width, boost, l_adv_km
        See ``advection_predict``.
    front_params : dict, optional
        Dictionary with keys ``gradient_threshold`` (float),
        ``front_fallback`` (``"mean"`` or ``"idw"``), and
        ``uncertainty_multiplier`` (float). Defaults are applied for missing
        keys.

    Returns
    -------
    pd.DataFrame
        Columns ``target_col``, ``predicted_residual``, ``corrected_residual``,
        ``front_day``, and ``front_uncertainty_multiplier``.
    """
    params = {
        "gradient_threshold": 0.05,
        "front_fallback": "mean",
        "uncertainty_multiplier": 2.0,
    }
    if front_params:
        params.update(front_params)

    front_fallback = params["front_fallback"]
    if front_fallback not in {"mean", "idw"}:
        raise ValueError(f"front_fallback must be 'mean' or 'idw', got {front_fallback!r}")

    front_days = detect_front_day(
        residuals, geom, gradient_threshold=params["gradient_threshold"], target_col=target_col
    )

    normal = advection_predict(
        residuals,
        geom,
        wind_df,
        target_col=target_col,
        p=p,
        half_width=half_width,
        boost=boost,
        l_adv_km=l_adv_km,
    )

    neighbor_cols = [c for c in residuals.columns if c != target_col]
    if not neighbor_cols:
        raise ValueError(f"No neighbor residual columns available for target {target_col!r}")

    out = normal.copy()
    out["front_day"] = front_days.reindex(out.index)
    out["front_uncertainty_multiplier"] = np.where(
        out["front_day"], params["uncertainty_multiplier"], 1.0
    )

    if front_fallback == "mean":
        fallback = residuals[neighbor_cols].mean(axis=1, skipna=True)
    else:  # idw
        fallback = inverse_distance_predict(
            residuals, geom, target_col=target_col, p=p
        )["predicted_residual"]

    out["predicted_residual"] = np.where(
        out["front_day"], fallback.reindex(out.index), out["predicted_residual"]
    )
    out["corrected_residual"] = out["predicted_residual"]
    return out
