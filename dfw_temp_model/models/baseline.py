"""Baseline inverse-distance-weighted temperature residual model."""

import pandas as pd


def inverse_distance_predict(
    residuals: pd.DataFrame,
    geom: pd.DataFrame,
    target_col: str = "KDFW",
    p: float = 2.0,
) -> pd.DataFrame:
    """Predict a target station residual as an inverse-distance-weighted mean of neighbors.

    Parameters
    ----------
    residuals : pd.DataFrame
        DataFrame with one column per station residual (including ``target_col``).
        Each row is an observation / forecast valid time.
    geom : pd.DataFrame
        Geometry table with a column ``icao`` (or index named ``icao``) and columns
        ``dist_km`` and ``bearing_from_target_deg``. Stations with distance <= 0 are
        ignored as neighbors.
    target_col : str, optional
        Name of the target station column in ``residuals``.
    p : float, optional
        Inverse-distance exponent. Neighbor weights are ``1 / dist_km ** p``.

    Returns
    -------
    pd.DataFrame
        Columns ``target_col`` (observed residual), ``predicted_residual``,
        and ``corrected_residual`` (which equals the predicted residual).
    """
    if "icao" in geom.columns:
        neighbor_index = geom.set_index("icao")
    else:
        neighbor_index = geom.copy()

    neighbor_index = neighbor_index[neighbor_index.index != target_col]
    neighbor_cols = [
        col for col in residuals.columns if col in neighbor_index.index and col != target_col
    ]

    if not neighbor_cols:
        raise ValueError("No neighbor residual columns available for target {target_col!r}")

    distances = neighbor_index.loc[neighbor_cols, "dist_km"]
    if (distances <= 0).any():
        raise ValueError("All neighbor distances must be positive for inverse-distance weighting")

    weights = 1.0 / (distances ** p)
    normalized = weights / weights.sum()

    neighbor_residuals = residuals[neighbor_cols]
    predicted = neighbor_residuals @ normalized

    result = pd.DataFrame(
        {
            target_col: residuals[target_col],
            "predicted_residual": predicted,
        }
    )
    result["corrected_residual"] = result["predicted_residual"]
    return result
