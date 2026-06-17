"""Evaluation metrics for forecast correction."""

import numpy as np
import pandas as pd


def _mask_valid(obs: pd.Series, pred: pd.Series) -> pd.Series:
    """Return boolean mask of pairwise non-missing values."""
    return obs.notna() & pred.notna()


def rmse(obs: pd.Series, pred: pd.Series) -> float:
    """Root mean squared error over pairwise non-missing observations."""
    mask = _mask_valid(obs, pred)
    errors = obs[mask] - pred[mask]
    return float(np.sqrt(np.mean(errors**2)))


def mae(obs: pd.Series, pred: pd.Series) -> float:
    """Mean absolute error over pairwise non-missing observations."""
    mask = _mask_valid(obs, pred)
    errors = obs[mask] - pred[mask]
    return float(np.mean(np.abs(errors)))


def bucket_hit_rate(obs: pd.Series, pred: pd.Series, bucket_width: float = 1.0) -> float:
    """Fraction of predictions whose rounded value matches the rounded observation.

    Both ``obs`` and ``pred`` are rounded to the nearest ``bucket_width``.
    Only pairwise non-missing entries are compared.
    """
    mask = _mask_valid(obs, pred)
    obs_rounded = (obs[mask] / bucket_width).round() * bucket_width
    pred_rounded = (pred[mask] / bucket_width).round() * bucket_width
    if len(obs_rounded) == 0:
        return np.nan
    return float((obs_rounded == pred_rounded).mean())


def evaluate_correction(
    obs: pd.Series, fcst: pd.Series, corrected: pd.Series
) -> dict:
    """Compare raw forecast and corrected forecast against observations.

    Returns
    -------
    dict
        raw_rmse, raw_mae, corrected_rmse, corrected_mae, rmse_improvement,
        and bucket_hit_rate (against the corrected forecast).
    """
    raw_rmse = rmse(obs, fcst)
    raw_mae = mae(obs, fcst)
    corrected_rmse = rmse(obs, corrected)
    corrected_mae = mae(obs, corrected)

    if np.isclose(raw_rmse, 0.0):
        rmse_improvement = 0.0
    else:
        rmse_improvement = (raw_rmse - corrected_rmse) / raw_rmse

    hit_rate = bucket_hit_rate(obs, corrected)

    return {
        "raw_rmse": raw_rmse,
        "raw_mae": raw_mae,
        "corrected_rmse": corrected_rmse,
        "corrected_mae": corrected_mae,
        "rmse_improvement": rmse_improvement,
        "bucket_hit_rate": hit_rate,
    }
