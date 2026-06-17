"""Grid-search utilities for tuning the advection model parameters."""

from __future__ import annotations

from itertools import product
from typing import Callable, Tuple

import pandas as pd

from dfw_temp_model.evaluation.metrics import rmse
from dfw_temp_model.models.advection import advection_predict

DEFAULT_PARAM_GRID = {
    "p": [1.0, 1.5, 2.0, 2.5, 3.0],
    "boost": [1.0, 2.0, 3.0, 5.0, 8.0],
    "half_width": [30.0, 45.0, 60.0, 90.0],
    "l_adv_km": [20.0, 50.0, 100.0],
}


def make_advection_scorer(
    residuals_full: pd.DataFrame,
    geom: pd.DataFrame,
    wind_df: pd.DataFrame,
    val_idx,
    target_col: str = "KDFW",
) -> Callable[[dict], float]:
    """Return a scoring function that maps advection params to ``-RMSE`` on ``val_idx``.

    The returned callable takes a parameter dict with keys ``p``, ``boost``,
    ``half_width``, and ``l_adv_km`` and returns the negative RMSE of the corrected
    target high temperature on the validation set.  It is suitable for use as an
    objective to maximize.

    Parameters
    ----------
    residuals_full:
        DataFrame whose columns include ``target_col`` and one column per neighbor
        residual.  The ``target_col`` column is interpreted as the target residual
        (observed minus forecast) to be corrected.
    geom:
        Station geometry table with ``icao`` and columns ``dist_km``,
        ``bearing_from_target_deg``.
    wind_df:
        DataFrame indexed by date containing ``wind_dir_deg`` and ``wind_speed_kts``.
    val_idx:
        Index labels (dates) to use for validation scoring.
    target_col:
        Name of the target station residual column.

    Returns
    -------
    callable
        ``scorer(params: dict) -> float`` returning ``-RMSE``.
    """

    def scorer(params: dict) -> float:
        pred_df = advection_predict(
            residuals_full.loc[residuals_full.index.intersection(val_idx)],
            geom,
            wind_df,
            target_col=target_col,
            p=params["p"],
            boost=params["boost"],
            half_width=params["half_width"],
            l_adv_km=params["l_adv_km"],
        )
        if pred_df.empty:
            raise ValueError("No overlapping validation dates between residuals and wind data")
        observed_residual = pred_df[target_col]
        predicted_residual = pred_df["predicted_residual"]
        error_rmse = rmse(observed_residual, predicted_residual)  # type: ignore[arg-type]
        return -float(error_rmse)

    return scorer


def grid_search_advection(
    residuals_full: pd.DataFrame,
    geom: pd.DataFrame,
    wind_df: pd.DataFrame,
    val_idx,
    param_grid: dict,
    target_col: str = "KDFW",
) -> Tuple[dict, float, pd.DataFrame]:
    """Exhaustive grid search over advection parameters on the validation set.

    Parameters
    ----------
    residuals_full, geom, wind_df, val_idx, target_col:
        Same as ``make_advection_scorer``.
    param_grid:
        Dict mapping parameter name to a list of candidate values.  Must contain
        ``p``, ``boost``, ``half_width``, and ``l_adv_km``.

    Returns
    -------
    best_params:
        Parameter dict with the highest validation score.
    best_score:
        The corresponding ``-RMSE``.
    all_results:
        DataFrame with one row per combination, columns for each parameter plus
        ``score`` and ``rmse``.
    """
    required_keys = {"p", "boost", "half_width", "l_adv_km"}
    missing = required_keys - set(param_grid.keys())
    if missing:
        raise ValueError(f"param_grid missing required keys: {sorted(missing)}")

    scorer = make_advection_scorer(residuals_full, geom, wind_df, val_idx, target_col=target_col)

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    results = []
    best_score = -float("inf")
    best_params = None

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            score = scorer(params)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("Scoring failed for params %s: %s", params, exc)
            score = -float("inf")
        row = params.copy()
        row["score"] = score
        row["rmse"] = -score
        results.append(row)

        if score > best_score and not score == -float("inf"):
            best_score = score
            best_params = params.copy()

    all_results = pd.DataFrame(results)
    return best_params, best_score, all_results
