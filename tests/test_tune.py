"""Tests for dfw_temp_model.training.tune."""

import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.training.tune import (
    DEFAULT_PARAM_GRID,
    grid_search_advection,
    make_advection_scorer,
)


def _synthetic_residuals(n_dates: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates).astype(str)
    df = pd.DataFrame(
        {
            "KDFW": rng.normal(size=n_dates),
            "KDAL": rng.normal(size=n_dates),
            "KADS": rng.normal(size=n_dates),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def _synthetic_geometry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "icao": ["KDAL", "KADS"],
            "x_m": [1e4, -1e4],
            "y_m": [0.0, 0.0],
            "dist_km": [10.0, 20.0],
            "bearing_from_target_deg": [90.0, 270.0],
        }
    )


def _synthetic_wind(n_dates: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates).astype(str)
    directions = [0.0, 90.0, 180.0, 270.0] * ((n_dates // 4) + 1)
    directions = directions[:n_dates]
    df = pd.DataFrame(
        {
            "wind_dir_deg": directions,
            "wind_speed_kts": [10.0] * n_dates,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_make_advection_scorer_returns_callable_and_score():
    residuals = _synthetic_residuals()
    geom = _synthetic_geometry()
    wind = _synthetic_wind()
    val_idx = residuals.index[:3]

    scorer = make_advection_scorer(residuals, geom, wind, val_idx, target_col="KDFW")
    assert callable(scorer)

    params = {"p": 2.0, "boost": 3.0, "half_width": 45.0, "l_adv_km": 50.0}
    score = scorer(params)
    assert isinstance(score, float)
    assert score <= 0.0

    # RMSE is always >= 0, so -RMSE <= 0.  A tiny constant residual vector gives
    # a predicted residual very close to 0, hence RMSE approximately equal to the
    # target residual RMSE.  The score should be finite.
    assert np.isfinite(score)


def test_grid_search_advection_runs_and_returns_best_params():
    residuals = _synthetic_residuals(n_dates=6)
    geom = _synthetic_geometry()
    wind = _synthetic_wind(n_dates=6)
    val_idx = residuals.index[3:]

    param_grid = {
        "p": [1.0, 2.0],
        "boost": [1.0, 2.0],
        "half_width": [30.0, 60.0],
        "l_adv_km": [20.0, 100.0],
    }

    best_params, best_score, all_results = grid_search_advection(
        residuals, geom, wind, val_idx, param_grid=param_grid, target_col="KDFW"
    )

    assert isinstance(best_params, dict)
    assert set(best_params.keys()) == {"p", "boost", "half_width", "l_adv_km"}
    assert isinstance(best_score, float)
    assert best_score <= 0.0
    assert isinstance(all_results, pd.DataFrame)
    assert len(all_results) == 16  # 2 * 2 * 2 * 2
    assert "score" in all_results.columns
    assert "rmse" in all_results.columns

    # Best score in all_results should match returned best_score.
    assert np.isclose(all_results["score"].max(), best_score)


def test_default_param_grid_matches_spec():
    assert DEFAULT_PARAM_GRID == {
        "p": [1.0, 1.5, 2.0, 2.5, 3.0],
        "boost": [1.0, 2.0, 3.0, 5.0, 8.0],
        "half_width": [30.0, 45.0, 60.0, 90.0],
        "l_adv_km": [20.0, 50.0, 100.0],
    }
