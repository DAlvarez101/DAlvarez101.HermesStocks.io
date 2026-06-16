import pandas as pd
import pytest

from dfw_temp_model.models.baseline import inverse_distance_predict


def test_idw_baseline_simple():
    residuals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 12:00", "2026-01-01 13:00"]),
            "KDFW": [2.0, 1.0],
            "KDAL": [4.0, 2.0],
            "KFTW": [0.0, 0.0],
        }
    )
    geom = pd.DataFrame(
        {
            "icao": ["KDAL", "KFTW"],
            "dist_km": [10.0, 20.0],
            "bearing_from_target_deg": [90.0, 180.0],
        }
    )
    result = inverse_distance_predict(residuals, geom, target_col="KDFW", p=2.0)

    # Weights: KDAL = 1/10^2 = 0.01, KFTW = 1/20^2 = 0.0025
    # Normalized: KDAL = 0.8, KFTW = 0.2
    expected_pred_0 = 0.8 * 4.0 + 0.2 * 0.0
    expected_pred_1 = 0.8 * 2.0 + 0.2 * 0.0
    assert result.loc[0, "predicted_residual"] == pytest.approx(expected_pred_0)
    assert result.loc[1, "predicted_residual"] == pytest.approx(expected_pred_1)


def test_idw_baseline_returns_columns():
    residuals = pd.DataFrame(
        {
            "KDFW": [2.0],
            "KDAL": [4.0],
        }
    )
    geom = pd.DataFrame(
        {"icao": ["KDAL"], "dist_km": [10.0], "bearing_from_target_deg": [90.0]}
    )
    result = inverse_distance_predict(residuals, geom, target_col="KDFW")
    assert list(result.columns) == ["KDFW", "predicted_residual", "corrected_residual"]
    assert result.loc[0, "predicted_residual"] == 4.0
    assert result.loc[0, "corrected_residual"] == 4.0


def test_idw_p_varies():
    residuals = pd.DataFrame(
        {
            "KDFW": [0.0],
            "KDAL": [10.0],
            "KFTW": [0.0],
        }
    )
    geom = pd.DataFrame(
        {
            "icao": ["KDAL", "KFTW"],
            "dist_km": [10.0, 20.0],
            "bearing_from_target_deg": [90.0, 180.0],
        }
    )
    low_p = inverse_distance_predict(residuals, geom, target_col="KDFW", p=1.0)
    high_p = inverse_distance_predict(residuals, geom, target_col="KDFW", p=4.0)
    # Higher p gives closer station more weight, so prediction should move toward 10.0
    assert low_p.loc[0, "predicted_residual"] < high_p.loc[0, "predicted_residual"]
