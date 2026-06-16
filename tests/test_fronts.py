import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.models.advection import (
    advection_predict,
    advection_predict_with_fronts,
    detect_front_day,
)


def test_detect_front_day_flags_high_gradient():
    """A sharp north-south residual gradient should be flagged as a front day."""
    residuals = pd.DataFrame(
        {
            "KDFW": [0.0],
            "KNORTH": [10.0],
            "KSOUTH": [-10.0],
        },
        index=pd.to_datetime(["2024-06-01"]),
    )
    geom = pd.DataFrame(
        {
            "icao": ["KNORTH", "KSOUTH"],
            "dist_km": [100.0, 100.0],
            "bearing_from_target_deg": [0.0, 180.0],
        }
    )
    flags = detect_front_day(residuals, geom, gradient_threshold=0.05)
    assert isinstance(flags, pd.Series)
    assert flags.index.equals(pd.to_datetime(["2024-06-01"]))
    assert flags.iloc[0] == True


def test_detect_front_day_no_flag_low_gradient():
    """A flat residual field should not be flagged as a front day."""
    residuals = pd.DataFrame(
        {
            "KDFW": [0.0],
            "KNORTH": [1.0],
            "KSOUTH": [1.0],
        },
        index=pd.to_datetime(["2024-06-01"]),
    )
    geom = pd.DataFrame(
        {
            "icao": ["KNORTH", "KSOUTH"],
            "dist_km": [100.0, 100.0],
            "bearing_from_target_deg": [0.0, 180.0],
        }
    )
    flags = detect_front_day(residuals, geom, gradient_threshold=0.05)
    assert isinstance(flags, pd.Series)
    assert flags.iloc[0] == False


def test_advection_predict_with_fronts_changes_on_front_day():
    """On a flagged front day, the front fallback prediction should differ from normal advection."""
    residuals = pd.DataFrame(
        {
            "KDFW": [0.0, 0.0],
            "KNORTH": [0.0, 5.0],
            "KSOUTH": [10.0, 5.0],
        },
        index=pd.to_datetime(["2024-06-01", "2024-06-02"]),
    )
    geom = pd.DataFrame(
        {
            "icao": ["KNORTH", "KSOUTH"],
            "dist_km": [100.0, 10.0],
            "bearing_from_target_deg": [0.0, 180.0],
        }
    )
    wind_df = pd.DataFrame(
        {
            "wind_dir_deg": [0.0, 0.0],
            "wind_speed_kts": [10.0, 10.0],
        },
        index=residuals.index,
    )
    # Make day 1 a front day by giving it a large north-south gradient.
    residuals.loc["2024-06-01", "KNORTH"] = 15.0
    residuals.loc["2024-06-01", "KSOUTH"] = -5.0

    normal = advection_predict(residuals, geom, wind_df, target_col="KDFW")
    with_fronts = advection_predict_with_fronts(
        residuals,
        geom,
        wind_df,
        target_col="KDFW",
        front_params={
            "gradient_threshold": 0.05,
            "front_fallback": "mean",
            "uncertainty_multiplier": 2.0,
        },
    )

    assert "front_day" in with_fronts.columns
    assert with_fronts.loc["2024-06-01", "front_day"] == True
    assert with_fronts.loc["2024-06-02", "front_day"] == False
    assert with_fronts.loc["2024-06-02", "predicted_residual"] == pytest.approx(
        normal.loc["2024-06-02", "predicted_residual"]
    )
    # On the front day, the mean fallback (5.0) differs from the advection-weighted result.
    assert with_fronts.loc["2024-06-01", "predicted_residual"] == pytest.approx(5.0)
    assert with_fronts.loc["2024-06-01", "corrected_residual"] == pytest.approx(5.0)
