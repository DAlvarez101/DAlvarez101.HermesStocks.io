import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.models.advection import advection_predict, upwind_weight


def test_upwind_weight_directly_upwind_high_boost():
    """A station directly upwind (bearing equals wind direction) gets the full boost."""
    w = upwind_weight(bearing_from_target=0.0, wind_dir=0.0, half_width=45.0, boost=3.0)
    assert w == pytest.approx(3.0)


def test_upwind_weight_at_half_width_is_unity():
    """At exactly the half-width the weight should taper to 1.0."""
    w = upwind_weight(bearing_from_target=45.0, wind_dir=0.0, half_width=45.0, boost=3.0)
    assert w == pytest.approx(1.0)


def test_upwind_weight_downwind_is_unity():
    """A station directly downwind gets no boost."""
    w = upwind_weight(bearing_from_target=180.0, wind_dir=0.0, half_width=45.0, boost=3.0)
    assert w == pytest.approx(1.0)


def test_upwind_weight_intermediate():
    """Mid-way inside the cone the boost is between 1 and boost."""
    w = upwind_weight(bearing_from_target=22.5, wind_dir=0.0, half_width=45.0, boost=3.0)
    assert 1.0 < w < 3.0


def test_advection_predict_outputs_columns():
    residuals = pd.DataFrame(
        {"KDFW": [0.0], "KNORTH": [0.0], "KSOUTH": [10.0]},
        index=pd.to_datetime(["2024-06-01"]),
    )
    geom = pd.DataFrame(
        {
            "icao": ["KNORTH", "KSOUTH"],
            "dist_km": [100.0, 10.0],
            "bearing_from_target_deg": [0.0, 180.0],
        }
    )
    wind_df = pd.DataFrame(
        {"wind_dir_deg": [0.0], "wind_speed_kts": [10.0]},
        index=residuals.index,
    )
    result = advection_predict(residuals, geom, wind_df)
    assert list(result.columns) == ["KDFW", "predicted_residual", "corrected_residual"]


def test_advection_predict_synthetic_southern_residual_dominates():
    """Wind from the north; a close southern neighbor with a large residual dominates."""
    residuals = pd.DataFrame(
        {
            "KDFW": [0.0, 0.0],
            "KNORTH": [0.0, 0.0],
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
    result = advection_predict(
        residuals,
        geom,
        wind_df,
        target_col="KDFW",
        p=2.0,
        half_width=45.0,
        boost=3.0,
        l_adv_km=50.0,
    )

    # The close southern residual should dominate despite not being upwind.
    assert result.loc["2024-06-01", "predicted_residual"] > 5.0
    assert result.loc["2024-06-01", "predicted_residual"] <= 10.0
    assert result.loc["2024-06-02", "predicted_residual"] > 2.5
    assert result.loc["2024-06-02", "predicted_residual"] <= 5.0

    # corrected_residual mirrors the predicted correction, as in the baseline model.
    assert result["corrected_residual"].equals(result["predicted_residual"])
