
from dfw_temp_model.trading.signal import forecast_high_temp_simple, probability_above_threshold


def test_probability_above_threshold_with_gaussian():
    # mean=90, std=2, threshold=95 -> ~0.0062
    p = probability_above_threshold(90.0, 2.0, 95.0)
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_probability_above_threshold_zero_std():
    assert probability_above_threshold(96.0, 0.0, 95.0) == 1.0
    assert probability_above_threshold(94.0, 0.0, 95.0) == 0.0


def test_forecast_high_temp_simple_returns_value():
    result = forecast_high_temp_simple(
        latest_observed=92.0,
        hrrr_raw_high=94.0,
        predicted_residual=1.5,
        model_std=2.0,
    )
    assert result["corrected_high"] == 95.5
    assert result["model_std"] == 2.0
