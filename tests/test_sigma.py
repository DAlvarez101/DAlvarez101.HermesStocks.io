import pytest

from dfw_temp_model.blending.sigma import sigma_for_forecast_hour


def test_sigma_grows_with_horizon():
    assert sigma_for_forecast_hour(0) == 0.8
    assert round(sigma_for_forecast_hour(1), 2) == 0.95
    assert round(sigma_for_forecast_hour(6), 2) == 1.7
    assert round(sigma_for_forecast_hour(18), 2) == 3.5
    assert sigma_for_forecast_hour(72) == 5.5
    assert sigma_for_forecast_hour(999) == 5.5


def test_sigma_monotonic():
    for h in range(0, 72):
        assert sigma_for_forecast_hour(h) <= sigma_for_forecast_hour(h + 1)


def test_sigma_custom_params():
    s = sigma_for_forecast_hour(10, base_sigma=1.0, growth_rate=0.2, max_sigma=4.0)
    assert s == 3.0  # 1.0 + 0.2 * 10


from dfw_temp_model.blending.sigma import effective_sigma


def test_effective_sigma_no_spread_returns_horizon_sigma():
    """With zero model spread, effective_sigma equals horizon sigma."""
    horizon = sigma_for_forecast_hour(6)  # 1.7
    assert effective_sigma(forecast_hour=6, model_spread=0.0) == pytest.approx(horizon)


def test_effective_sigma_with_spread_exceeds_horizon():
    """Non-zero spread should increase sigma beyond horizon-only."""
    horizon = sigma_for_forecast_hour(6)
    eff = effective_sigma(forecast_hour=6, model_spread=4.0)
    assert eff > horizon


def test_effective_sigma_quadrature_formula():
    """Verify the quadrature formula: sqrt(horizon^2 + (spread_weight*spread)^2)."""
    horizon = sigma_for_forecast_hour(6)  # 1.7
    spread = 4.0
    weight = 0.5
    expected = (horizon**2 + (weight * spread)**2) ** 0.5
    assert effective_sigma(forecast_hour=6, model_spread=spread, spread_weight=weight) == pytest.approx(expected, rel=1e-6)


def test_effective_sigma_floors_at_horizon():
    """Without n_models, spread can never reduce sigma below horizon-only value."""
    horizon = sigma_for_forecast_hour(18)  # 3.5
    eff = effective_sigma(forecast_hour=18, model_spread=0.0, spread_weight=0.5)
    assert eff >= horizon - 1e-10


def test_effective_sigma_spread_constrains_at_long_horizon():
    """When n_models >= 2 and models agree, sigma is constrained below horizon cap."""
    # fhr=33: horizon sigma = 5.5 (capped). spread=1.91, spread_sigma=0.96.
    # With n_models=2: sigma should be max(spread_sigma, min_floor) = max(0.96, 2.0) = 2.0
    # NOT the horizon cap of 5.5.
    eff = effective_sigma(forecast_hour=33, model_spread=1.91, n_models=2)
    assert eff < 5.5, f"Sigma should be constrained below 5.5, got {eff}"
    assert eff >= 2.0 - 1e-10, f"Sigma should be at least min_floor_sigma=2.0, got {eff}"
    assert eff == pytest.approx(2.0, abs=0.01)


def test_effective_sigma_n_models_none_preserves_old_behavior():
    """When n_models=None (default), spread cannot reduce sigma (backward compat)."""
    horizon = sigma_for_forecast_hour(33)  # 5.5 (capped)
    eff = effective_sigma(forecast_hour=33, model_spread=1.91, n_models=None)
    assert eff == pytest.approx(horizon, rel=1e-6)


def test_effective_sigma_n_models_1_preserves_old_behavior():
    """When n_models=1 (single model), spread cannot reduce sigma."""
    horizon = sigma_for_forecast_hour(33)  # 5.5
    eff = effective_sigma(forecast_hour=33, model_spread=1.91, n_models=1)
    assert eff == pytest.approx(horizon, rel=1e-6)


def test_effective_sigma_large_spread_not_constrained():
    """When models disagree (spread large), sigma stays high even with n_models >= 2."""
    # fhr=33: horizon=5.5, spread=8.0, spread_sigma=4.0
    # quadrature = sqrt(5.5^2 + 4.0^2) = 6.8 -> capped at 5.5
    # spread_ceiling = max(4.0, 2.0) = 4.0
    # sigma = min(5.5, 4.0) = 4.0
    eff = effective_sigma(forecast_hour=33, model_spread=8.0, n_models=2)
    assert eff >= 4.0 - 1e-10, f"Large spread should keep sigma high, got {eff}"
    assert eff <= 5.5 + 1e-10


def test_effective_sigma_short_horizon_unchanged_with_n_models():
    """At short horizon, n_models >= 2 doesn't reduce sigma below quadrature."""
    # fhr=6: horizon=1.7, spread=0.5, spread_sigma=0.25
    # quadrature = sqrt(1.7^2 + 0.25^2) = 1.718
    # spread_ceiling = max(0.25, 2.0) = 2.0
    # sigma = min(1.718, 2.0) = 1.718
    horizon = sigma_for_forecast_hour(6)  # 1.7
    quadrature = (horizon**2 + (0.5 * 0.5)**2) ** 0.5
    eff = effective_sigma(forecast_hour=6, model_spread=0.5, n_models=2)
    assert eff == pytest.approx(quadrature, rel=1e-6)


def test_effective_sigma_zero_spread_with_n_models_uses_floor():
    """When n_models >= 2 and spread=0, sigma is min_floor_sigma (not horizon)."""
    eff = effective_sigma(forecast_hour=33, model_spread=0.0, n_models=2)
    assert eff == pytest.approx(2.0, abs=0.01)


def test_effective_sigma_custom_min_floor():
    """Custom min_floor_sigma changes the floor."""
    eff_default = effective_sigma(forecast_hour=33, model_spread=0.0, n_models=2, min_floor_sigma=2.0)
    eff_custom = effective_sigma(forecast_hour=33, model_spread=0.0, n_models=2, min_floor_sigma=3.0)
    assert eff_custom > eff_default
    assert eff_custom == pytest.approx(3.0, abs=0.01)


def test_effective_sigma_custom_weight():
    """Custom spread_weight changes the contribution."""
    eff_default = effective_sigma(forecast_hour=6, model_spread=4.0, spread_weight=0.5)
    eff_high = effective_sigma(forecast_hour=6, model_spread=4.0, spread_weight=1.0)
    assert eff_high > eff_default


def test_effective_sigma_respects_max_sigma():
    """Effective sigma should not exceed max_sigma cap."""
    eff = effective_sigma(forecast_hour=6, model_spread=100.0, max_sigma=5.5)
    assert eff <= 5.5 + 1e-10


from dfw_temp_model.blending.sigma import shrink_sigma_for_observations


def test_shrink_sigma_zero_hours_returns_original():
    """Zero hours elapsed -> sigma unchanged."""
    sigma = 3.5
    assert shrink_sigma_for_observations(sigma, hours_elapsed=0.0) == pytest.approx(sigma)


def test_shrink_sigma_full_day_shrinks_by_half():
    """24 hours elapsed with default shrink_rate=0.5 -> sigma halved."""
    sigma = 3.5
    result = shrink_sigma_for_observations(sigma, hours_elapsed=24.0, shrink_rate=0.5)
    assert result == pytest.approx(sigma * 0.5)


def test_shrink_sigma_half_day():
    """12 hours elapsed with shrink_rate=0.5 -> 25% reduction."""
    sigma = 3.5
    result = shrink_sigma_for_observations(sigma, hours_elapsed=12.0, shrink_rate=0.5)
    assert result == pytest.approx(sigma * (1 - 0.5 * 12.0 / 24.0))


def test_shrink_sigma_floors_at_30_percent():
    """Sigma should never shrink below 30% of original."""
    sigma = 3.5
    result = shrink_sigma_for_observations(sigma, hours_elapsed=48.0, shrink_rate=1.0)
    assert result >= sigma * 0.3 - 1e-10


def test_shrink_sigma_custom_rate():
    """Custom shrink_rate changes the reduction."""
    sigma = 3.5
    result_default = shrink_sigma_for_observations(sigma, hours_elapsed=12.0, shrink_rate=0.5)
    result_aggressive = shrink_sigma_for_observations(sigma, hours_elapsed=12.0, shrink_rate=0.8)
    assert result_aggressive < result_default


def test_shrink_sigma_zero_sigma_stays_zero():
    """Edge case: sigma=0 stays 0."""
    assert shrink_sigma_for_observations(0.0, hours_elapsed=12.0) == 0.0


def test_shrink_sigma_fractional_hours():
    """10.5 hours elapsed works (5-min obs precision)."""
    sigma = 3.5
    result = shrink_sigma_for_observations(sigma, hours_elapsed=10.5, shrink_rate=0.5)
    expected = sigma * (1 - 0.5 * 10.5 / 24.0)
    assert result == pytest.approx(expected)


def test_shrink_sigma_caps_at_24_hours():
    """Hours elapsed beyond 24 should be treated as 24 (full day)."""
    sigma = 3.5
    result_24 = shrink_sigma_for_observations(sigma, hours_elapsed=24.0, shrink_rate=0.5)
    result_30 = shrink_sigma_for_observations(sigma, hours_elapsed=30.0, shrink_rate=0.5)
    assert result_24 == pytest.approx(result_30)


def test_weight_fallback_does_not_equal_verified_weight():
    """When a model has no MAE weight (w=0), its fallback weight should be
    less than 1.0 so it doesn't dilute verified models equally."""
    from dfw_temp_model.blending.multi import UNVERIFIED_FALLBACK_WEIGHT
    assert UNVERIFIED_FALLBACK_WEIGHT < 1.0, (
        f"Fallback weight should be < 1.0 to avoid diluting verified models, "
        f"got {UNVERIFIED_FALLBACK_WEIGHT}"
    )
    assert UNVERIFIED_FALLBACK_WEIGHT > 0.0, "Should still contribute"