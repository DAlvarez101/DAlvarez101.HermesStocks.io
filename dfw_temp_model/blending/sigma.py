"""Horizon-dependent forecast uncertainty (sigma).

Weather forecast error grows with lead time. A fixed sigma underestimates
uncertainty at long horizons and overestimates it at short ones. This linear
model approximates HRRR error growth:

    sigma = base_sigma + growth_rate * forecast_hour

capped at *max_sigma*.
"""


def sigma_for_forecast_hour(
    forecast_hour: float,
    base_sigma: float = 0.8,
    growth_rate: float = 0.15,
    max_sigma: float = 5.5,
) -> float:
    """Return forecast uncertainty (sigma) that grows with forecast hour.

    Parameters
    ----------
    forecast_hour : float
        Hours ahead the forecast is valid (e.g. 1 for the next hour,
        18 for the last HRRR frame).
    base_sigma : float
        Sigma at hour 0 — the irreducible analysis/observation error.
    growth_rate : float
        Sigma increase per forecast hour, in degrees F.
    max_sigma : float
        Hard ceiling so absurdly long horizons don't explode.

    Examples
    --------
    >>> round(sigma_for_forecast_hour(1), 2)
    0.95
    >>> round(sigma_for_forecast_hour(6), 2)
    1.7
    >>> round(sigma_for_forecast_hour(18), 2)
    3.5
    >>> round(sigma_for_forecast_hour(72), 2)
    5.5
    """
    return min(base_sigma + growth_rate * forecast_hour, max_sigma)


def effective_sigma(
    forecast_hour: float,
    model_spread: float = 0.0,
    spread_weight: float = 0.5,
    base_sigma: float = 0.8,
    growth_rate: float = 0.15,
    max_sigma: float = 5.5,
    n_models: int | None = None,
    min_floor_sigma: float = 2.0,
) -> float:
    """Combine horizon-based sigma with inter-model spread.

    When two independent models disagree, the true forecast uncertainty
    is higher than horizon alone suggests.  We add the spread contribution
    in quadrature (independent variances add) so a large spread inflates
    sigma but a zero spread leaves it unchanged.

    When ``n_models >= 2``, model agreement can also *constrain* sigma.
    The horizon-based sigma is a prior calibrated for single-model HRRR;
    at long lead times it hits the ``max_sigma`` cap and becomes
    uninformative.  Inter-model spread is a direct empirical measurement
    of uncertainty — when models agree tightly, sigma should be lower than
    the horizon cap.  The spread-based sigma (``spread_weight *
    model_spread``) serves as a ceiling, floored at ``min_floor_sigma`` to
    prevent overconfidence.

    Parameters
    ----------
    forecast_hour : float
        Hours ahead the forecast is valid.
    model_spread : float
        Absolute difference between model forecasts at the relevant valid
        hour, in degrees F.
    spread_weight : float
        Fraction of the spread to fold into sigma (default 0.5).  This
        dampens the spread contribution so a 6°F disagreement adds only
        3°F in quadrature.
    base_sigma, growth_rate, max_sigma : float
        Parameters for the underlying ``sigma_for_forecast_hour``.
    n_models : int or None
        Number of models contributing to the spread.  When None or < 2,
        the spread can only inflate sigma (backward-compatible behavior).
        When >= 2, the spread also constrains sigma downward — model
        agreement caps sigma at ``max(spread_weight * model_spread,
        min_floor_sigma)``.
    min_floor_sigma : float
        Minimum sigma when the spread constraint is active (default 2.0°F).
        Prevents overconfidence when models agree perfectly (spread=0).

    Returns
    -------
    float
        ``min(max_sigma, quadrature)`` when ``n_models`` is None or < 2.
        ``min(max_sigma, quadrature, max(spread_sigma, min_floor_sigma))``
        when ``n_models >= 2``.
    """
    sigma_horizon = sigma_for_forecast_hour(
        forecast_hour, base_sigma, growth_rate, max_sigma
    )
    if model_spread <= 0 and (n_models is None or n_models < 2):
        return sigma_horizon

    spread_component = spread_weight * model_spread
    quadrature = min(
        (sigma_horizon**2 + spread_component**2) ** 0.5,
        max_sigma,
    )

    if n_models is not None and n_models >= 2:
        # Multi-model: spread is an empirical uncertainty measurement.
        # When models agree, this constrains sigma below the horizon cap.
        spread_ceiling = max(spread_component, min_floor_sigma)
        return min(quadrature, spread_ceiling)

    return quadrature


def shrink_sigma_for_observations(
    sigma: float,
    hours_elapsed: float,
    shrink_rate: float = 0.5,
    min_fraction: float = 0.3,
) -> float:
    """Reduce sigma proportional to time elapsed in the climate day.

    As live observations accumulate, uncertainty about the remaining
    trajectory shrinks — we have more information.  Uses the time of the
    latest observation relative to climate-day start, so it works
    naturally with both 5-minute and hourly data.

        sigma_shrunk = sigma * max(min_fraction, 1 - shrink_rate * hours_elapsed / 24)

    With default ``shrink_rate=0.5``, a fully observed day (24h elapsed)
    halves sigma; a half-observed day (12h) shrinks by 25%.  The floor at
    ``min_fraction`` (default 0.3) prevents collapse to zero.

    Parameters
    ----------
    sigma : float
        The prior sigma (e.g. from ``effective_sigma`` or
        ``sigma_for_forecast_hour``).
    hours_elapsed : float
        Hours between climate-day start and the latest observation,
        capped at 24.0.  Use a float to support 5-minute observation
        granularity (e.g. 10.5 hours).
    shrink_rate : float
        Fraction of sigma to remove per full day of elapsed observations
        (default 0.5).
    min_fraction : float
        Floor as a fraction of original sigma (default 0.3).

    Returns
    -------
    float
        Shrunk sigma, never below ``sigma * min_fraction``.
    """
    if sigma <= 0:
        return 0.0
    if hours_elapsed <= 0:
        return sigma
    hours_elapsed = min(hours_elapsed, 24.0)
    fraction = max(min_fraction, 1.0 - shrink_rate * hours_elapsed / 24.0)
    return sigma * fraction