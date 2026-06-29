"""Generate a corrected temperature-forecast signal from the live DB."""
import re
from typing import Optional

import pandas as pd

from dfw_temp_model.config import TARGET_ICAO
from dfw_temp_model.storage.obs_db import get_db, hrrr_forecast_for_cycle, latest_complete_hrrr_cycle
from dfw_temp_model.storage.obs_db import latest_by_station
from dfw_temp_model.blending.sigma import sigma_for_forecast_hour


def probability_above_threshold(mean_temp: float, std: float, threshold: float) -> float:
    """Probability that the true high exceeds *threshold* under a Gaussian model."""
    # Lazy import scipy inside the function to avoid a deadlock when this module
    # is imported alongside modules that load web3/py_clob_client_v2.
    from scipy.stats import norm
    if std <= 0:
        return 1.0 if mean_temp > threshold else 0.0
    return float(1.0 - norm.cdf(threshold, loc=mean_temp, scale=std))


def _extract_market_threshold(question: str) -> Optional[float]:
    """Naive threshold extractor: find the first temperature-like number in °F."""
    match = re.search(r"(\d+)\s*°?F", question, re.IGNORECASE)
    return float(match.group(1)) if match else None


def forecast_high_temp(
    db_path: str,
    market_question: str,
    predicted_residual: Optional[float] = None,
    model_std: Optional[float] = None,
) -> dict:
    """Return corrected forecast for the target station and market threshold.

    When *model_std* is None (default), sigma is computed from the mean
    forecast hour of today's HRRR rows via :func:`sigma_for_forecast_hour`.
    Pass an explicit float to override (used by tests).
    """
    conn = get_db(db_path)
    try:
        latest = latest_by_station(conn)
        target_row = latest[latest["station"] == TARGET_ICAO]
        latest_observed = float(target_row.iloc[0]["tmpf"]) if not target_row.empty else None

        init_dt = latest_complete_hrrr_cycle(conn, TARGET_ICAO, required_hours=18)
        hrrr_raw_high = None
        if init_dt:
            df = hrrr_forecast_for_cycle(conn, TARGET_ICAO, init_dt)
            if not df.empty:
                df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
                today = pd.Timestamp.utcnow().floor("d")
                today_rows = df[df["valid_dt"].dt.floor("d") == today]
                if not today_rows.empty:
                    hrrr_raw_high = float(today_rows["tmpf"].max())

        if hrrr_raw_high is None:
            raise ValueError("No HRRR forecast available for today")

        corrected_high = hrrr_raw_high + (predicted_residual or 0.0)
        threshold = _extract_market_threshold(market_question)
        if threshold is None:
            raise ValueError(f"Could not extract temperature threshold from: {market_question}")

        # Horizon-dependent sigma: uncertainty grows with forecast lead time.
        # Use the mean forecast hour of today's HRRR rows as the effective horizon.
        mean_fhr = float(today_rows["forecast_hour"].mean()) if not today_rows.empty else 6.0
        model_std = sigma_for_forecast_hour(mean_fhr) if model_std is None else model_std

        prob_yes = probability_above_threshold(corrected_high, model_std, threshold)
        return {
            "corrected_high": round(corrected_high, 2),
            "hrrr_raw_high": round(hrrr_raw_high, 2),
            "predicted_residual": predicted_residual,
            "model_std": round(model_std, 4),
            "forecast_hour": round(mean_fhr, 1),
            "threshold": threshold,
            "probability_yes": round(prob_yes, 4),
            "probability_no": round(1.0 - prob_yes, 4),
            "latest_observed": latest_observed,
        }
    finally:
        conn.close()


def forecast_high_temp_simple(
    latest_observed: Optional[float],
    hrrr_raw_high: float,
    predicted_residual: Optional[float],
    model_std: float,
) -> dict:
    """Pure helper for tests and offline use."""
    corrected_high = hrrr_raw_high + (predicted_residual or 0.0)
    return {
        "corrected_high": corrected_high,
        "hrrr_raw_high": hrrr_raw_high,
        "predicted_residual": predicted_residual,
        "model_std": model_std,
    }
