"""5-minute observation fetcher from the NWS API (api.weather.gov).

No API key required — just a User-Agent header. Returns 5-minute ASOS
observations for a single station. Data is in GeoJSON format.

Endpoint: https://api.weather.gov/stations/{ICAO}/observations?limit=N
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

NWS_API_BASE = "https://api.weather.gov"
NWS_USER_AGENT = "(dfw-weather-dashboard, contact@dalvarez101.dev)"

# Same output columns as aviationweather.py for consistency.
OUTPUT_COLUMNS = [
    "station",
    "valid",
    "lat",
    "lon",
    "tmpf",
    "dewpf",
    "drct",
    "sknt",
    "skyc1",
    "mslp",
    "p01i",
]


def _c_to_f(c: float | None) -> float | None:
    """Convert Celsius to Fahrenheit, preserving None."""
    if c is None:
        return None
    return c * 9.0 / 5.0 + 32.0


def _kmh_to_kt(kmh: float | None) -> float | None:
    """Convert km/h to knots, preserving None."""
    if kmh is None:
        return None
    return kmh / 1.852


def _pa_to_hpa(pa: float | None) -> float | None:
    """Convert Pascals to hPa (same as mb), preserving None."""
    if pa is None:
        return None
    return pa / 100.0


def parse_nws_observations(payload: dict) -> pd.DataFrame:
    """Parse NWS API GeoJSON response into the project observation schema.

    Parameters
    ----------
    payload : dict
        Parsed JSON from the NWS API observations endpoint.

    Returns
    -------
    pd.DataFrame
        Observations in the standard schema (same columns as
        ``aviationweather.OUTPUT_COLUMNS``).
    """
    features = payload.get("features", [])
    if not features:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        station = props.get("stationId", "")
        ts = props.get("timestamp")

        temp_obj = props.get("temperature", {})
        dewp_obj = props.get("dewpoint", {})
        wdir_obj = props.get("windDirection", {})
        wspd_obj = props.get("windSpeed", {})
        pres_obj = props.get("barometricPressure", {})

        rows.append({
            "station": station,
            "valid": ts,
            "lat": coords[1] if len(coords) >= 2 else None,
            "lon": coords[0] if len(coords) >= 1 else None,
            "tmpf": _c_to_f(temp_obj.get("value") if temp_obj else None),
            "dewpf": _c_to_f(dewp_obj.get("value") if dewp_obj else None),
            "drct": (wdir_obj.get("value") if wdir_obj else None),
            "sknt": _kmh_to_kt(wspd_obj.get("value") if wspd_obj else None),
            "skyc1": None,  # NWS API cloud layers are structured differently
            "mslp": _pa_to_hpa(pres_obj.get("value") if pres_obj else None),
            "p01i": None,  # NWS API doesn't provide precip in this endpoint
        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # Convert valid to datetime
    df["valid"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    # Drop rows where valid is NaT
    df = df.dropna(subset=["valid"]).reset_index(drop=True)
    return df


def fetch_nws_observations(
    station: str,
    limit: int = 25,
    timeout: float = 15.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Fetch recent observations from the NWS API for a single station.

    Parameters
    ----------
    station : str
        ICAO code (e.g. ``"KDAL"``).
    limit : int
        Maximum number of observations to request (each is ~5 minutes apart).
    timeout : float
        HTTP request timeout in seconds.
    retries : int
        Retry count for transient HTTP errors.
    backoff : float
        Exponential backoff multiplier.

    Returns
    -------
    pd.DataFrame
        Parsed observations in the project schema.
    """
    url = f"{NWS_API_BASE}/stations/{station.upper()}/observations"
    headers = {"User-Agent": NWS_USER_AGENT}
    params = {"limit": limit}

    last_response: Optional[requests.Response] = None
    for attempt in range(retries):
        last_response = requests.get(url, headers=headers, params=params, timeout=timeout)
        if last_response.status_code == 429 and attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
            continue
        last_response.raise_for_status()
        break
    else:
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError(f"Failed to fetch NWS observations for {station}")

    payload = last_response.json()
    return parse_nws_observations(payload)