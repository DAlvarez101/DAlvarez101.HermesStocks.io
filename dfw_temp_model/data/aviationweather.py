"""Live METAR fetcher from AviationWeather.gov (free, no API key)."""
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

from dfw_temp_model.config import Station

AVWX_BASE = "https://aviationweather.gov/api/data/metar"

# Standard columns shared with the IEM ASOS fetcher.
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


def _station_lookup(stations: Iterable[Station]) -> dict[str, dict]:
    """Return {icao: {lat, lon}} from a list of Station objects."""
    return {
        s.icao.upper(): {"lat": s.lat, "lon": s.lon}
        for s in stations
    }


def build_aviationweather_url(stations: Iterable[Station], hours: int = 2) -> str:
    """Build the AviationWeather.gov METAR JSON URL.

    Parameters
    ----------
    stations : Iterable[Station]
        Stations to request.
    hours : int
        How many hours back to request (endpoint max is roughly 24).

    Returns
    -------
    str
        Fully-qualified request URL.
    """
    ids = ",".join(s.icao for s in stations)
    req = requests.Request(
        "GET",
        AVWX_BASE,
        params={
            "ids": ids,
            "format": "json",
            "hours": hours,
        },
    )
    return req.prepare().url


def _parse_wind(wdir: object) -> float:
    """Convert AviationWeather wind direction to degrees.

    Numeric values are returned as ints/floats. Variable (VRB) becomes NaN.
    """
    if isinstance(wdir, (int, float)):
        return float(wdir)
    if isinstance(wdir, str):
        if wdir.upper() == "VRB":
            return float("nan")
        try:
            return float(wdir)
        except ValueError:
            return float("nan")
    return float("nan")


def _first_sky_cover(clouds: Optional[List[dict]]) -> Optional[str]:
    """Return the first sky-cover abbreviation if available."""
    if not clouds or not isinstance(clouds, list):
        return None
    first = clouds[0]
    if isinstance(first, dict):
        return first.get("cover")
    return None


def parse_metar_json(
    payload: List[dict],
    stations: Iterable[Station],
) -> pd.DataFrame:
    """Convert AviationWeather METAR JSON into the project observation schema.

    Parameters
    ----------
    payload : List[dict]
        JSON array from AviationWeather.gov.
    stations : Iterable[Station]
        Station definitions used to attach lat/lon.

    Returns
    -------
    pd.DataFrame
        Observations in the standard schema, or an empty DataFrame with the
        correct columns if payload is empty.
    """
    if not payload:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    lookup = _station_lookup(stations)
    rows: List[dict] = []

    for r in payload:
        if not isinstance(r, dict):
            continue
        icao = (r.get("icaoId") or "").upper()
        if not icao:
            continue

        # obsTime is a Unix timestamp in seconds.
        obs_ts = r.get("obsTime")
        try:
            valid = (
                datetime.fromtimestamp(float(obs_ts), tz=timezone.utc)
                if obs_ts is not None
                else pd.NaT
            )
        except (TypeError, ValueError):
            valid = pd.NaT

        temp_c = r.get("temp")
        dewp_c = r.get("dewp")
        wdir = _parse_wind(r.get("wdir"))
        wspd = r.get("wspd")
        altim = r.get("altim")
        p01i = r.get("precip")  # AviationWeather may not provide precip

        rows.append(
            {
                "station": icao,
                "valid": valid,
                "lat": lookup.get(icao, {}).get("lat"),
                "lon": lookup.get(icao, {}).get("lon"),
                "tmpf": temp_c * 9.0 / 5.0 + 32.0 if temp_c is not None else None,
                "dewpf": dewp_c * 9.0 / 5.0 + 32.0 if dewp_c is not None else None,
                "drct": wdir,
                "sknt": float(wspd) if wspd is not None else None,
                "skyc1": _first_sky_cover(r.get("clouds")),
                "mslp": float(altim) * 100.0 if altim is not None else None,
                "p01i": float(p01i) if p01i is not None else None,
            }
        )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df


def fetch_aviationweather(
    stations: Iterable[Station],
    hours: int = 2,
    cache_path: Optional[str] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Fetch live METARs from AviationWeather.gov and optionally cache them.

    Parameters
    ----------
    stations : Iterable[Station]
        Stations to request.
    hours : int
        Hours back to request.
    cache_path : Optional[str]
        If provided, the resulting DataFrame is written to this Parquet path.
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
    url = build_aviationweather_url(stations, hours)
    last_response: Optional[requests.Response] = None

    for attempt in range(retries):
        last_response = requests.get(url, timeout=timeout)
        if last_response.status_code == 429 and attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
            continue
        last_response.raise_for_status()
        break
    else:
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("failed to fetch METARs from AviationWeather.gov")

    payload = last_response.json()
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected JSON list from AviationWeather, got {type(payload).__name__}"
        )

    df = parse_metar_json(payload, stations)

    if cache_path is not None and not df.empty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def fetch_single_station(
    station: str,
    hours: int = 2,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper for a single station ICAO string."""
    return fetch_aviationweather(
        stations=[Station(station, 0.0, 0.0, 0.0, "unknown")],
        hours=hours,
        cache_path=cache_path,
    )
