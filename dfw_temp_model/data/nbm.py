"""Lightweight NBM 2-m temperature fetcher from NOAA Open Data on AWS.

Uses rasterio to read Cloud Optimized GeoTIFF (COG) files via HTTP range
requests. Only the target pixel window is downloaded -- no full file fetch.

NBM temp COG files store temperature directly in Fahrenheit as int16.
Endpoint: https://noaa-nbm-pds.s3.amazonaws.com/blendv5.0/conus/...
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
import rasterio
import requests

from dfw_temp_model.config import Station, TARGET_ICAO

NBM_OUTPUT_COLUMNS = [
    "station",
    "init_dt",
    "forecast_hour",
    "valid_dt",
    "lat",
    "lon",
    "tmpf",
]

# NOAA NBM AWS Open Data public bucket (HTTPS endpoint).
NBM_BASE_URL = "https://noaa-nbm-pds.s3.amazonaws.com"


def _cog_url(init_dt: datetime, forecast_hour: int) -> str:
    """Build the NBM temp COG URL for a given cycle and forecast hour.

    NBM path pattern:
      blendv5.0/conus/{YYYY}/{MM}/{DD}/{HH}00/temp/
      blendv5.0_conus_temp_{init_iso}_{valid_iso}.tif
    """
    ymd = init_dt.strftime("%Y/%m/%d")
    hh = init_dt.strftime("%H")
    init_iso = init_dt.strftime("%Y-%m-%dT%H:00")
    valid_dt = init_dt + timedelta(hours=forecast_hour)
    valid_iso = valid_dt.strftime("%Y-%m-%dT%H:00")
    return (
        f"{NBM_BASE_URL}/blendv5.0/conus/{ymd}/{hh}00/temp/"
        f"blendv5.0_conus_temp_{init_iso}_{valid_iso}.tif"
    )


def _valid_dt(init_dt: pd.Timestamp, forecast_hour: int) -> pd.Timestamp:
    """Return valid time = init_dt + forecast_hour hours."""
    return init_dt + pd.Timedelta(hours=forecast_hour)


def _find_latest_cycle(
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    now: Optional[datetime] = None,
    timeout: float = 15.0,
) -> tuple[pd.Timestamp, bool]:
    """Find the most recent published NBM cycle by checking COG file existence.

    Returns (init_dt, found). Init_dt is floored to the hour in UTC.
    NBM cycles are published with a ~2 hour lag (at run time, the cycle from
    2 hours ago is typically the latest available).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    ts_now = pd.Timestamp(now)
    if ts_now.tz is None:
        ts_now = ts_now.tz_localize("UTC")
    else:
        ts_now = ts_now.tz_convert("UTC")

    for i in range(lookback_hours + 1):
        init = ts_now - pd.Timedelta(hours=i)
        init = init.floor("h")
        url = _cog_url(init.to_pydatetime(), forecast_hour)
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                return init, True
        except Exception:
            pass
        time.sleep(0.2)

    return ts_now.floor("h"), False


def _read_temp_pixel(url: str, lat: float, lon: float, timeout: float = 60.0) -> float:
    """Read a single temperature pixel from an NBM COG file via HTTP range request.

    Returns temperature in Fahrenheit (NBM stores int16 Fahrenheit directly).
    """
    with rasterio.open(url) as src:
        col, row = src.index(lon, lat)
        val = src.read(1, window=((row, row + 1), (col, col + 1)))[0][0]
        if val == src.nodata:
            return float("nan")
        return float(val)


def fetch_latest_nbm_2m_temp(
    stations: Iterable[Station],
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    cache_path: Optional[str] = None,
    timeout: float = 90.0,
) -> pd.DataFrame:
    """Fetch the latest NBM 2 m temperature forecast for each station.

    Parameters
    ----------
    stations : Iterable[Station]
        Stations to extract.
    forecast_hour : int
        Forecast hour to read (default 1).
    lookback_hours : int
        How many past cycles to try if the latest is not yet published.
    cache_path : Optional[str]
        Optional Parquet cache path.
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        One row per station with init_dt, forecast_hour, valid_dt, tmpf.
    """
    init_dt, found = _find_latest_cycle(forecast_hour, lookback_hours)
    if not found:
        return pd.DataFrame(columns=NBM_OUTPUT_COLUMNS)

    url = _cog_url(init_dt.to_pydatetime(), forecast_hour)

    rows = []
    for s in stations:
        tmpf = _read_temp_pixel(url, s.lat, s.lon, timeout=timeout)
        rows.append(
            {
                "station": s.icao.upper(),
                "init_dt": init_dt,
                "forecast_hour": forecast_hour,
                "valid_dt": _valid_dt(init_dt, forecast_hour),
                "lat": s.lat,
                "lon": s.lon,
                "tmpf": tmpf,
            }
        )

    df = pd.DataFrame(rows, columns=NBM_OUTPUT_COLUMNS)
    if cache_path is not None and not df.empty:
        from pathlib import Path

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def fetch_target_nbm_2m_temp(
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    cache_path: Optional[str] = None,
    timeout: float = 90.0,
) -> Optional[pd.Series]:
    """Convenience wrapper returning the NBM 2 m temp row for the target station."""
    df = fetch_latest_nbm_2m_temp(
        stations=[Station(TARGET_ICAO, 0.0, 0.0, 0.0, "target")],
        forecast_hour=forecast_hour,
        lookback_hours=lookback_hours,
        cache_path=cache_path,
        timeout=timeout,
    )
    if df.empty:
        return None
    return df.iloc[0]


def _cycle_has_all_frames(
    init_dt: pd.Timestamp, max_forecast_hour: int, timeout: float = 15.0
) -> bool:
    """Return True iff the COG files for f01..max_forecast_hour all exist."""
    for fh in range(1, max_forecast_hour + 1):
        url = _cog_url(init_dt.to_pydatetime(), fh)
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            if r.status_code != 200:
                return False
        except Exception:
            return False
    return True


def fetch_nbm_forecast_range(
    stations: Iterable[Station],
    max_forecast_hour: int = 18,
    lookback_hours: int = 6,
    timeout: float = 90.0,
) -> pd.DataFrame:
    """Fetch a complete NBM run (f01..max_forecast_hour) for each station.

    Walks back through recent cycles and uses the most recent one whose first
    max_forecast_hour frames are all published.  Every returned row then belongs
    to the same model run.
    """
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC")
    # NBM has a ~2 hour publication lag. Start looking 2 hours back.
    init_dt = None
    for i in range(lookback_hours + 1):
        candidate = (now - pd.Timedelta(hours=2 + i)).floor("h")
        if _cycle_has_all_frames(candidate, max_forecast_hour, timeout=timeout):
            init_dt = candidate
            break

    if init_dt is None:
        return pd.DataFrame(columns=NBM_OUTPUT_COLUMNS)

    all_rows: list[dict] = []
    for fh in range(1, max_forecast_hour + 1):
        url = _cog_url(init_dt.to_pydatetime(), fh)
        for s in stations:
            tmpf = _read_temp_pixel(url, s.lat, s.lon, timeout=timeout)
            all_rows.append(
                {
                    "station": s.icao.upper(),
                    "init_dt": init_dt,
                    "forecast_hour": fh,
                    "valid_dt": _valid_dt(init_dt, fh),
                    "lat": s.lat,
                    "lon": s.lon,
                    "tmpf": tmpf,
                }
            )

    return pd.DataFrame(all_rows, columns=NBM_OUTPUT_COLUMNS)