"""Lightweight HRRR 2-m temperature fetcher from NOAA Open Data on AWS.

Uses only cfgrib + requests. Downloads a tiny byte range from the GRIB2 file
via the .idx index, opens the subset, and extracts the nearest-grid-point 2 m
temperature for each configured station.
"""
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from dfw_temp_model.config import Station, TARGET_ICAO

HRRR_OUTPUT_COLUMNS = [
    "station",
    "init_dt",
    "forecast_hour",
    "valid_dt",
    "lat",
    "lon",
    "tmpf",
]

# NOAA HRRR AWS Open Data public bucket (HTTPS endpoint).
HRRR_BASE_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def _grib_url(init_dt: datetime, forecast_hour: int) -> str:
    """Build the HRRR GRIB2 URL for a given cycle and forecast hour."""
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")
    ff = f"{forecast_hour:02d}"
    return f"{HRRR_BASE_URL}/hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{ff}.grib2"


def _valid_dt(init_dt: pd.Timestamp, forecast_hour: int) -> pd.Timestamp:
    """Return valid time = init_dt + forecast_hour hours."""
    return init_dt + pd.Timedelta(hours=forecast_hour)


def _find_latest_cycle(
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    now: Optional[datetime] = None,
    timeout: float = 15.0,
) -> tuple[pd.Timestamp, bool]:
    """Find the most recent published HRRR cycle by checking .idx files.

    Returns (init_dt, found). Init_dt is floored to the hour in UTC.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Convert to UTC Timestamp safely whether input is naive or tz-aware.
    ts_now = pd.Timestamp(now)
    if ts_now.tz is None:
        ts_now = ts_now.tz_localize("UTC")
    else:
        ts_now = ts_now.tz_convert("UTC")

    for i in range(lookback_hours + 1):
        init = ts_now - pd.Timedelta(hours=i)
        init = init.floor("h")
        idx_url = f"{_grib_url(init.to_pydatetime(), forecast_hour)}.idx"
        try:
            r = requests.head(idx_url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                return init, True
        except Exception:
            pass
        time.sleep(0.2)

    return ts_now.floor("h"), False


def _cfgrib_subset(grib_url: str, timeout: float = 60.0):
    """Download only the 2 m temperature GRIB message and open with cfgrib.

    Parses the .idx file to find the byte range for the first
    ``TMP:2 m above ground`` message, requests that range, writes it to a temp
    file, and returns an xarray Dataset.
    """
    idx_url = f"{grib_url}.idx"
    idx_text = requests.get(idx_url, timeout=timeout).text

    # idx lines: 1:0:d=2026061700:REFC:entire atmosphere:1 hour fcst:
    # We want the specific 2 m temperature message.
    pattern = re.compile(
        r"^(\d+):(\d+):d=(\d+):TMP:2 m above ground:(.+)$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(idx_text))
    if not matches:
        raise ValueError("Could not find TMP:2 m above ground in HRRR index")

    first = matches[0]
    start_byte = int(first.group(2))
    end_byte = ""
    if len(matches) > 1:
        end_byte = int(matches[1].group(2)) - 1

    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    r = requests.get(grib_url, headers=headers, timeout=timeout)
    r.raise_for_status()

    tmp_path = Path(tempfile.gettempdir()) / "hrrr_t2m_subset.grib2"
    tmp_path.write_bytes(r.content)

    import xarray as xr

    ds = xr.open_dataset(
        str(tmp_path),
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2}},
    )
    return ds


def _nearest_temp(ds, lat: float, lon: float) -> float:
    """Extract temperature (K) at the nearest grid point and return °F."""
    lat_arr = ds.latitude.values
    lon_arr = ds.longitude.values

    # Normalize model longitude to [-180, 180] before computing distance.
    lon_arr = ((lon_arr + 180) % 360) - 180
    dlat = lat_arr - lat
    dlon = lon_arr - lon
    dist = dlat**2 + dlon**2
    idx_flat = dist.argmin()

    var_candidates = [v for v in ds.data_vars if v.lower() in ("t2m", "tmp")]
    if not var_candidates:
        var_candidates = list(ds.data_vars)
    var_name = var_candidates[0]
    values = ds[var_name].values
    k = float(values.ravel()[idx_flat])
    return k * 9.0 / 5.0 - 459.67


def fetch_latest_hrrr_2m_temp(
    stations: Iterable[Station],
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    cache_path: Optional[str] = None,
    timeout: float = 90.0,
) -> pd.DataFrame:
    """Fetch the latest HRRR 2 m temperature forecast for each station.

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
        return pd.DataFrame(columns=HRRR_OUTPUT_COLUMNS)

    url = _grib_url(init_dt.to_pydatetime(), forecast_hour)
    ds = _cfgrib_subset(url, timeout=timeout)

    rows = []
    for s in stations:
        tmpf = _nearest_temp(ds, s.lat, s.lon)
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

    df = pd.DataFrame(rows, columns=HRRR_OUTPUT_COLUMNS)
    if cache_path is not None and not df.empty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def fetch_target_hrrr_2m_temp(
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    cache_path: Optional[str] = None,
    timeout: float = 90.0,
) -> Optional[pd.Series]:
    """Convenience wrapper returning the HRRR 2 m temp row for the target station."""
    df = fetch_latest_hrrr_2m_temp(
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
    """Return True iff the .idx files for f01..max_forecast_hour all exist."""
    for fh in range(1, max_forecast_hour + 1):
        url = f"{_grib_url(init_dt.to_pydatetime(), fh)}.idx"
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            if r.status_code != 200:
                return False
        except Exception:
            return False
    return True


def fetch_hrrr_forecast_range(
    stations: Iterable[Station],
    max_forecast_hour: int = 18,
    lookback_hours: int = 6,
    timeout: float = 90.0,
) -> pd.DataFrame:
    """Fetch a complete HRRR run (f01..max_forecast_hour) for each station.

    Walks back through recent cycles and uses the most recent one whose first
    max_forecast_hour frames are all published.  Every returned row then belongs
    to the same model run.
    """
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("UTC")
    init_dt = None
    for i in range(lookback_hours + 1):
        candidate = (now - pd.Timedelta(hours=i)).floor("h")
        if _cycle_has_all_frames(candidate, max_forecast_hour, timeout=timeout):
            init_dt = candidate
            break

    if init_dt is None:
        return pd.DataFrame(columns=HRRR_OUTPUT_COLUMNS)

    all_rows: list[dict] = []
    for fh in range(1, max_forecast_hour + 1):
        url = _grib_url(init_dt.to_pydatetime(), fh)
        ds = _cfgrib_subset(url, timeout=timeout)
        for s in stations:
            all_rows.append(
                {
                    "station": s.icao.upper(),
                    "init_dt": init_dt,
                    "forecast_hour": fh,
                    "valid_dt": _valid_dt(init_dt, fh),
                    "lat": s.lat,
                    "lon": s.lon,
                    "tmpf": _nearest_temp(ds, s.lat, s.lon),
                }
            )

    return pd.DataFrame(all_rows, columns=HRRR_OUTPUT_COLUMNS)
