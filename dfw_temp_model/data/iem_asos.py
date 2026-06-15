from datetime import datetime
import io
from pathlib import Path
import time
from typing import Iterable, List, Optional

import pandas as pd
import requests

IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

VARIABLES = ["tmpf", "dwpf", "drct", "sknt", "skyc1", "mslp", "p01i"]


def build_iem_url(
    station: str,
    start: str,
    end: str,
    variables: List[str],
) -> str:
    """Build an IEM ASOS CSV request URL.

    Parameters
    ----------
    station : str
        ASOS station identifier (e.g. ``KDFW``).
    start : str
        Inclusive start date as ``YYYY-MM-DD``.
    end : str
        Inclusive end date as ``YYYY-MM-DD``.
    variables : List[str]
        Observed variables to request (e.g. ``tmpf``).

    Returns
    -------
    str
        Fully-qualified request URL.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    params = {
        "station": station,
        "data": ",".join(variables),
        "year1": start_dt.year,
        "month1": start_dt.month,
        "day1": start_dt.day,
        "year2": end_dt.year,
        "month2": end_dt.month,
        "day2": end_dt.day,
        "tz": "UTC",
        "format": "csv",
        "latlon": "yes",
        "direct": "no",
        "report_type": ["1", "2"],
    }
    req = requests.Request("GET", IEM_BASE, params=params)
    return req.prepare().url


def fetch_asos_csv(
    station: str,
    start: str,
    end: str,
    variables: Optional[List[str]] = None,
    delay: float = 0.5,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Fetch hourly ASOS observations from IEM for a single station.

    Parameters
    ----------
    station : str
        ASOS station identifier.
    start : str
        Inclusive start date ``YYYY-MM-DD``.
    end : str
        Inclusive end date ``YYYY-MM-DD``.
    variables : Optional[List[str]]
        Variables to request; defaults to :data:`VARIABLES`.
    delay : float
        Seconds to sleep after a successful request to be polite to IEM.
    retries : int
        Number of times to retry on transient HTTP errors.
    backoff : float
        Base multiplier for exponential backoff between retries.

    Returns
    -------
    pd.DataFrame
        Hourly observations with a ``station`` column added.
    """
    variables = variables if variables is not None else VARIABLES
    url = build_iem_url(station, start, end, variables)

    last_response: Optional[requests.Response] = None
    for attempt in range(retries):
        last_response = requests.get(url, timeout=60)
        if last_response.status_code == 429 and attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
            continue
        last_response.raise_for_status()
        break
    else:
        # Ran out of retries: re-raise the last response's HTTP status.
        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError(f"failed to fetch {station}")

    response = last_response

    text = response.text
    # IEM prepends DEBUG comment lines that pandas mis-interprets as headers.
    text = "\n".join(line for line in text.splitlines() if not line.startswith("#DEBUG"))
    df = pd.read_csv(
        io.StringIO(text),
        parse_dates=["valid"],
        na_values=["M"],
    )
    df["station"] = station

    if delay > 0:
        time.sleep(delay)
    return df


def fetch_all_stations(
    start: str,
    end: str,
    stations: Iterable,
    cache_path: Optional[str] = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch hourly ASOS observations for multiple stations and optionally cache them.

    Parameters
    ----------
    start : str
        Inclusive start date ``YYYY-MM-DD``.
    end : str
        Inclusive end date ``YYYY-MM-DD``.
    stations : Iterable
        Iterable of :class:`~dfw_temp_model.config.Station` objects.
    cache_path : Optional[str]
        If provided, the concatenated DataFrame is written to this Parquet path.
    delay : float
        Seconds to sleep between station requests.

    Returns
    -------
    pd.DataFrame
        Concatenated hourly observations for all stations.
    """
    dfs: List[pd.DataFrame] = []
    for station in stations:
        # Disable per-request delay in batch mode; politeness is handled here.
        df = fetch_asos_csv(station.icao, start, end, delay=0.0)
        dfs.append(df)
        if delay > 0:
            time.sleep(delay)

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if cache_path is not None and not combined.empty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path, index=False)

    return combined
