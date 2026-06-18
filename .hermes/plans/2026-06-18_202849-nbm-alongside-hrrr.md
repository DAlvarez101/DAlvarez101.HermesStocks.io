# Add NBM Alongside HRRR as Second Model Data Source

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add NBM (National Blend of Models) as a second model data source alongside the existing HRRR pipeline, so the dashboard shows both models. NBM provides a more accurate (already bias-corrected by NOAA) but ~2h-delayed forecast; HRRR provides a faster-updating but raw forecast. The blending/bias-correction engine runs independently for each provider, and the dashboard displays both side by side with a dropdown to switch between model runs.

**Architecture:** The current HRRR pipeline downloads GRIB2 files from `noaa-hrrr-bdp-pds.s3.amazonaws.com`, parses them with cfgrib, extracts nearest-grid-point 2m temperature, stores rows in the `hrrr_forecasts` SQLite table with `source='hrrr-aws'`, and feeds them through a `HRRRProvider` into the blending/bias-correction engine. The NBM addition mirrors this exactly but reads Cloud Optimized GeoTIFF (COG) files from `noaa-nbm-pds.s3.amazonaws.com` using `rasterio` (which supports HTTP range requests on COG files -- no full download needed). NBM temp files store temperature directly in Fahrenheit as int16. NBM rows go into the same `hrrr_forecasts` table with `source='nbm-aws'`. A new `NBMProvider` class feeds NBM data through the same blending pipeline. The dashboard gets a second chart section for NBM and a model dropdown in the blended forecast chart that lets users switch between HRRR and NBM cycles.

**Tech Stack:** Python 3.13, rasterio (new dependency, already installed in venv), requests, pandas, SQLite, plotly, matplotlib, pytest

---

## Current Context

### What NBM data looks like (verified live)

NBM v5.0 CONUS temperature COG files are on AWS S3 at:
```
https://noaa-nbm-pds.s3.amazonaws.com/blendv5.0/conus/{YYYY}/{MM}/{DD}/{HH}00/temp/blendv5.0_conus_temp_{init_iso}_{valid_iso}.tif
```

Example URL:
```
https://noaa-nbm-pds.s3.amazonaws.com/blendv5.0/conus/2026/06/18/1800/temp/blendv5.0_conus_temp_2026-06-18T18:00_2026-06-18T19:00.tif
```

Key facts verified through live API calls:
- **Update frequency**: Every 1 hour (20 cycles available today: 0000-1900)
- **Resolution**: 2.5 km Lambert Conformal Conic grid, 2345x1597 pixels
- **Data format**: int16 GeoTIFF (COG), temperature in **Fahrenheit** directly (no conversion needed)
- **Nodata value**: -9999
- **Forecast hours**: 108 per cycle (48 hourly, then 48 3-hourly, then 12 6-hourly up to 264h)
- **For our use case**: We need f01-f18 (18 hourly files), same as HRRR
- **HTTP range requests work**: rasterio opens the URL and reads just the target pixel window (~1.25s per file)
- **Cycle availability**: ~2-hour lag (at 20:17Z, the 1800Z cycle was the latest with f01-f18 available; 1900Z and 2000Z were not ready yet)
- **File naming**: `blendv5.0_conus_temp_{init}T{HH}:00_{valid}T{HH}:00.tif` where init and valid are ISO timestamps

### Existing HRRR pipeline structure (kept as-is, NBM added alongside)

| Component | HRRR (unchanged) | NBM (new addition) |
|-----------|-------------------|---------------------|
| Data fetcher | `dfw_temp_model/data/hrrr.py` | `dfw_temp_model/data/nbm.py` (new) |
| DB storage | `obs_db.py` table `hrrr_forecasts`, `source='hrrr-aws'` | Same table, `source='nbm-aws'` |
| Provider | `blending/providers.py` `HRRRProvider` | `NBMProvider` (new class in same file) |
| Ingest script | `scripts/ingest_live_metars.py` `--hrrr` flag | `--nbm` flag (new, alongside `--hrrr`) |
| Dashboard | HRRR chart + HRRR in blended chart | NBM chart added + NBM in blended chart dropdown |
| Cron script | `scripts/cron_update_dashboard.sh` `--hrrr` | Both `--hrrr` and `--nbm` |
| Tests | Existing HRRR tests unchanged | New NBM tests added |

### Key design decisions

1. **Both models coexist** -- HRRR stays fully functional. NBM is added as a second source. The `hrrr_forecasts` DB table already has a `source` column and stores both models. HRRRProvider filters by `source='hrrr-aws'`, NBMProvider filters by `source='nbm-aws'`.

2. **NBMProvider mirrors HRRRProvider** -- same `ForecastProvider` protocol, same DB table, just different `source` filter. The blending/bias-correction engine (`blend.py`, `bias.py`) is already provider-agnostic and needs no changes.

3. **18 forecast hours** -- same as HRRR. NBM has 48 hourly hours available, but we only use 18 to match the existing dashboard and blending configuration.

4. **rasterio for COG reading** -- already installed in the venv. Uses HTTP range requests to read just the target pixel, no full file download.

5. **Dashboard shows both models** -- the existing HRRR chart stays. A new NBM chart section is added. The blended forecast chart gets a richer dropdown: it shows cycles from both HRRR and NBM, with the model name in the dropdown label (e.g., "NBM 06/18 18:00Z", "HRRR 06/18 19:00Z").

6. **No existing code is commented out or removed** -- this is purely additive.

---

## Step-by-Step Plan

### Task 1: Add rasterio to pyproject.toml dependencies

**Objective:** Make rasterio a declared project dependency so `uv sync` installs it.

**Files:**
- Modify: `pyproject.toml:6-25` (dependencies list)

**Step 1: Edit pyproject.toml**

Add `rasterio>=1.5` to the dependencies list. Insert after the `requests` line:

```toml
    "requests>=2.31",
    "rasterio>=1.5",
```

**Step 2: Verify rasterio is in the venv**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "import rasterio; print(rasterio.__version__)"`
Expected: `1.5.0` (already installed)

**Step 3: Verify uv lock is consistent**

Run: `cd /opt/data/stock-research/dfw_temp_model && uv lock --check 2>&1 || uv lock`
Expected: lock file updated or already consistent

**Step 4: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add pyproject.toml uv.lock
git commit -m "deps: add rasterio for NBM COG file reading"
```

---

### Task 2: Create the NBM data fetcher module

**Objective:** Create `dfw_temp_model/data/nbm.py` that mirrors `data/hrrr.py` but reads NBM COG files from AWS S3 using rasterio.

**Files:**
- Create: `dfw_temp_model/data/nbm.py`

**Step 1: Write the module**

```python
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
```

**Step 2: Verify the module imports**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "from dfw_temp_model.data.nbm import _cog_url, fetch_nbm_forecast_range; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dfw_temp_model/data/nbm.py
git commit -m "feat: add NBM COG data fetcher module"
```

---

### Task 3: Write tests for the NBM fetcher

**Objective:** Create `tests/test_nbm.py` with unit tests for URL building, cycle finding, and a network smoke test.

**Files:**
- Create: `tests/test_nbm.py`

**Step 1: Write the tests**

```python
"""Tests for NBM 2-m temperature fetcher."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS, TARGET_ICAO
from dfw_temp_model.data.nbm import (
    _cog_url,
    _find_latest_cycle,
    _valid_dt,
    fetch_latest_nbm_2m_temp,
    fetch_nbm_forecast_range,
)


def test_cog_url_format():
    """COG URL follows the expected NBM S3 path pattern."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC").to_pydatetime()
    url = _cog_url(init, forecast_hour=1)
    assert url.startswith("https://noaa-nbm-pds.s3.amazonaws.com/")
    assert "blendv5.0/conus/2026/06/18/1800/temp/" in url
    assert "blendv5.0_conus_temp_2026-06-18T18:00_2026-06-18T19:00.tif" in url


def test_cog_url_forecast_hour_18():
    """Forecast hour 18 produces the correct valid time in the URL."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC").to_pydatetime()
    url = _cog_url(init, forecast_hour=18)
    assert "blendv5.0_conus_temp_2026-06-18T18:00_2026-06-19T12:00.tif" in url


def test_valid_dt_calculation():
    """valid_dt = init_dt + forecast_hour hours."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC")
    assert _valid_dt(init, 1) == pd.Timestamp("2026-06-18T19:00:00", tz="UTC")
    assert _valid_dt(init, 18) == pd.Timestamp("2026-06-19T12:00:00", tz="UTC")


def test_find_latest_cycle_returns_utc_timestamp():
    """_find_latest_cycle returns a tz-aware Timestamp and found=False when no cycle exists."""
    init, found = _find_latest_cycle(
        forecast_hour=1,
        lookback_hours=0,
        now=datetime(1990, 1, 1, tzinfo=timezone.utc),
    )
    assert isinstance(init, pd.Timestamp)
    assert init.tz is not None
    assert found is False


def test_fetch_latest_nbm_2m_temp_empty_when_no_cycle(mocker):
    """Returns empty DataFrame with correct columns when no cycle is found."""
    mocker.patch(
        "dfw_temp_model.data.nbm._find_latest_cycle",
        return_value=(pd.Timestamp("2026-06-18T18:00:00", tz="UTC"), False),
    )
    df = fetch_latest_nbm_2m_temp(STATIONS, forecast_hour=1, lookback_hours=0)
    assert df.empty
    assert list(df.columns) == [
        "station", "init_dt", "forecast_hour", "valid_dt", "lat", "lon", "tmpf",
    ]


def test_fetch_latest_nbm_2m_temp_with_mocked_pixel(mocker):
    """Returns one row per station with correct fields when a cycle is found."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC")
    mocker.patch(
        "dfw_temp_model.data.nbm._find_latest_cycle",
        return_value=(init, True),
    )
    mocker.patch(
        "dfw_temp_model.data.nbm._read_temp_pixel",
        return_value=85.0,
    )
    df = fetch_latest_nbm_2m_temp(STATIONS[:1], forecast_hour=1, lookback_hours=0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["station"] == "KDFW"
    assert row["init_dt"] == init
    assert row["forecast_hour"] == 1
    assert row["valid_dt"] == pd.Timestamp("2026-06-18T19:00:00", tz="UTC")
    assert row["tmpf"] == 85.0


@pytest.mark.network
@pytest.mark.slow
def test_fetch_nbm_forecast_range_smoke():
    """Real network smoke test: fetch 3 NBM forecast hours for KDAL."""
    target = [s for s in STATIONS if s.icao == TARGET_ICAO][0]
    df = fetch_nbm_forecast_range(
        stations=[target],
        max_forecast_hour=3,
        lookback_hours=6,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "NBM fetch returned empty DataFrame (no cycle found)"
    assert TARGET_ICAO in df["station"].values
    assert len(df) == 3  # 3 forecast hours
    row = df.iloc[0]
    assert 20.0 <= row["tmpf"] <= 120.0
    assert pd.notna(row["init_dt"])
    assert row["forecast_hour"] == 1
    assert pd.notna(row["valid_dt"])
    assert row["valid_dt"] == row["init_dt"] + pd.Timedelta(hours=1)
```

**Step 2: Run tests to verify they pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_nbm.py -v -k "not network and not slow"`
Expected: 6 passed (the smoke test is skipped)

**Step 3: Run the network smoke test**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_nbm.py::test_fetch_nbm_forecast_range_smoke -v --timeout=120`
Expected: PASS -- real NBM data fetched from AWS S3

**Step 4: Commit**

```bash
git add tests/test_nbm.py
git commit -m "test: add NBM fetcher unit + smoke tests"
```

---

### Task 4: Add NBMProvider to the blending providers module

**Objective:** Add an `NBMProvider` class to `blending/providers.py` that implements the `ForecastProvider` protocol, reading from the same `hrrr_forecasts` table but filtering by `source = 'nbm-aws'`.

**Files:**
- Modify: `dfw_temp_model/blending/providers.py`
- Modify: `tests/test_blending_providers.py`

**Step 1: Write failing tests**

Add to `tests/test_blending_providers.py` (add `NBMProvider` to the import line too):

```python
from dfw_temp_model.blending.providers import ForecastProvider, HRRRProvider, NBMProvider


def test_nbm_provider_returns_forecast():
    """NBMProvider reads from the SQLite DB filtered by source='nbm-aws'."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station)
        );
    """)
    conn.executemany(
        "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "2026-01-01T00:00:00Z", "nbm-aws", "KDAL",
             "2026-01-01T00:00:00Z", 1, "2026-01-01T01:00:00Z", 32.0, -96.0, 80.0),
            (2, "2026-01-01T00:00:00Z", "hrrr-aws", "KDAL",
             "2026-01-01T00:00:00Z", 1, "2026-01-01T01:00:00Z", 32.0, -96.0, 82.0),
        ],
    )
    conn.commit()

    provider = NBMProvider()
    df = provider.fetch_forecast(conn, "KDAL", "2026-01-01T00:00:00Z", forecast_hours=2)
    # Should only return the nbm-aws row, not the hrrr-aws row
    assert len(df) == 1
    assert df.iloc[0]["source"] == "nbm-aws"
    assert "valid_dt" in df.columns
    conn.close()


def test_nbm_provider_recent_cycles():
    """recent_cycles returns init_dt strings sorted newest first, filtered by source."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station)
        );
    """)
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "nbm-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    for fh in range(1, 9):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "hrrr-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    conn.commit()

    provider = NBMProvider()
    cycles = provider.recent_cycles(conn, "KDAL", min_hours=18)
    assert "2026-01-01T12:00:00Z" in cycles
    conn.close()


def test_hrrr_and_nbm_providers_are_independent():
    """HRRR and NBM providers return disjoint sets when both sources exist."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station)
        );
    """)
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "hrrr-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "nbm-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 82),
        )
    conn.commit()

    hrrr_provider = HRRRProvider()
    nbm_provider = NBMProvider()
    hrrr_df = hrrr_provider.fetch_forecast(conn, "KDAL", "2026-01-01T12:00:00Z")
    nbm_df = nbm_provider.fetch_forecast(conn, "KDAL", "2026-01-01T12:00:00Z")
    assert len(hrrr_df) == 18
    assert len(nbm_df) == 18
    assert hrrr_df["tmpf"].iloc[0] == 80.0
    assert nbm_df["tmpf"].iloc[0] == 82.0
    conn.close()
```

**Step 2: Run tests to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_providers.py::test_nbm_provider_returns_forecast -v`
Expected: FAIL -- `NBMProvider` not defined

**Step 3: Add NBMProvider to providers.py**

Add the following class to `dfw_temp_model/blending/providers.py`, after the `HRRRProvider` class:

```python
class NBMProvider:
    """Reads NBM forecasts from the SQLite ``hrrr_forecasts`` table.

    NBM (National Blend of Models) forecasts are fetched from AWS S3 COG
    files by ``dfw_temp_model.data.nbm`` and stored in the same table with
    ``source = 'nbm-aws'``. This provider filters by that source tag so the
    blending logic can use NBM independently of HRRR.
    """

    SOURCE = "nbm-aws"

    def fetch_forecast(
        self,
        conn: sqlite3.Connection,
        station: str,
        init_dt: str,
        forecast_hours: int = 18,
    ) -> pd.DataFrame:
        """Return all forecast hours for a given station and init cycle."""
        df = pd.read_sql_query(
            """
            SELECT init_dt, forecast_hour, valid_dt, tmpf, lat, lon, station, source
            FROM hrrr_forecasts
            WHERE station = ? AND init_dt = ? AND source = ?
            ORDER BY forecast_hour ASC
            """,
            conn,
            params=[station, init_dt, self.SOURCE],
        )
        return df

    def recent_cycles(
        self,
        conn: sqlite3.Connection,
        station: str,
        min_hours: int = 18,
    ) -> list[str]:
        """Return init_dt strings that have at least min_hours of frames.

        Sorted newest-first. Only includes complete cycles for this source.
        """
        df = pd.read_sql_query(
            """
            SELECT init_dt, COUNT(*) AS n
            FROM hrrr_forecasts
            WHERE station = ? AND source = ?
            GROUP BY init_dt
            HAVING n >= ?
            ORDER BY init_dt DESC
            """,
            conn,
            params=[station, self.SOURCE, min_hours],
        )
        if df.empty:
            return []
        return df["init_dt"].tolist()
```

**Step 4: Run tests to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_providers.py -v -k "not network and not slow"`
Expected: All pass including the new NBM tests and the independence test

**Step 5: Commit**

```bash
git add dfw_temp_model/blending/providers.py tests/test_blending_providers.py
git commit -m "feat: add NBMProvider for blending pipeline"
```

---

### Task 5: Add source filtering to obs_db query functions

**Objective:** Add an optional `source` parameter to `latest_complete_hrrr_cycle` and `hrrr_forecast_for_cycle` in `obs_db.py` so the dashboard can query NBM-specific or HRRR-specific cycles independently.

**Files:**
- Modify: `dfw_temp_model/storage/obs_db.py`
- Modify: `tests/test_obs_db.py`

**Step 1: Write failing test**

Add to `tests/test_obs_db.py` (add `latest_complete_hrrr_cycle` and `hrrr_forecast_for_cycle` to the imports if not already there):

```python
from dfw_temp_model.storage.obs_db import (
    ensure_schema,
    get_db,
    hrrr_forecast_for_cycle,
    insert_hrrr_forecasts,
    insert_observations,
    latest_by_station,
    latest_complete_hrrr_cycle,
    read_all,
    row_count,
    station_count,
    time_range,
)


def test_latest_complete_hrrr_cycle_filters_by_source(empty_conn):
    """latest_complete_hrrr_cycle with source param filters correctly."""
    # Insert 18 nbm-aws rows for one cycle
    for fh in range(1, 19):
        empty_conn.execute(
            "INSERT INTO hrrr_forecasts (fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf) "
            "VALUES ('t', 'nbm-aws', 'KDAL', '2026-01-01T12:00:00Z', ?, 't', 0, 0, 80)",
            (fh,),
        )
    # Insert 5 hrrr-aws rows for the same cycle (incomplete)
    for fh in range(1, 6):
        empty_conn.execute(
            "INSERT INTO hrrr_forecasts (fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf) "
            "VALUES ('t', 'hrrr-aws', 'KDAL', '2026-01-01T12:00:00Z', ?, 't', 0, 0, 80)",
            (fh,),
        )
    empty_conn.commit()

    # Without source filter: finds the cycle (23 total rows >= 18)
    assert latest_complete_hrrr_cycle(empty_conn, "KDAL") == "2026-01-01T12:00:00Z"
    # With source='nbm-aws': finds the cycle (18 rows >= 18)
    assert latest_complete_hrrr_cycle(empty_conn, "KDAL", source="nbm-aws") == "2026-01-01T12:00:00Z"
    # With source='hrrr-aws': does NOT find the cycle (5 rows < 18)
    assert latest_complete_hrrr_cycle(empty_conn, "KDAL", source="hrrr-aws") is None


def test_hrrr_forecast_for_cycle_filters_by_source(empty_conn):
    """hrrr_forecast_for_cycle with source param filters correctly."""
    for fh in range(1, 4):
        empty_conn.execute(
            "INSERT INTO hrrr_forecasts (fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf) "
            "VALUES ('t', 'nbm-aws', 'KDAL', '2026-01-01T12:00:00Z', ?, 't', 0, 0, 80)",
            (fh,),
        )
    for fh in range(1, 4):
        empty_conn.execute(
            "INSERT INTO hrrr_forecasts (fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf) "
            "VALUES ('t', 'hrrr-aws', 'KDAL', '2026-01-01T12:00:00Z', ?, 't', 0, 0, 82)",
            (fh,),
        )
    empty_conn.commit()

    # Without source: returns all 6 rows
    all_df = hrrr_forecast_for_cycle(empty_conn, "KDAL", "2026-01-01T12:00:00Z")
    assert len(all_df) == 6
    # With source='nbm-aws': returns only 3 rows
    nbm_df = hrrr_forecast_for_cycle(empty_conn, "KDAL", "2026-01-01T12:00:00Z", source="nbm-aws")
    assert len(nbm_df) == 3
    assert (nbm_df["source"] == "nbm-aws").all()
    assert nbm_df["tmpf"].iloc[0] == 80.0
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_obs_db.py::test_latest_complete_hrrr_cycle_filters_by_source -v`
Expected: FAIL -- `source` parameter not accepted

**Step 3: Add source parameter to obs_db functions**

In `dfw_temp_model/storage/obs_db.py`, update `latest_complete_hrrr_cycle`:

```python
def latest_complete_hrrr_cycle(
    conn: sqlite3.Connection, station: str, required_hours: int = 18,
    source: Optional[str] = None,
) -> Optional[str]:
    """Return the latest init_dt (ISO string) that has >= required_hours frames.

    If source is given, only counts rows with that source value.
    """
    if source:
        df = pd.read_sql_query(
            """
            SELECT init_dt, COUNT(*) AS n
            FROM hrrr_forecasts
            WHERE station = ? AND source = ?
            GROUP BY init_dt
            HAVING n >= ?
            ORDER BY init_dt DESC
            LIMIT 1
            """,
            conn,
            params=[station, source, required_hours],
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT init_dt, COUNT(*) AS n
            FROM hrrr_forecasts
            WHERE station = ?
            GROUP BY init_dt
            HAVING n >= ?
            ORDER BY init_dt DESC
            LIMIT 1
            """,
            conn,
            params=[station, required_hours],
        )
    if df.empty:
        return None
    return str(df.iloc[0]["init_dt"])
```

Similarly update `hrrr_forecast_for_cycle`:

```python
def hrrr_forecast_for_cycle(
    conn: sqlite3.Connection, station: str, init_dt: str,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Return every forecast hour for a given station and model cycle.

    If source is given, filters by that source value.
    """
    if source:
        return pd.read_sql_query(
            """
            SELECT * FROM hrrr_forecasts
            WHERE station = ? AND init_dt = ? AND source = ?
            ORDER BY forecast_hour ASC
            """,
            conn,
            params=[station, init_dt, source],
        )
    return pd.read_sql_query(
        """
        SELECT * FROM hrrr_forecasts
        WHERE station = ? AND init_dt = ?
        ORDER BY forecast_hour ASC
        """,
        conn,
        params=[station, init_dt],
    )
```

**Step 4: Run tests to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_obs_db.py -v`
Expected: All pass including new source-filter tests

**Step 5: Commit**

```bash
git add dfw_temp_model/storage/obs_db.py tests/test_obs_db.py
git commit -m "feat: add source filtering to obs_db query functions"
```

---

### Task 6: Add NBM ingestion to the ingest script

**Objective:** Update `scripts/ingest_live_metars.py` to support both `--hrrr` and `--nbm` flags. Both can be passed simultaneously to ingest both models. The existing `--hrrr` code path stays unchanged; a new `--nbm` code path is added after it.

**Files:**
- Modify: `scripts/ingest_live_metars.py`

**Step 1: Edit the script**

Add the NBM import at the top (keep the HRRR import):

```python
from dfw_temp_model.data.hrrr import fetch_hrrr_forecast_range, fetch_latest_hrrr_2m_temp
from dfw_temp_model.data.nbm import fetch_nbm_forecast_range
```

Add new argparse arguments after the existing `--hrrr-hours` argument:

```python
    parser.add_argument(
        "--nbm",
        action="store_true",
        help="Also fetch NBM 2m temperature forecast for the next 18 hours",
    )
    parser.add_argument(
        "--nbm-hours",
        type=int,
        default=18,
        help="Number of NBM forecast hours to fetch (default 18)",
    )
```

Add the NBM fetch block after the existing HRRR block (do NOT modify or comment out the HRRR block):

```python
    if args.nbm:
        print(f"[{fetched_at}] Fetching NBM 2m temp (f01-f{args.nbm_hours:02d}) ...")
        nbm_df = fetch_nbm_forecast_range(
            STATIONS, max_forecast_hour=args.nbm_hours, lookback_hours=6
        )
        if nbm_df.empty:
            print("No NBM forecast returned.", file=sys.stderr)
        else:
            nbm_inserted = insert_hrrr_forecasts(
                conn, nbm_df, source="nbm-aws", fetched_at=fetched_at
            )
            nbm_total = conn.execute(
                "SELECT COUNT(*) FROM hrrr_forecasts WHERE source = 'nbm-aws'"
            ).fetchone()[0]
            print(
                f"Inserted {nbm_inserted} NBM rows "
                f"({len(nbm_df)} fetched across {nbm_df['forecast_hour'].nunique()} hours). "
                f"Total NBM rows: {nbm_total}"
            )
```

**Step 2: Verify the script runs with just METARs (no model flags)**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/ingest_live_metars.py --db /tmp/test_dual_ingest.db --hours 1`
Expected: METAR rows inserted, no errors

**Step 3: Verify with --nbm (network test)**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/ingest_live_metars.py --db /tmp/test_dual_ingest.db --hours 1 --nbm --nbm-hours 3`
Expected: METAR rows + NBM rows inserted. Output should say "Inserted N NBM rows"

**Step 4: Verify with both --hrrr and --nbm (network test)**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/ingest_live_metars.py --db /tmp/test_dual_ingest.db --hours 1 --hrrr --hrrr-hours 3 --nbm --nbm-hours 3`
Expected: METAR rows + HRRR rows + NBM rows inserted. Both model summaries in output.

**Step 5: Clean up**

Run: `rm -f /tmp/test_dual_ingest.db`

**Step 6: Commit**

```bash
git add scripts/ingest_live_metars.py
git commit -m "feat: add --nbm flag to ingest script alongside --hrrr"
```

---

### Task 7: Add NBM chart and multi-model dropdown to the dashboard

**Objective:** Update `scripts/generate_dashboard.py` to add an NBM forecast chart (alongside the existing HRRR chart) and update the blended forecast chart to show cycles from both HRRR and NBM models with model name in the dropdown labels.

**Files:**
- Modify: `scripts/generate_dashboard.py`

**Step 1: Update the HTML template**

In `HTML_TEMPLATE`, add an NBM chart section after the HRRR chart section. Change:

```html
    <h2>HRRR 18-hour forecast ({TARGET_ICAO})</h2>
    {hrrr_chart}

    <h2>Bias-Corrected Forecast ({TARGET_ICAO})</h2>
    {blended_chart}
```

to:

```html
    <h2>HRRR 18-hour forecast ({TARGET_ICAO})</h2>
    {hrrr_chart}

    <h2>NBM 18-hour forecast ({TARGET_ICAO})</h2>
    {nbm_chart}

    <h2>Bias-Corrected Forecast ({TARGET_ICAO})</h2>
    {blended_chart}
```

Also update the source line to include NBM:

```html
    <p>Updated at {updated_at} UTC · {updated_at_ct} CT<br>Sources: AviationWeather.gov METAR JSON, NOAA HRRR + NBM AWS Open Data</p>
```

**Step 2: Add NBM chart function**

Add a new function `nbm_forecast_chart` that mirrors `hrrr_forecast_chart` but reads NBM data. Place it right after the `hrrr_forecast_chart` function:

```python
def nbm_forecast_chart(conn) -> str:
    """Interactive Plotly NBM 2 m temperature forecast chart for the target station.

    Uses the latest complete NBM cycle in the DB (filtered by source='nbm-aws').
    """
    init_dt_str = latest_complete_hrrr_cycle(conn, TARGET_ICAO, required_hours=18, source="nbm-aws")
    if init_dt_str is None:
        return "<p>No NBM forecast data yet</p>"

    df = hrrr_forecast_for_cycle(conn, TARGET_ICAO, init_dt_str, source="nbm-aws")
    if df.empty or len(df) < 18:
        return "<p>Complete NBM forecast cycle not available</p>"

    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
    df["init_dt"] = pd.to_datetime(df["init_dt"], utc=True)
    df = df.sort_values("forecast_hour")
    init_label = df["init_dt"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
    init_ct_label = df["init_dt"].iloc[0].tz_convert(_CT).strftime("%I:%M %p CT")

    df["valid_ct"] = df["valid_dt"].apply(lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT"))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["valid_dt"],
                y=df["tmpf"],
                mode="lines+markers",
                name="NBM 2m temp",
                line={"color": "#3b82f6", "width": 2},
                marker={"size": 6, "color": "#3b82f6"},
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.15)",
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M UTC}</b><br>"
                    "%{customdata}<br>"
                    "Temp: %{y:.1f}°F<br>"
                    f"Cycle: {init_label}<br>"
                    "f%{text}<extra></extra>"
                ),
                text=df["forecast_hour"].astype(int),
                customdata=df["valid_ct"],
            )
        ]
    )

    ymin, ymax = df["tmpf"].min(), df["tmpf"].max()
    pad = max(1.0, (ymax - ymin) * 0.15)
    y_min = ymin - pad
    y_max = ymax + pad

    fig.update_layout(
        title=f"NBM 18-hour forecast — {TARGET_ICAO} 2 m temp<br><sup>Cycle {init_label} · {init_ct_label}</sup>",
        xaxis_title="Valid time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        yaxis={"range": [y_min, y_max], "gridcolor": "#334155"},
        xaxis={"gridcolor": "#334155"},
        showlegend=False,
        hovermode="x unified",
    )

    return pyo.plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False})
```

**Step 3: Update `blended_forecast_chart` to show both HRRR and NBM cycles**

The current `blended_forecast_chart` uses `HRRRProvider` only. Update it to query both providers and combine their cycles into the dropdown. Each dropdown button should be labeled with the model name + cycle time.

Replace the beginning of `blended_forecast_chart`:

```python
def blended_forecast_chart(conn) -> str:
    """Interactive Plotly chart: raw vs bias-corrected vs METAR observations.

    Shows cycles from BOTH HRRR and NBM models in the dropdown, with the model
    name in each dropdown label so users can compare models side by side.
    """
    from dfw_temp_model.blending.blend import blended_forecast, list_recent_cycles
    from dfw_temp_model.blending.providers import HRRRProvider, NBMProvider

    hrrr_provider = HRRRProvider()
    nbm_provider = NBMProvider()

    hrrr_cycles = list_recent_cycles(conn, TARGET_ICAO, hrrr_provider, min_hours=18)
    nbm_cycles = list_recent_cycles(conn, TARGET_ICAO, nbm_provider, min_hours=18)

    # Build a combined list of (provider, cycle_dt, model_label) tuples
    # Limit each model to its 3 most recent cycles to keep the dropdown manageable
    all_cycles = []
    for cycle_dt in hrrr_cycles[:3]:
        all_cycles.append((hrrr_provider, cycle_dt, "HRRR"))
    for cycle_dt in nbm_cycles[:3]:
        all_cycles.append((nbm_provider, cycle_dt, "NBM"))

    if not all_cycles:
        return "<p>No complete forecast cycles available for blending (need HRRR or NBM data)</p>"
```

Then update the cycle loop to use the combined list:

```python
    # Load ALL observations for overlay (both 5-minute NWS API and hourly AviationWeather)
    obs_df = pd.read_sql_query(
        "SELECT valid, tmpf, source FROM metar_observations WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn,
        params=[TARGET_ICAO],
    )
    if not obs_df.empty:
        obs_df["valid"] = pd.to_datetime(obs_df["valid"], utc=True)
        obs_df["ct_label"] = obs_df["valid"].apply(
            lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")
        )
        obs_5min = obs_df[obs_df["source"] == "nws-api"].copy()
        obs_hourly = obs_df[obs_df["source"] == "aviationweather"].copy()
    else:
        obs_5min = pd.DataFrame()
        obs_hourly = pd.DataFrame()

    fig = go.Figure()

    # Add 5-minute NWS API observations
    if not obs_5min.empty:
        fig.add_trace(go.Scatter(
            x=obs_5min["valid"],
            y=obs_5min["tmpf"],
            mode="markers",
            name="5-min obs (NWS API)",
            marker={"size": 4, "color": "#818cf8", "symbol": "circle", "opacity": 0.6},
            hovertemplate=(
                "<b>5-min obs</b><br>"
                "%{x|%Y-%m-%d %H:%M UTC}<br>"
                "%{customdata}<br>"
                "Temp: %{y:.1f}°F<extra></extra>"
            ),
            customdata=obs_5min.get("ct_label", ""),
            visible=True,
        ))

    # Add hourly METAR observations
    if not obs_hourly.empty:
        fig.add_trace(go.Scatter(
            x=obs_hourly["valid"],
            y=obs_hourly["tmpf"],
            mode="markers",
            name="METAR observed",
            marker={"size": 8, "color": "#38bdf8", "symbol": "circle"},
            hovertemplate=(
                "<b>METAR</b><br>"
                "%{x|%Y-%m-%d %H:%M UTC}<br>"
                "%{customdata}<br>"
                "Temp: %{y:.1f}°F<extra></extra>"
            ),
            customdata=obs_hourly.get("ct_label", ""),
            visible=True,
        ))

    n_obs_traces = int(not obs_5min.empty) + int(not obs_hourly.empty)
    dropdown_buttons = []

    # Assign colors per model: HRRR = orange (#f59e0b), NBM = blue (#3b82f6)
    model_colors = {"HRRR": "#f59e0b", "NBM": "#3b82f6"}

    for i, (provider, cycle_dt, model_label) in enumerate(all_cycles):
        blended = blended_forecast(conn, TARGET_ICAO, provider, init_dt=cycle_dt, trend_weight=0.15)
        if blended.empty:
            continue

        blended["valid_dt"] = pd.to_datetime(blended["valid_dt"], utc=True)
        blended = blended.sort_values("forecast_hour")
        init_ts = pd.to_datetime(cycle_dt, utc=True)
        init_label = init_ts.strftime("%Y-%m-%d %H:%M UTC")
        init_ct = init_ts.tz_convert(_CT).strftime("%I:%M %p CT")

        ct_labels = blended["valid_dt"].apply(
            lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")
        )

        color = model_colors[model_label]

        # Raw model forecast
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"],
            y=blended["tmpf"],
            mode="lines+markers",
            name=f"{model_label} raw (cycle {i+1})",
            line={"color": color, "width": 2, "dash": "dot"},
            marker={"size": 5, "color": color},
            hovertemplate=(
                f"<b>{model_label} raw</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                f"%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>"
                f"Cycle: {init_label}<extra></extra>"
            ),
            customdata=ct_labels,
            visible=(i == 0),
        ))

        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=list(blended["valid_dt"]) + list(blended["valid_dt"])[::-1],
            y=list(blended["uncertainty_high"]) + list(blended["uncertainty_low"])[::-1],
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.12)",
            line={"color": "rgba(34, 197, 94, 0)", "width": 0},
            name=f"Uncertainty (cycle {i+1})",
            hoverinfo="skip",
            visible=(i == 0),
            showlegend=False,
        ))

        # Bias-corrected
        bias_val = float(blended["bias_applied"].iloc[0]) if "bias_applied" in blended.columns else 0.0
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"],
            y=blended["tmpf_corrected"],
            mode="lines+markers",
            name=f"Corrected (cycle {i+1})",
            line={"color": "#22c55e", "width": 2.5},
            marker={"size": 6, "color": "#22c55e"},
            hovertemplate=(
                f"<b>Corrected ({model_label})</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                f"%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>"
                f"Bias: {bias_val:+.1f}°F<br>"
                f"Cycle: {init_label} · {init_ct}<extra></extra>"
            ),
            customdata=ct_labels,
            visible=(i == 0),
        ))

        # Trend-adjusted
        if "tmpf_trend_adjusted" in blended.columns:
            fig.add_trace(go.Scatter(
                x=blended["valid_dt"],
                y=blended["tmpf_trend_adjusted"],
                mode="lines+markers",
                name=f"Trend-adjusted (cycle {i+1})",
                line={"color": "#a78bfa", "width": 2},
                marker={"size": 5, "color": "#a78bfa"},
                hovertemplate=(
                    f"<b>Trend-adjusted ({model_label})</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                    f"%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>"
                    f"Bias: {bias_val:+.1f}°F + trend<br>"
                    f"Cycle: {init_label} · {init_ct}<extra></extra>"
                ),
                customdata=ct_labels,
                visible=(i == 0),
            ))

        n_traces_per_cycle = 4 if "tmpf_trend_adjusted" in blended.columns else 3
        visibility = [True] * n_obs_traces
        for j in range(len(all_cycles)):
            if j == i:
                visibility.extend([True] * n_traces_per_cycle)
            else:
                visibility.extend([False] * n_traces_per_cycle)

        dropdown_buttons.append(dict(
            label=f"{model_label} {init_ts.strftime('%m/%d %H:00Z')}",
            method="update",
            args=[{"visible": visibility}],
        ))

    if dropdown_buttons:
        fig.update_layout(
            updatemenus=[dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )],
        )

    fig.update_layout(
        title=f"Blended Forecast — {TARGET_ICAO}<br><sup>HRRR (orange) + NBM (blue) · corrected (green) · trend (purple) · 5-min obs (indigo) · METAR (blue)</sup>",
        xaxis_title="Valid time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        hovermode="x unified",
        showlegend=True,
        legend={"x": 0.01, "xanchor": "left", "y": 0.99, "yanchor": "top",
                "bgcolor": "rgba(15,23,42,0.8)", "font": {"size": 10}},
    )

    return pyo.plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False})
```

**Step 4: Update `generate_dashboard` to include the NBM chart**

In the `HTML_TEMPLATE.format(...)` call inside `generate_dashboard()`, add the `nbm_chart` parameter:

```python
    html = HTML_TEMPLATE.format(
        ...
        hrrr_chart=hrrr_forecast_chart(get_db(db_path)),
        nbm_chart=nbm_forecast_chart(get_db(db_path)),
        blended_chart=blended_forecast_chart(get_db(db_path)),
        ...
    )
```

**Step 5: Run the existing dashboard test to see if it still passes**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_generate_dashboard.py -v`
Expected: The test may need updating if it checks for specific chart content. The basic test (HTML exists, KDAL present) should pass. The "METAR vs HRRR" assertion still passes since we didn't change that section.

**Step 6: Commit**

```bash
git add scripts/generate_dashboard.py
git commit -m "feat: add NBM chart and multi-model dropdown to dashboard"
```

---

### Task 8: Update dashboard generation tests for NBM

**Objective:** Update `tests/test_generate_dashboard.py` to include NBM data in the test fixture and verify NBM chart appears in output.

**Files:**
- Modify: `tests/test_generate_dashboard.py`

**Step 1: Update the test fixture**

In `populated_db` fixture, add NBM forecast rows after the HRRR rows:

```python
    # NBM forecast rows (same cycle, different source)
    nbm_rows = []
    for fh in range(1, 19):
        nbm_rows.append(
            {
                "station": "KDAL",
                "init_dt": "2026-06-16T16:00:00+00:00",
                "forecast_hour": fh,
                "valid_dt": (pd.Timestamp("2026-06-16T16:00:00+00:00") + pd.Timedelta(hours=fh)).isoformat(),
                "lat": 32.848,
                "lon": -96.851,
                "tmpf": 84.0 + (fh - 1) * 0.1,
            }
        )
    nbm_df = pd.DataFrame(nbm_rows)
    insert_hrrr_forecasts(conn, nbm_df, source="nbm-aws")
```

**Step 2: Update assertions**

Add assertions for NBM chart presence:

```python
    assert "NBM" in html
    assert "NBM 18-hour forecast" in html
```

**Step 3: Run tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_generate_dashboard.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/test_generate_dashboard.py
git commit -m "test: add NBM data to dashboard test fixture"
```

---

### Task 9: Update the cron script to fetch both HRRR and NBM

**Objective:** Update `scripts/cron_update_dashboard.sh` to pass both `--hrrr` and `--nbm` flags.

**Files:**
- Modify: `scripts/cron_update_dashboard.sh`

**Step 1: Edit the cron script**

Change the ingest line (line 32):

```bash
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK" --hrrr
```

to:

```bash
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK" --hrrr --nbm
```

Also update the echo on line 26:

```bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ingesting observations + HRRR forecast ..."
```

to:

```bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ingesting observations + HRRR + NBM forecasts ..."
```

And update the comment on line 29:

```bash
# AviationWeather METAR + HRRR forecast (HRRR is hourly-only; AviationWeather serves as cross-validation)
```

to:

```bash
# AviationWeather METAR + HRRR + NBM forecasts (both models update hourly; AviationWeather serves as cross-validation)
```

**Step 2: Verify the script syntax**

Run: `bash -n /opt/data/stock-research/dfw_temp_model/scripts/cron_update_dashboard.sh`
Expected: No syntax errors

**Step 3: Commit**

```bash
git add scripts/cron_update_dashboard.sh
git commit -m "feat: cron script now fetches both HRRR and NBM"
```

---

### Task 10: Run the full test suite

**Objective:** Verify all tests pass after adding NBM alongside HRRR.

**Step 1: Run all tests (excluding network/slow)**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -v -k "not network and not slow" --timeout=60`
Expected: All tests pass. Existing HRRR tests are unchanged and should still pass. New NBM tests pass.

**Step 2: Run network smoke tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_nbm.py::test_fetch_nbm_forecast_range_smoke -v --timeout=120`
Expected: PASS

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_generate_dashboard.py -v --timeout=60`
Expected: PASS

**Step 3: Commit if any fixes were needed**

```bash
git add -A
git commit -m "test: fix any test failures from NBM addition" || echo "No changes needed"
```

---

### Task 11: End-to-end smoke test -- generate a real dashboard with both HRRR and NBM

**Objective:** Run the full pipeline end-to-end with both HRRR and NBM data and verify the dashboard HTML contains both models.

**Step 1: Run the combined ingest + dashboard script**

Run:
```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python scripts/ingest_live_metars.py --db /tmp/dual_e2e.db --hours 2 --hrrr --hrrr-hours 18 --nbm --nbm-hours 18
.venv/bin/python scripts/generate_dashboard.py --db /tmp/dual_e2e.db --output-dir /tmp/dual_dash_test
```
Expected: Both succeed. HRRR + NBM rows inserted. Dashboard HTML written.

**Step 2: Verify the dashboard HTML contains both models**

Run:
```bash
grep -c "HRRR" /tmp/dual_dash_test/index.html
grep -c "NBM" /tmp/dual_dash_test/index.html
```
Expected: Both counts > 0

**Step 3: Verify both models have data in the DB**

Run:
```bash
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('/tmp/dual_e2e.db')
hrrr = conn.execute(\"SELECT COUNT(*) FROM hrrr_forecasts WHERE source='hrrr-aws'\").fetchone()[0]
nbm = conn.execute(\"SELECT COUNT(*) FROM hrrr_forecasts WHERE source='nbm-aws'\").fetchone()[0]
print(f'HRRR rows: {hrrr}, NBM rows: {nbm}')
assert hrrr > 0, 'No HRRR data'
assert nbm > 0, 'No NBM data'
conn.close()
"
```
Expected: Both counts > 0

**Step 4: Verify NBM temperatures are reasonable**

Run:
```bash
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('/tmp/dual_e2e.db')
rows = conn.execute(\"SELECT forecast_hour, valid_dt, tmpf FROM hrrr_forecasts WHERE source='nbm-aws' ORDER BY forecast_hour LIMIT 5\").fetchall()
for r in rows: print(f'  NBM f{r[0]:02d} valid={r[1]} temp={r[2]}F')
rows2 = conn.execute(\"SELECT forecast_hour, valid_dt, tmpf FROM hrrr_forecasts WHERE source='hrrr-aws' ORDER BY forecast_hour LIMIT 5\").fetchall()
for r in rows2: print(f'  HRRR f{r[0]:02d} valid={r[1]} temp={r[2]}F')
conn.close()
"
```
Expected: Both show reasonable temperatures (50-110F range)

**Step 5: Clean up test files**

Run: `rm -f /tmp/dual_e2e.db && rm -rf /tmp/dual_dash_test`

**Step 6: Commit final state**

```bash
git add -A
git commit -m "feat: NBM model data added alongside HRRR pipeline" || echo "All committed"
```

---

## Files Likely to Change

| File | Change type | Description |
|------|-------------|-------------|
| `pyproject.toml` | Modify | Add `rasterio>=1.5` dependency |
| `uv.lock` | Modify | Updated by `uv lock` |
| `dfw_temp_model/data/nbm.py` | **Create** | NBM COG fetcher (mirrors `data/hrrr.py`) |
| `dfw_temp_model/blending/providers.py` | Modify | Add `NBMProvider` class (HRRRProvider unchanged) |
| `dfw_temp_model/storage/obs_db.py` | Modify | Add `source` param to query functions |
| `scripts/ingest_live_metars.py` | Modify | Add `--nbm` flag alongside `--hrrr` |
| `scripts/generate_dashboard.py` | Modify | Add NBM chart, multi-model dropdown in blended chart |
| `scripts/cron_update_dashboard.sh` | Modify | Add `--nbm` alongside `--hrrr` |
| `tests/test_nbm.py` | **Create** | NBM fetcher tests |
| `tests/test_blending_providers.py` | Modify | Add NBM provider tests |
| `tests/test_obs_db.py` | Modify | Add source-filtering tests |
| `tests/test_generate_dashboard.py` | Modify | Add NBM data to fixture, NBM assertions |

Files NOT changed:
- `dfw_temp_model/data/hrrr.py` -- kept as-is, fully functional
- `dfw_temp_model/blending/bias.py` -- no changes (model-agnostic)
- `dfw_temp_model/blending/blend.py` -- no changes (provider-agnostic)
- `tests/test_blending_blend.py` -- no changes (already uses HRRRProvider, still valid)
- `tests/test_hrrr.py` -- no changes (HRRR tests still valid)

---

## Risks, Tradeoffs, and Open Questions

### Risks

1. **NBM publication lag (~2h)**: NBM cycles take longer to publish than HRRR. The `fetch_nbm_forecast_range` function starts looking 2 hours back. This means the "latest" NBM cycle on the dashboard will be ~2 hours old at cron time, while HRRR is ~1 hour behind. This is why we keep both -- HRRR provides fresher data, NBM provides more accurate (bias-corrected) data.

2. **rasterio binary dependency**: rasterio requires GDAL system libraries. It's already installed and working in the venv. The wheel package includes bundled GDAL on Linux x86_64, so `uv sync` should work without system GDAL.

3. **NBM grid is Lambert Conformal, not lat/lon**: The `src.index(lon, lat)` call handles the projection transform automatically via rasterio. Not a problem.

4. **NBM temp is int16 (whole degrees F)**: NBM stores temperature as integer Fahrenheit. HRRR stored float. The bias correction works the same way, but raw NBM temps have 1-degree granularity. This is slightly coarser but NBM is already bias-corrected by NOAA so precision is adequate.

5. **Ingest time doubles**: Fetching both HRRR (18 GRIB2 files) and NBM (18 COG files) takes ~40s total instead of ~20s. This is well within the cron 5-minute window. Could parallelize later if needed.

6. **Dashboard complexity**: The blended chart dropdown now has up to 6 entries (3 HRRR + 3 NBM). This is manageable with clear model-name labels.

### Tradeoffs

- **Freshness vs accuracy**: HRRR updates faster (~1h lag) but is raw model output. NBM updates slower (~2h lag) but is already bias-corrected by NOAA using URMA analysis. Keeping both gives users the choice.
- **Redundancy is intentional**: Both models forecast the same temperature at the same station. Showing both lets users see where models agree (high confidence) or diverge (uncertainty). This is a feature, not a bug.

### Open Questions

- Should we eventually blend HRRR and NBM together (weighted average) rather than showing them separately? The infrastructure now supports this -- a `MultiModelProvider` could wrap both and return a weighted blend. But the user asked for both shown side by side for now.
- The NBM publication lag means the NBM chart will show a cycle ~2h old. The cycle timestamp is displayed on the chart so users can see how fresh it is.

---

## Verification Checklist

- [ ] `rasterio` in pyproject.toml
- [ ] `dfw_temp_model/data/nbm.py` created with fetcher functions
- [ ] `tests/test_nbm.py` passes (unit + network smoke)
- [ ] `NBMProvider` in `blending/providers.py` with source filtering
- [ ] HRRR and NBM providers return independent results
- [ ] `obs_db.py` query functions accept `source` parameter
- [ ] `scripts/ingest_live_metars.py` has both `--hrrr` and `--nbm` flags
- [ ] `scripts/generate_dashboard.py` has NBM chart + multi-model dropdown
- [ ] `scripts/cron_update_dashboard.sh` fetches both `--hrrr` and `--nbm`
- [ ] Dashboard tests include NBM data and pass
- [ ] Existing HRRR tests still pass (no regressions)
- [ ] Full test suite passes (non-network)
- [ ] End-to-end smoke test produces HTML with both HRRR and NBM data