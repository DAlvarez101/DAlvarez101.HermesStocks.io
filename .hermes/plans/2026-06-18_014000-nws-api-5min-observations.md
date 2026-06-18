# NWS API 5-Minute Observation Ingestion + Blending Integration Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add a 5-minute observation ingestion source from the NWS API (api.weather.gov), wire it into the existing blending bias correction, display 5-minute observations on the dashboard chart, and run it on a separate 5-minute cron job — all without disrupting the existing hourly HRRR pipeline.

**Architecture:** A new `nws_api.py` fetcher module mirrors the existing `aviationweather.py` pattern. It stores observations into the same `metar_observations` table using `source='nws-api'`, so the UNIQUE(source, station, valid) constraint prevents duplicates and no schema migration is needed. The blending pipeline's `_load_metar_for_station` is updated to use 5-minute observations (no longer floored to hourly — it now uses all observations). The dashboard chart adds a lightweight 5-minute observation trace alongside the existing hourly METAR markers. Two separate cron jobs: the existing hourly job (HRRR + METAR + dashboard), and a new 5-minute job (NWS API observations + dashboard regenerate, no HRRR).

**Tech Stack:** Python 3.13, requests, pandas, sqlite3, Plotly, bash cron wrapper

---

## Current Context

The project lives at `/opt/data/stock-research/dfw_temp_model/`. Key existing components:

- `dfw_temp_model/data/aviationweather.py` — fetches hourly METARs from AviationWeather.gov
- `dfw_temp_model/storage/obs_db.py` — SQLite schema with `metar_observations` table, UNIQUE(source, station, valid)
- `dfw_temp_model/blending/blend.py` — `blended_forecast()` reads METARs via `_load_metar_for_station()` which floors to hourly and takes only the latest obs per hour
- `dfw_temp_model/blending/bias.py` — `compute_rolling_bias()` matches obs to forecasts on `valid_hour` (floored to hour)
- `scripts/generate_dashboard.py` — `blended_forecast_chart()` loads METAR observations, floors to hourly, displays as blue markers
- `scripts/ingest_live_metars.py` — CLI for METAR + optional HRRR ingestion
- `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` — production cron wrapper (hourly)
- DB: `/opt/data/stock-research/dfw_temp_model/data/cache/db/weather_observations.db`
- Config: `cron.script_timeout_seconds: 300` already set

The NWS API endpoint `https://api.weather.gov/stations/{ICAO}/observations?limit=N` returns JSON GeoJSON features with 5-minute observation timestamps, temperature in Celsius, dewpoint, wind, pressure, humidity, visibility, cloud layers. No API key needed — just a User-Agent header. 3-day rolling history window.

**Key design decisions:**
1. NWS API observations go into the SAME `metar_observations` table with `source='nws-api'`. This means the UNIQUE constraint is `(source, station, valid)` — so nws-api and aviationweather rows for the same timestamp coexist without conflict. This is fine because the blending pipeline reads ALL observations regardless of source.
2. The 5-minute cron job does NOT fetch HRRR. It only fetches NWS API observations and regenerates the dashboard. HRRR remains on the hourly cron.
3. The blending pipeline change is minimal: `_load_metar_for_station` currently floors to hourly and takes the latest obs per hour. We change it to use ALL observations (both 5-minute and hourly) without flooring, so the bias correction benefits from 12x more data points per hour.
4. The bias matching still floors to hourly for the HRRR side (HRRR is hourly), but the observation side now provides multiple points per hour, which the groupby in `compute_rolling_bias` already handles via `mean()`.

---

## Task 1: Create NWS API fetcher module

**Objective:** Create `dfw_temp_model/data/nws_api.py` that fetches 5-minute observations from api.weather.gov and returns a DataFrame in the project schema.

**Files:**
- Create: `dfw_temp_model/data/nws_api.py`
- Test: `tests/test_nws_api.py`

**Step 1: Write failing test**

```python
"""Tests for the NWS API observation fetcher."""
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from dfw_temp_model.data.nws_api import parse_nws_observations, fetch_nws_observations


def _sample_nws_payload():
    """Minimal NWS API GeoJSON payload for testing."""
    return {
        "features": [
            {
                "id": "https://api.weather.gov/stations/KDAL/observations/2026-06-18T17:45:00+00:00",
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-96.85, 32.85]},
                "properties": {
                    "timestamp": "2026-06-18T17:45:00+00:00",
                    "stationId": "KDAL",
                    "temperature": {"value": 32.0, "unitCode": "wmoUnit:degC"},
                    "dewpoint": {"value": 24.0, "unitCode": "wmoUnit:degC"},
                    "windDirection": {"value": 130.0, "unitCode": "wmoUnit:degree_(angle)"},
                    "windSpeed": {"value": 18.504, "unitCode": "wmoUnit:km_h-1"},
                    "windGust": {"value": None, "unitCode": "wmoUnit:km_h-1"},
                    "barometricPressure": {"value": 100575.74, "unitCode": "wmoUnit:Pa"},
                    "relativeHumidity": {"value": 62.7, "unitCode": "wmoUnit:percent"},
                    "visibility": {"value": 16093.44, "unitCode": "wmoUnit:m"},
                },
            },
            {
                "id": "https://api.weather.gov/stations/KDAL/observations/2026-06-18T17:40:00+00:00",
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-96.85, 32.85]},
                "properties": {
                    "timestamp": "2026-06-18T17:40:00+00:00",
                    "stationId": "KDAL",
                    "temperature": {"value": 31.8, "unitCode": "wmoUnit:degC"},
                    "dewpoint": {"value": 24.0, "unitCode": "wmoUnit:degC"},
                    "windDirection": {"value": 130.0, "unitCode": "wmoUnit:degree_(angle)"},
                    "windSpeed": {"value": 18.504, "unitCode": "wmoUnit:km_h-1"},
                    "windGust": {"value": None, "unitCode": "wmoUnit:km_h-1"},
                    "barometricPressure": {"value": 100575.74, "unitCode": "wmoUnit:Pa"},
                    "relativeHumidity": {"value": 62.7, "unitCode": "wmoUnit:percent"},
                    "visibility": {"value": 16093.44, "unitCode": "wmoUnit:m"},
                },
            },
        ]
    }


def test_parse_nws_observations_basic():
    """Parse a minimal NWS API payload into the project schema."""
    payload = _sample_nws_payload()
    df = parse_nws_observations(payload)
    assert len(df) == 2
    assert "station" in df.columns
    assert "valid" in df.columns
    assert "tmpf" in df.columns
    assert df.iloc[0]["station"] == "KDAL"
    # 32.0 C = 89.6 F
    assert df.iloc[0]["tmpf"] == pytest.approx(89.6, abs=0.1)
    # 31.8 C = 89.24 F
    assert df.iloc[1]["tmpf"] == pytest.approx(89.24, abs=0.1)


def test_parse_nws_observations_null_temperature():
    """Null temperature values become None in the DataFrame."""
    payload = {
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-96.85, 32.85]},
                "properties": {
                    "timestamp": "2026-06-18T17:45:00+00:00",
                    "stationId": "KDAL",
                    "temperature": {"value": None, "unitCode": "wmoUnit:degC"},
                    "dewpoint": {"value": None, "unitCode": "wmoUnit:degC"},
                    "windDirection": {"value": None, "unitCode": "wmoUnit:degree_(angle)"},
                    "windSpeed": {"value": None, "unitCode": "wmoUnit:km_h-1"},
                    "barometricPressure": {"value": None, "unitCode": "wmoUnit:Pa"},
                    "relativeHumidity": {"value": None, "unitCode": "wmoUnit:percent"},
                    "visibility": {"value": None, "unitCode": "wmoUnit:m"},
                },
            },
        ]
    }
    df = parse_nws_observations(payload)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["tmpf"])


def test_parse_nws_observations_empty():
    """Empty payload returns empty DataFrame with correct columns."""
    df = parse_nws_observations({"features": []})
    assert df.empty
    assert "station" in df.columns
    assert "tmpf" in df.columns


def test_fetch_nws_observations_mocked():
    """Fetch with a mocked HTTP response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _sample_nws_payload()
    mock_response.raise_for_status = MagicMock()

    with patch("dfw_temp_model.data.nws_api.requests.get", return_value=mock_response):
        df = fetch_nws_observations("KDAL", limit=2)
    assert len(df) == 2
    assert df.iloc[0]["station"] == "KDAL"
    assert df.iloc[0]["tmpf"] == pytest.approx(89.6, abs=0.1)
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_nws_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfw_temp_model.data.nws_api'`

**Step 3: Write minimal implementation**

```python
"""5-minute observation fetcher from the NWS API (api.weather.gov).

No API key required — just a User-Agent header. Returns 5-minute ASOS
observations for a single station. Data is in GeoJSON format.

Endpoint: https://api.weather.gov/stations/{ICAO}/observations?limit=N
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
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
```

**Step 4: Run test to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_nws_api.py -v`
Expected: 4 passed

**Step 5: Live smoke test**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "from dfw_temp_model.data.nws_api import fetch_nws_observations; df = fetch_nws_observations('KDAL', limit=5); print(df[['station','valid','tmpf']].to_string())"`
Expected: 5 rows with 5-minute-spaced timestamps, temperatures in Fahrenheit around 85-90F.

**Step 6: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/data/nws_api.py tests/test_nws_api.py
git commit -m "feat: add NWS API 5-minute observation fetcher"
```

---

## Task 2: Update blending pipeline to use all observations (not just hourly)

**Objective:** Modify `_load_metar_for_station` in `blend.py` to use ALL observations (both 5-minute and hourly) without flooring to hourly. The bias correction will benefit from 12x more data points per hour.

**Files:**
- Modify: `dfw_temp_model/blending/blend.py:22-40` (`_load_metar_for_station`)
- Test: `tests/test_blending_blend.py`

**Key insight:** The existing `compute_rolling_bias` in `bias.py` already floors both obs and fcst to `valid_hour` before merging. So if we pass 5-minute observations, the groupby in `compute_rolling_bias` will aggregate them by hour (taking the mean). This means the change is backward-compatible — it just provides more data points that get aggregated into the same hourly bins.

**Step 1: Write failing test**

Add to `tests/test_blending_blend.py`:

```python
def test_load_metar_uses_all_observations():
    """_load_metar_for_station should return all obs, not just one per hour."""
    conn = _make_db()
    # Insert two observations in the same hour (5-minute data)
    conn.execute(
        "INSERT INTO metar_observations (fetched_at, source, station, valid, lat, lon, tmpf) "
        "VALUES ('t','nws-api','KDAL','2026-06-17T18:30:00+00:00',32,-96,85.0)"
    )
    conn.execute(
        "INSERT INTO metar_observations (fetched_at, source, station, valid, lat, lon, tmpf) "
        "VALUES ('t','nws-api','KDAL','2026-06-17T18:45:00+00:00',32,-96,86.0)"
    )
    conn.execute(
        "INSERT INTO metar_observations (fetched_at, source, station, valid, lat, lon, tmpf) "
        "VALUES ('t','aviationweather','KDAL','2026-06-17T18:53:00+00:00',32,-96,87.0)"
    )
    from dfw_temp_model.blending.blend import _load_metar_for_station
    obs = _load_metar_for_station(conn, "KDAL")
    # All 3 observations should be present (no deduplication to hourly)
    assert len(obs) == 3
    # Column should be named valid_hour but contain actual timestamps (floored to hour for matching)
    assert "valid_hour" in obs.columns
    assert "tmpf_obs" in obs.columns
    conn.close()
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_blend.py::test_load_metar_uses_all_observations -v`
Expected: FAIL — the current code deduplicates to 1 obs per hour, so `len(obs) == 1` not 3.

**Step 3: Write minimal implementation**

Replace `_load_metar_for_station` in `blend.py`:

```python
def _load_metar_for_station(conn: sqlite3.Connection, station: str) -> pd.DataFrame:
    """Load all METAR observations for a station, floored to the hour.

    Returns ALL observations (both 5-minute NWS API and hourly AviationWeather)
    without deduplication. The compute_rolling_bias function handles
    aggregation via groupby(valid_hour).mean().
    """
    df = pd.read_sql_query(
        """
        SELECT valid, tmpf
        FROM metar_observations
        WHERE station = ? AND tmpf IS NOT NULL
        ORDER BY valid
        """,
        conn,
        params=[station],
    )
    if df.empty:
        return pd.DataFrame(columns=["valid_hour", "tmpf_obs"])
    df["valid_hour"] = pd.to_datetime(df["valid"], utc=True).dt.floor("h")
    df = df.rename(columns={"tmpf": "tmpf_obs"})
    return df[["valid_hour", "tmpf_obs"]]
```

The change is: remove `df.sort_values("valid").groupby("valid_hour").tail(1)` which deduplicated to 1-per-hour. Now all observations pass through, and `compute_rolling_bias` aggregates them.

**Step 4: Run test to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_blend.py -v`
Expected: all tests pass (including the new one)

**Step 5: Run full test suite**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow"`
Expected: all pass, no regressions

**Step 6: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/blending/blend.py tests/test_blending_blend.py
git commit -m "feat: use all observations in bias correction (5-minute + hourly)"
```

---

## Task 3: Create 5-minute ingestion script

**Objective:** Create `scripts/ingest_nws_observations.py` that fetches NWS API observations for all stations and stores them in the DB. This is the script the 5-minute cron will call.

**Files:**
- Create: `scripts/ingest_nws_observations.py`

**Step 1: Write the script**

```python
"""Fetch 5-minute observations from the NWS API for all stations and store in SQLite."""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make dfw_temp_model importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dfw_temp_model.config import STATIONS, CACHE_DIR
from dfw_temp_model.data.nws_api import fetch_nws_observations
from dfw_temp_model.storage.obs_db import get_db, insert_observations


def main():
    parser = argparse.ArgumentParser(description="Ingest 5-minute NWS API observations")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max observations per station (default 25 = ~2 hours of 5-min data)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="nws-api",
        help="Source label to store in the database",
    )
    args = parser.parse_args()

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = get_db(args.db)
    total_inserted = 0

    for station in STATIONS:
        icao = station.icao
        try:
            df = fetch_nws_observations(icao, limit=args.limit)
            if df.empty:
                print(f"  {icao}: no observations returned", file=sys.stderr)
                continue
            inserted = insert_observations(conn, df, source=args.source, fetched_at=fetched_at)
            total_inserted += inserted
            print(f"  {icao}: inserted {inserted} rows ({len(df)} fetched)")
        except Exception as exc:
            # Don't let one station failure stop the others.
            print(f"  {icao}: ERROR - {exc}", file=sys.stderr)
            continue
        # Brief pause between stations to be polite to the API.
        time.sleep(0.5)

    total = conn.execute("SELECT COUNT(*) FROM metar_observations").fetchone()[0]
    print(f"[{fetched_at}] Inserted {total_inserted} NWS API rows. Total METAR rows: {total}")
    conn.close()


if __name__ == "__main__":
    main()
```

**Step 2: Run it live**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/ingest_nws_observations.py --limit 25`
Expected: 8 stations fetched, ~12-25 rows each inserted, total row count increases.

**Step 3: Verify data in DB**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "import sqlite3; conn=sqlite3.connect('data/cache/db/weather_observations.db'); rows=conn.execute(\"SELECT source, COUNT(*) FROM metar_observations GROUP BY source\").fetchall(); print(rows)"`
Expected: `[('aviationweather', NNN), ('nws-api', MMM)]`

**Step 4: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add scripts/ingest_nws_observations.py
git commit -m "feat: add 5-minute NWS API ingestion script"
```

---

## Task 4: Update dashboard to show 5-minute observations

**Objective:** Modify the `blended_forecast_chart` in `generate_dashboard.py` to show 5-minute observations as a separate lightweight trace, in addition to the existing hourly METAR markers. Also update `summary_stats` and the latest-observation display to consider NWS API data.

**Files:**
- Modify: `scripts/generate_dashboard.py:297-312` (METAR loading in `blended_forecast_chart`)
- Modify: `scripts/generate_dashboard.py:130-151` (`summary_stats` — already reads all rows, no change needed since it doesn't filter by source)

**Step 1: Update the METAR loading in `blended_forecast_chart`**

Replace the METAR loading block (lines 297-312) with code that loads ALL observations (both sources) and displays them at their actual timestamp (not floored to hourly):

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
        # Split into 5-minute (nws-api) and hourly (aviationweather) for separate traces
        obs_5min = obs_df[obs_df["source"] == "nws-api"].copy()
        obs_hourly = obs_df[obs_df["source"] == "aviationweather"].copy()
    else:
        obs_5min = pd.DataFrame()
        obs_hourly = pd.DataFrame()
```

**Step 2: Update the trace additions**

Replace the single METAR trace block (lines 317-333) with two traces:

```python
    # Add 5-minute NWS API observations (smaller, lighter markers)
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

    # Add hourly METAR observations (larger blue markers, as before)
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
```

**Step 3: Update n_metar count for visibility logic**

Replace `n_metar = 1 if not metar_hourly.empty else 0` with:

```python
    n_obs_traces = int(not obs_5min.empty) + int(not obs_hourly.empty)
```

And update all references from `n_metar` to `n_obs_traces` in the visibility logic (lines 425, 426):

```python
        visibility = [True] * n_obs_traces  # observation traces always on
```

**Step 4: Update chart title**

Replace the title string:

```python
        title=f"Blended Forecast — {TARGET_ICAO}<br><sup>raw (orange) · corrected (green) · trend (purple) · 5-min obs (indigo) · METAR (blue)</sup>",
```

**Step 5: Run dashboard generation**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/generate_dashboard.py --db data/cache/db/weather_observations.db --output-dir /tmp/dfw-5min-test`
Expected: Dashboard written, no errors.

**Step 6: Verify the 5-minute trace is present**

Run: `grep -c '5-min obs' /tmp/dfw-5min-test/index.html`
Expected: >= 1

**Step 7: Run full test suite**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow"`
Expected: all pass

**Step 8: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add scripts/generate_dashboard.py
git commit -m "feat: display 5-minute NWS API observations on dashboard chart"
```

---

## Task 5: Create the 5-minute cron script

**Objective:** Create a separate bash wrapper for the 5-minute cron job. It fetches NWS API observations and regenerates the dashboard (no HRRR, no DB viewer). This keeps the HRRR pipeline untouched.

**Files:**
- Create: `/opt/data/.hermes/scripts/dfw_nws_5min.sh`

**Step 1: Write the script**

```bash
#!/bin/bash
# Fetch 5-minute NWS API observations and regenerate the dashboard.
# Runs every 5 minutes via Hermes cron. Does NOT fetch HRRR.
set -euo pipefail

export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1

PROJECT_DIR="/opt/data/stock-research/dfw_temp_model"
PAGES_DIR="/opt/data/DAlvarez101.HermesStocks.io"
DASHBOARD_SUBDIR="dfw-live-dashboard"
DB_PATH="${PROJECT_DIR}/data/cache/db/weather_observations.db"

cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Fetching NWS API 5-minute observations ..."
"$PYTHON" scripts/ingest_nws_observations.py --db "$DB_PATH" --limit 25

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Regenerating dashboard ..."
"$PYTHON" scripts/generate_dashboard.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/${DASHBOARD_SUBDIR}"

cd "$PAGES_DIR"

if ! git diff --quiet -- "${DASHBOARD_SUBDIR}/" || ! git diff --cached --quiet -- "${DASHBOARD_SUBDIR}/"; then
    git add "${DASHBOARD_SUBDIR}/"
    git commit -m "auto: 5-min NWS obs update $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushing to GitHub Pages ..."
    git push origin main
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No dashboard changes to push."
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."
```

**Step 2: Make it executable**

Run: `chmod +x /opt/data/.hermes/scripts/dfw_nws_5min.sh`

**Step 3: Test it manually**

Run: `/opt/data/.hermes/scripts/dfw_nws_5min.sh`
Expected: Fetches NWS observations, regenerates dashboard, pushes to GitHub Pages.

**Step 4: Verify the dashboard is live with 5-minute data**

Run: `sleep 20 && curl -sL https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/ | grep -c '5-min obs'`
Expected: >= 1

**Step 5: Commit the script to the repo (copy)**

```bash
cd /opt/data/stock-research/dfw_temp_model
cp /opt/data/.hermes/scripts/dfw_nws_5min.sh scripts/cron_nws_5min.sh
git add scripts/cron_nws_5min.sh
git commit -m "feat: add 5-minute cron script for NWS API observations"
```

---

## Task 6: Create the Hermes cron job for 5-minute runs

**Objective:** Register a new Hermes cron job that runs the 5-minute script every 5 minutes.

**Step 1: Create the cron job**

Run:
```bash
export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1
/opt/hermes/.venv/bin/hermes cron create \
    "*/5 * * * *" \
    --name "dfw-nws-5min-obs" \
    --script "dfw_nws_5min.sh" \
    --no-agent \
    --deliver local
```

**Step 2: Verify it appears in the cron list**

Run: `/opt/hermes/.venv/bin/hermes cron list`
Expected: New job `dfw-nws-5min-obs` listed with schedule `*/5 * * * *`, active.

**Step 3: Wait for first scheduled run and check output**

Run: `sleep 360 && ls -lt /opt/data/cron/output/ | head -5`
Expected: New output file for the 5-minute job.

Run: Check the latest output file for the new job ID — should show successful ingestion and dashboard push.

**Step 4: Verify the dashboard updates with 5-minute data**

Run: `curl -sL https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/ | grep -oE 'Latest observation.*</p>' | head -1`
Expected: Timestamp should be within the last 10 minutes (not the previous hour's :53).

---

## Task 7: Update the existing hourly cron to skip redundant METAR fetch

**Objective:** Since the 5-minute cron now fetches observations continuously, the hourly cron's AviationWeather METAR fetch is redundant for observation purposes. However, it still serves as a fallback. We leave the hourly cron unchanged — it will continue to fetch AviationWeather METARs (which provides a different source for cross-validation) and HRRR forecasts. The INSERT OR IGNORE constraint means duplicate timestamps from different sources coexist fine.

**No code changes needed.** The hourly cron remains as-is. The two jobs are complementary:
- 5-minute cron: NWS API observations + dashboard regenerate (fast, ~10 seconds)
- Hourly cron: AviationWeather METARs + HRRR fetch + dashboard + DB viewer (slow, ~160 seconds)

---

## Task 8: Run full test suite and push everything live

**Step 1: Run all tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow"`
Expected: all pass, 0 failures

**Step 2: Run the 5-minute cron script end-to-end**

Run: `/opt/data/.hermes/scripts/dfw_nws_5min.sh`
Expected: NWS observations ingested, dashboard regenerated, pushed to GitHub Pages.

**Step 3: Verify the live dashboard**

Run: `curl -sL https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/ | grep -oE '(5-min obs|METAR observed|Corrected|Trend-adjusted|Latest observation.*</p>)' | sort | uniq -c`
Expected: All traces present, latest observation within 10 minutes.

**Step 4: Push all code to GitHub**

```bash
cd /opt/data/stock-research/dfw_temp_model
git push origin main
```

---

## Risks, Tradeoffs, and Open Questions

### Risks
1. **NWS API rate limits**: The API has "generous" rate limits but they're not published. Fetching 8 stations every 5 minutes with 0.5s pauses = 8 requests per 5 min = ~96 requests/hour. This should be well within limits.
2. **NWS API downtime**: The API could go down. The 5-minute script uses retry+backoff and per-station error handling, so one station failure doesn't stop the rest. The hourly AviationWeather cron remains as a fallback.
3. **Dashboard git push contention**: Both cron jobs push to the same GitHub Pages repo. If they run simultaneously, one push may be rejected. Mitigation: the 5-minute job runs fast (~10s), the hourly job runs slow (~160s). Collisions are unlikely but possible. The scripts already handle this with `git push` which will fail gracefully if the other job's push is in progress.

### Tradeoffs
1. **Two observation sources**: Having both `nws-api` and `aviationweather` in the same table means some timestamps will have two observations (the :53 METAR from AviationWeather and a nearby :55 from NWS API). This is fine — the bias correction aggregates by hour, and having more data points improves the estimate.
2. **5-minute cron regenerates the full dashboard**: This is slightly wasteful (the HRRR chart doesn't change between hourly runs) but keeps the code simple. The dashboard generation takes ~3 seconds, so it's not a performance concern.

### Open Questions
1. Should the 5-minute cron also update the DB viewer? Currently no — the DB viewer is a heavier page and only needs hourly updates. The user can decide later.
2. Should we eventually drop the AviationWeather source entirely? No — it's a good cross-validation source and provides cloud/precip data that the NWS API observations endpoint doesn't fully cover.