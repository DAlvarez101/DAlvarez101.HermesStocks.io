# HRRR Live Model Data Puller — Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add the lightest-weight possible HRRR forecast-model data puller that fetches the most recent 2-meter temperature value for the DFW target location (KDFW), records the model initialization time, run time (cycle), and forecast hour, and stores it alongside live METAR observations.

**Architecture:** Use NOAA/NCEP’s public AWS Open Data HRRR GRIB2 archive (`s3://noaa-hrrr-bdp-pds/hrrr.{YYYYMMdd}/conus/hrrr.t{HH}z.wrfsfcf{FF}.grib2`) plus the lightweight `herbie-data` package (or `cfgrib`/`xarray` if herbie is unavailable). Pull only the 2-meter temperature band, extract the value at KDFW via nearest grid point, and append a new `hrrr_forecasts` row keyed by `init_dt`, `forecast_hour`, `station`. Keep the implementation dependency-light: add a single module, one ingest line in the cron script, and one test file.

**Tech Stack:** Python 3.11+, pandas, SQLite, requests, `herbie-data` (preferred) or `cfgrib` fallback. Project already uses pandas, requests, and SQLite.

---

## Current Context

- Existing live data source: `dfw_temp_model/data/aviationweather.py` fetches METAR JSON from AviationWeather.gov and returns a `pd.DataFrame` in project schema.
- Existing storage: `dfw_temp_model/storage/obs_db.py` manages SQLite `metar_observations` table keyed by `(source, station, valid)`.
- Existing ingestion script: `scripts/ingest_live_metars.py` fetches METARs and appends them to SQLite.
- Existing hourly cron wrapper: `scripts/dfw_live_metar_hourly.sh` ingests METARs, regenerates dashboard, and pushes to GitHub Pages.
- Target station: `KDFW` at `32.897, -97.038` defined in `dfw_temp_model/config.py`.
- No GRIB2 / HRRR code exists in the repo today.

---

## Design Decisions

1. **No bulk model downloads.** HRRR GRIB2 files are large (~150 MB per cycle). We only download the `TMP:2 m above ground` band using a remote index file and byte-range requests. `herbie-data` does this out of the box.
2. **Fallback if herbie is unavailable or fails.** If `herbie-data` cannot be installed/used, implement a minimal byte-range GRIB2 subsetter using the `.idx` files and `cfgrib` — but prefer herbie for speed.
3. **Store forecasts, not observations.** Create a new table `hrrr_forecasts` with columns: `id`, `fetched_at`, `station`, `init_dt` (model cycle UTC), `forecast_hour`, `valid_dt` (= init_dt + forecast_hour), `tmpf`, `lat`, `lon`, `source`. Unique on `(init_dt, forecast_hour, station)`.
4. **Find latest available cycle.** HRRR cycles every hour; determine the most recent published cycle by checking AWS for `hrrr.t{HH}z.wrfsfcf01.grib2` and walking backward up to 6 hours.
5. **Forecast hour = 1 by default for “current model value.”** The user wants the model predicted value at the same location; f01 is the first forecast step and closest to now-cast. Keep it configurable.
6. **Dashboard integration (out of scope for first pass).** This plan focuses on ingestion + storage + tests. A follow-up plan will join `metar_observations` and `hrrr_forecasts` in the dashboard.

---

## Step-by-Step Plan

### Task 1: Verify Python environment and add herbie-data dependency

**Objective:** Ensure the project venv can install `herbie-data` and its dependencies without breaking existing packages.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/pyproject.toml`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_dependencies.py`

**Step 1: Add dependency**

```toml
# In pyproject.toml under [project].dependencies
"herbie-data[extras]>=2024.4.0",
"cfgrib>=0.9.14",
"eccodes>=2.38.0",
```

> Note: `herbie-data[extras]` pulls xarray/cfgrib. If it fails due to binary wheels, use `herbie-data` (no extras) and rely on `cfgrib` directly.

**Step 2: Install in project venv**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pip install -e .`
Expected: installs `herbie-data`, `cfgrib`, and `eccodes`; existing tests still import.

**Step 3: Update dependency test**

Add to `tests/test_dependencies.py`:

```python
def test_hrrr_deps_importable():
    import herbie  # or herbie_data depending on package name
    import cfgrib
    assert herbie.__version__
```

Run: `.venv/bin/python -m pytest tests/test_dependencies.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add pyproject.toml tests/test_dependencies.py
git commit -m "deps: add herbie-data and cfgrib for HRRR GRIB2 subsetting"
```

---

### Task 2: Create HRRR fetcher module

**Objective:** Implement a module that finds the latest HRRR cycle, downloads only the 2m temperature band, extracts temperature at KDFW, and returns a DataFrame.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/hrrr.py`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_hrrr.py`

**Step 1: Write failing test**

In `tests/test_hrrr.py`:

```python
import pandas as pd
import pytest
from dfw_temp_model.config import STATIONS, TARGET_ICAO
from dfw_temp_model.data.hrrr import _valid_dt, fetch_latest_hrrr_2m_temp


def test_valid_dt_calculation():
    init = pd.Timestamp("2026-06-17T18:00:00", tz="UTC")
    assert _valid_dt(init, 1) == pd.Timestamp("2026-06-17T19:00:00", tz="UTC")
    assert _valid_dt(init, 3) == pd.Timestamp("2026-06-17T21:00:00", tz="UTC")


@pytest.mark.network
@pytest.mark.slow
def test_fetch_latest_hrrr_2m_temp_smoke():
    df = fetch_latest_hrrr_2m_temp(stations=STATIONS, forecast_hour=1, lookback_hours=6)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert TARGET_ICAO in df["station"].values
    kdfw = df[df["station"] == TARGET_ICAO].iloc[0]
    assert 20.0 <= kdfw["tmpf"] <= 120.0  # sane June DFW range
    assert pd.notna(kdfw["init_dt"])
    assert kdfw["forecast_hour"] == 1
    assert pd.notna(kdfw["valid_dt"])
    assert pd.notna(kdfw["lat"])
    assert pd.notna(kdfw["lon"])
```

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_valid_dt_calculation -v`
Expected: FAIL — module/functions do not exist.

**Step 2: Implement module**

Create `dfw_temp_model/data/hrrr.py`:

```python
"""Lightweight HRRR 2-m temperature fetcher from NOAA Open Data on AWS."""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

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


def _valid_dt(init_dt: pd.Timestamp, forecast_hour: int) -> pd.Timestamp:
    """Return valid time = init_dt + forecast_hour hours."""
    return init_dt + pd.Timedelta(hours=forecast_hour)


def _find_latest_cycle(
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    now: Optional[datetime] = None,
) -> tuple[pd.Timestamp, bool]:
    """Find the most recent published HRRR cycle for the given forecast hour.

    Returns (init_dt, found) where init_dt is the model cycle UTC timestamp.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    for i in range(lookback_hours + 1):
        init = now - timedelta(hours=i)
        init = init.replace(minute=0, second=0, microsecond=0)
        url = _grib_url(init, forecast_hour)
        # Lightweight HEAD check; the idx file is tiny.
        idx_url = f"{url}.idx"
        try:
            import requests
            r = requests.head(idx_url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                return pd.Timestamp(init, tz="UTC"), True
        except Exception:
            pass
        time.sleep(0.2)
    return pd.Timestamp(now.replace(minute=0, second=0, microsecond=0), tz="UTC"), False


def _grib_url(init_dt: datetime, forecast_hour: int) -> str:
    """Build the NOAA HRRR GRIB2 URL on AWS Open Data."""
    ymd = init_dt.strftime("%Y%m%d")
    hh = init_dt.strftime("%H")
    ff = f"{forecast_hour:02d}"
    return (
        f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
        f"hrrr.{ymd}/conus/hrrr.t{hh}z.wrfsfcf{ff}.grib2"
    )


def _nearest_point(ds, lat: float, lon: float) -> float:
    """Extract the temperature (K) at the nearest grid point and return °F."""
    # herbie returns an xarray dataset with latitude/longitude as 1D coords.
    # Use xarray’s .sel nearest if available; otherwise fall back to numpy.
    try:
        val = ds.sel(latitude=lat, longitude=lon, method="nearest")
    except Exception:
        # cfgrib sometimes has latitude/longitude as 2D aux coords.
        lat2d = ds.latitude.values
        lon2d = ds.longitude.values
        # Normalize lon to [-180,180] for comparison if needed.
        d = (lat2d - lat) ** 2 + ((lon2d - lon) % 360) ** 2
        idx = d.argmin()
        flat = ds.to_array().values.ravel()
        val = flat[idx]
        return float(val) * 9.0 / 5.0 - 459.67

    # herbie subset usually has a single variable named t2m or TMP_2maboveground.
    var_names = [v for v in val.data_vars if "tmp" in v.lower() or "t2m" in v.lower()]
    if not var_names:
        var_names = list(val.data_vars)
    k = float(val[var_names[0]].values)
    return k * 9.0 / 5.0 - 459.67


def fetch_latest_hrrr_2m_temp(
    stations: Iterable[Station],
    forecast_hour: int = 1,
    lookback_hours: int = 6,
    cache_path: Optional[str] = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch the latest HRRR 2m temperature for each station.

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
        Per-request timeout.

    Returns
    -------
    pd.DataFrame
        One row per station with init_dt, forecast_hour, valid_dt, tmpf.
    """
    init_dt, found = _find_latest_cycle(forecast_hour, lookback_hours)
    if not found:
        return pd.DataFrame(columns=HRRR_OUTPUT_COLUMNS)

    url = _grib_url(init_dt.to_pydatetime(), forecast_hour)

    # Try Herbie first (lightweight remote subsetting).
    try:
        from herbie_data import Herbie

        H = Herbie(
            init_dt.to_pydatetime(),
            model="hrrr",
            product="sfc",
            fxx=forecast_hour,
            verbose=False,
        )
        ds = H.xarray("TMP:2 m", verbose=False)
    except Exception:
        # Fallback: download only the relevant byte range via .idx and open with cfgrib.
        ds = _cfgrib_subset(url, "TMP:2 m above ground", timeout)

    rows = []
    for s in stations:
        tmpf = _nearest_point(ds, s.lat, s.lon)
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


def _cfgrib_subset(grib_url: str, search_key: str, timeout: float):
    """Minimal fallback: parse .idx, request byte range, open with cfgrib."""
    import re
    import tempfile

    import requests

    idx_url = f"{grib_url}.idx"
    idx_text = requests.get(idx_url, timeout=timeout).text

    # Find the line containing the search key.
    pattern = re.compile(r"^(\d+):(\d+):(\S+):TMP:2 m above ground:.+$", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(idx_text)
    if not match:
        raise ValueError(f"Could not find {search_key} in HRRR index")

    start_byte = int(match.group(2))
    # End byte is the start of the next message minus one, or EOF.
    next_match = pattern.search(idx_text, match.end())
    end_byte = int(next_match.group(2)) - 1 if next_match else ""

    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    r = requests.get(grib_url, headers=headers, timeout=timeout)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    import cfgrib

    return cfgrib.open_dataset(tmp_path)
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_valid_dt_calculation -v`
Expected: PASS

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_fetch_latest_hrrr_2m_temp_smoke -v`
Expected: PASS (network; may take 10-30s)

**Step 4: Commit**

```bash
git add dfw_temp_model/data/hrrr.py tests/test_hrrr.py
git commit -m "feat: add lightweight HRRR 2m temperature fetcher"
```

---

### Task 3: Extend SQLite storage for HRRR forecasts

**Objective:** Add a `hrrr_forecasts` table and insert method so forecasts are stored alongside METAR observations.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/storage/obs_db.py`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_obs_db.py`

**Step 1: Write failing test**

In `tests/test_obs_db.py` add:

```python
def test_insert_hrrr_forecasts(tmp_path):
    db = tmp_path / "test.db"
    conn = get_db(str(db))
    df = pd.DataFrame([
        {
            "station": "KDFW",
            "init_dt": "2026-06-17T18:00:00+00:00",
            "forecast_hour": 1,
            "valid_dt": "2026-06-17T19:00:00+00:00",
            "lat": 32.897,
            "lon": -97.038,
            "tmpf": 86.5,
        }
    ])
    inserted = insert_hrrr_forecasts(conn, df)
    assert inserted == 1
    rows = conn.execute("SELECT * FROM hrrr_forecasts").fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "KDFW"
```

Run: `.venv/bin/python -m pytest tests/test_obs_db.py::test_insert_hrrr_forecasts -v`
Expected: FAIL — `insert_hrrr_forecasts` not defined.

**Step 2: Add schema and insert function**

In `dfw_temp_model/storage/obs_db.py` add after `SCHEMA_SQL`:

```python
HRRR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hrrr_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'hrrr-aws',
    station TEXT NOT NULL,
    init_dt TEXT NOT NULL,
    forecast_hour INTEGER NOT NULL,
    valid_dt TEXT NOT NULL,
    lat REAL,
    lon REAL,
    tmpf REAL,
    UNIQUE(init_dt, forecast_hour, station)
);

CREATE INDEX IF NOT EXISTS idx_hrrr_station_valid
    ON hrrr_forecasts(station, valid_dt);

CREATE INDEX IF NOT EXISTS idx_hrrr_fetched_at
    ON hrrr_forecasts(fetched_at);
"""
```

Update `ensure_schema` to execute `HRRR_SCHEMA_SQL` as well.

Add `insert_hrrr_forecasts`:

```python
def insert_hrrr_forecasts(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source: str = "hrrr-aws",
    fetched_at: Optional[str] = None,
) -> int:
    """Insert HRRR forecast rows using INSERT OR IGNORE.

    Returns the number of newly inserted rows.
    """
    if df.empty:
        return 0
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).isoformat()

    columns = [
        "fetched_at", "source", "station", "init_dt", "forecast_hour",
        "valid_dt", "lat", "lon", "tmpf",
    ]
    rows = []
    for _, row in df.iterrows():
        init_dt = row.get("init_dt")
        valid_dt = row.get("valid_dt")
        if isinstance(init_dt, pd.Timestamp):
            init_dt = init_dt.isoformat()
        if isinstance(valid_dt, pd.Timestamp):
            valid_dt = valid_dt.isoformat()
        rows.append((
            fetched_at, source, row.get("station"), init_dt,
            int(row.get("forecast_hour")), valid_dt,
            row.get("lat"), row.get("lon"), row.get("tmpf"),
        ))

    cursor = conn.cursor()
    cursor.executemany(
        f"""
        INSERT OR IGNORE INTO hrrr_forecasts (
            {', '.join(columns)}
        ) VALUES ({', '.join('?' for _ in columns)})
        """,
        rows,
    )
    conn.commit()
    return cursor.rowcount
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_obs_db.py -v`
Expected: all existing + new tests PASS

**Step 4: Commit**

```bash
git add dfw_temp_model/storage/obs_db.py tests/test_obs_db.py
git commit -m "feat: add hrrr_forecasts SQLite table and insert helper"
```

---

### Task 4: Wire HRRR ingestion into live script

**Objective:** Add an optional `--hrrr` flag to the live METAR ingest script so each hourly run can pull the latest HRRR forecast.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/scripts/ingest_live_metars.py`

**Step 1: Add imports and flag**

```python
from dfw_temp_model.data.hrrr import fetch_latest_hrrr_2m_temp
from dfw_temp_model.storage.obs_db import insert_hrrr_forecasts
```

Add CLI argument:

```python
parser.add_argument(
    "--hrrr",
    action="store_true",
    help="Also fetch the latest HRRR 2m temperature forecast",
)
```

**Step 2: Add HRRR fetch block after METAR insert**

```python
if args.hrrr:
    print("Fetching latest HRRR 2m temperature ...")
    hrrr_df = fetch_latest_hrrr_2m_temp(
        STATIONS, forecast_hour=1, lookback_hours=6
    )
    if hrrr_df.empty:
        print("No HRRR forecast returned.", file=sys.stderr)
    else:
        conn = get_db(args.db)
        inserted = insert_hrrr_forecasts(
            conn, hrrr_df, source="hrrr-aws", fetched_at=fetched_at
        )
        total = conn.execute("SELECT COUNT(*) FROM hrrr_forecasts").fetchone()[0]
        conn.close()
        print(f"Inserted {inserted} HRRR rows. Total HRRR rows: {total}")
```

**Step 3: Update live script test**

In `tests/test_ingest_script.py` (or create if missing), add a unit test for argument parsing and a network smoke test with `--hrrr`.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_ingest_script.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/ingest_live_metars.py tests/test_ingest_script.py
git commit -m "feat: add --hrrr flag to live ingest script"
```

---

### Task 5: Update hourly cron wrapper to enable HRRR pull

**Objective:** Enable HRRR ingestion in the existing cronjob without breaking the existing METAR-only path.

**Files:**
- Modify: `/opt/data/scripts/dfw_live_metar_hourly.sh`

**Step 1: Add --hrrr to ingest call**

Change line 24 from:
```bash
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK"
```
to:
```bash
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK" --hrrr
```

**Step 2: Verify script syntax**

Run: `bash -n /opt/data/scripts/dfw_live_metar_hourly.sh`
Expected: no output (success).

**Step 3: Commit**

```bash
git add /opt/data/scripts/dfw_live_metar_hourly.sh
git commit -m "ops: enable HRRR pull in hourly cron wrapper"
```

---

### Task 6: Add smoke-run verification

**Objective:** Run the HRRR fetch end-to-end once on the target station and confirm real data lands in the database.

**Files:**
- None new; uses existing script and DB path.

**Step 1: Run HRRR fetch directly**

Run:
```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python -c "
from dfw_temp_model.config import STATIONS
from dfw_temp_model.data.hrrr import fetch_latest_hrrr_2m_temp
df = fetch_latest_hrrr_2m_temp(STATIONS[:1], forecast_hour=1)
print(df.to_string(index=False))
"
```
Expected: prints one row with KDFW init_dt, forecast_hour=1, valid_dt, tmpf in °F.

**Step 2: Run ingest script with --hrrr**

Run:
```bash
cd /opt/data/stock-research/dfw_temp_model
DB_PATH=data/cache/db/weather_observations.db
.venv/bin/python scripts/ingest_live_metars.py --db "$DB_PATH" --hours 2 --hrrr
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('data/cache/db/weather_observations.db')
print('HRRR rows:', conn.execute('SELECT COUNT(*) FROM hrrr_forecasts').fetchone()[0])
print(conn.execute('SELECT * FROM hrrr_forecasts ORDER BY init_dt DESC LIMIT 1').fetchone())
conn.close()
"
```
Expected: HRRR rows > 0; latest row has real temp and cycle metadata.

**Step 3: Commit nothing extra; confirm clean working tree**

Run: `git status`
Expected: working tree clean (all changes already committed).

---

## Tests / Validation

- Unit: `_valid_dt` calculation.
- Unit: empty DataFrame shape when no cycle found.
- Unit: `insert_hrrr_forecasts` deduplication via `INSERT OR IGNORE`.
- Network smoke: `fetch_latest_hrrr_2m_temp` returns a real KDFW row with sane temperature.
- Integration: `scripts/ingest_live_metars.py --hrrr` appends to SQLite.
- Regression: run full pytest suite: `.venv/bin/python -m pytest tests/ -m 'not network'`

---

## Risks, Tradeoffs, and Open Questions

| Risk | Mitigation |
|------|-----------|
| `herbie-data` fails to install due to eccodes binary dependency | Use `cfgrib` fallback path; if both fail, skip HRRR and log error. |
| HRRR cycle not yet published | `_find_latest_cycle` looks back up to 6 hours. |
| Large GRIB2 download despite subsetting | Use herbie’s remote index byte-range request; never download full file. |
| Nearest-grid-point extraction differs from station location | Record station lat/lon and document that value is nearest grid cell. |
| Dashboard not updated in this plan | Follow-up plan will join `hrrr_forecasts` to METAR table for display. |

**Open questions for the user:**
1. Do you want the dashboard to show HRRR vs METAR in this same PR, or a separate follow-up?
2. Should we pull a single forecast hour (f01) or a small ladder (f01–f06) for richer comparison?
3. Is `herbie-data` acceptable as a dependency, or do you prefer a pure `cfgrib` implementation to avoid an extra package?
