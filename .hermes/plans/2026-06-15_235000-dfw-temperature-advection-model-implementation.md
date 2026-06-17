# DFW Temperature Advection Model — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a verified historical data pipeline and baseline advection model for DFW daily high temperature prediction, starting from the saved design instructions and producing a reproducible first experiment.

**Architecture:** A Python package `dfw_temp_model/` fetches IEM ASOS observations and Open-Meteo historical forecasts, aligns them into a Parquet dataset, computes observation-minus-forecast residuals, and runs two models: (1) a simple inverse-distance-weighted baseline and (2) a wind-advection-weighted model. Both are evaluated on a 2024 time-based holdout using RMSE and bucket hit rate.

**Tech Stack:** Python 3.13, `pandas`, `pyarrow`, `requests`, `pyproj`, `scikit-learn` (for weighted plane fit and K-means), `optuna` (optional later), `pytest`.

---

## Current Context

- Design instructions saved at `/opt/data/DAlvarez101.HermesStocks.io/2026-06-15-dfw-temperature-advection-model-instructions.md`.
- Target station: KDFW (Dallas/Fort Worth International).
- Neighbor stations: KDAL, KADS, KAFW, KDTO, KGKY, KACT, KTYR.
- No existing code or data repository for this project.
- Active Python environment is `/opt/hermes/.venv/bin/python` (uv-managed).

## Open Questions / Assumptions

- Assume Open-Meteo historical API is sufficient for the first forecast source (simpler than HRRR). We will validate coverage for 2020-2024 before committing.
- Assume daily high temperature is the first predicted variable.
- Assume IEM ASOS hourly data is sufficient for daily highs; 1-minute data is a later optimization.
- Lead time for initial experiments: 0-24 hours ahead. We will start with same-day (lead_time_hours=0) to remove forecast lead-time complexity, then add lead times.

---

## Task 1: Create Project Skeleton

**Objective:** Create the `dfw_temp_model/` package structure under `/opt/data/stock-research/` with config and a minimal README.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/README.md`
- Create: `/opt/data/stock-research/dfw_temp_model/pyproject.toml`
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/__init__.py`
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/config.py`
- Create: `/opt/data/stock-research/dfw_temp_model/data/.gitkeep`
- Create: `/opt/data/stock-research/dfw_temp_model/notebooks/.gitkeep`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/.gitkeep`

**Step 1: Write `pyproject.toml`**

```toml
[project]
name = "dfw-temp-model"
version = "0.1.0"
description = "DFW-area temperature advection model for prediction market research"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "pyarrow>=14.0",
    "requests>=2.31",
    "pyproj>=3.6",
    "scikit-learn>=1.3",
    "optuna>=3.6",
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Write `dfw_temp_model/config.py`**

```python
from dataclasses import dataclass

@dataclass
class Station:
    icao: str
    lat: float
    lon: float
    elevation_ft: float
    role: str

STATIONS = [
    Station("KDFW", 32.897, -97.038, 607, "target"),
    Station("KDAL", 32.848, -96.851, 487, "urban_east"),
    Station("KADS", 33.075, -96.837, 645, "north_suburban"),
    Station("KAFW", 32.990, -97.319, 679, "northwest_exurban"),
    Station("KDTO", 33.200, -97.198, 642, "north_rural"),
    Station("KGKY", 32.664, -97.094, 628, "south_arlington"),
    Station("KACT", 31.611, -97.230, 686, "south_rural"),
    Station("KTYR", 32.354, -95.402, 550, "east_rural"),
]

TARGET_ICAO = "KDFW"
CACHE_DIR = "data/cache"
```

**Step 3: Write README.md**

Brief project description, install command (`uv sync` or `pip install -e .`), and note that this is research code for the Polymarket weather markets.

**Step 4: Verify structure**

Run:

```bash
cd /opt/data/stock-research/dfw_temp_model
find . -type f
```

Expected output shows the files above.

**Step 5: Commit**

```bash
git add .
git commit -m "chore: create dfw_temp_model project skeleton"
```

---

## Task 2: Verify Python Environment and Dependencies

**Objective:** Ensure the active venv can import required packages and install missing ones.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/pyproject.toml` (already created if missing deps found)

**Step 1: Run dependency check script**

```python
import importlib
packages = ["pandas", "pyarrow", "requests", "pyproj", "sklearn", "optuna", "pytest"]
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"OK: {pkg}")
    except ImportError:
        print(f"MISSING: {pkg}")
```

**Step 2: Install missing packages**

Run:

```bash
/opt/hermes/.venv/bin/python -m pip install pandas pyarrow requests pyproj scikit-learn optuna pytest
```

or, if uv project is active:

```bash
uv add pandas pyarrow requests pyproj scikit-learn optuna pytest
```

**Step 3: Verify imports again**

Run the check script. Expected: all `OK`.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add project dependencies"
```

---

## Task 3: Implement IEM ASOS Data Fetcher

**Objective:** Download hourly ASOS observations from IEM for all 8 stations and years 2020-2024, cache as Parquet.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/iem_asos.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_iem_asos.py`

**Step 1: Write failing test**

```python
def test_build_iem_url():
    from dfw_temp_model.data.iem_asos import build_iem_url
    url = build_iem_url("KDFW", "2022-01-01", "2022-01-02", ["tmpf", "drct", "sknt"])
    assert "station=KDFW" in url
    assert "data=tmpf" in url
    assert "year1=2022" in url
```

Run:

```bash
cd /opt/data/stock-research/dfw_temp_model
/opt/hermes/.venv/bin/python -m pytest tests/test_iem_asos.py::test_build_iem_url -v
```

Expected: FAIL — module not found.

**Step 2: Implement URL builder and fetcher**

```python
import io
from typing import List
import pandas as pd
import requests

IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

VARIABLES = ["tmpf", "dwpf", "drct", "sknt", "skyc1", "mslp", "p01i"]


def build_iem_url(station: str, start: str, end: str, variables: List[str]) -> str:
    params = {
        "station": station,
        "data": ",".join(variables),
        "year1": start[:4],
        "month1": int(start[5:7]),
        "day1": int(start[8:10]),
        "year2": end[:4],
        "month2": int(end[5:7]),
        "day2": int(end[8:10]),
        "tz": "UTC",
        "format": "csv",
        "latlon": "yes",
        "direct": "no",
        "report_type": ["1", "2"],
    }
    r = requests.Request("GET", IEM_BASE, params=params).prepare()
    return r.url


def fetch_asos_csv(station: str, start: str, end: str, variables: List[str] = None) -> pd.DataFrame:
    variables = variables or VARIABLES
    url = build_iem_url(station, start, end, variables)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), parse_dates=["valid"], na_values=["M"])
    df["station"] = station
    return df
```

**Step 3: Run test**

```bash
/opt/hermes/.venv/bin/python -m pytest tests/test_iem_asos.py::test_build_iem_url -v
```

Expected: PASS.

**Step 4: Add integration test (small fetch)**

```python
def test_fetch_asos_csv_smoke():
    from dfw_temp_model.data.iem_asos import fetch_asos_csv
    df = fetch_asos_csv("KDFW", "2024-06-01", "2024-06-03")
    assert not df.empty
    assert "tmpf" in df.columns
    assert "valid" in df.columns
```

Run:

```bash
/opt/hermes/.venv/bin/python -m pytest tests/test_iem_asos.py::test_fetch_asos_csv_smoke -v
```

Expected: PASS (requires network).

**Step 5: Add batch fetch + Parquet cache**

Add function `fetch_all_stations(start, end, stations, cache_path=None)` that loops over stations, concatenates into a single DataFrame, optionally writes Parquet, and returns the DataFrame.

Test:

```python
def test_fetch_all_stations():
    from dfw_temp_model.data.iem_asos import fetch_all_stations
    from dfw_temp_model.config import STATIONS
    df = fetch_all_stations("2024-06-01", "2024-06-03", STATIONS, cache_path="/tmp/test_asos.parquet")
    assert not df.empty
    assert "station" in df.columns
```

Run test. Expected: PASS.

**Step 6: Commit**

```bash
git add dfw_temp_model/data/iem_asos.py tests/test_iem_asos.py
git commit -m "feat: add IEM ASOS hourly fetcher with Parquet cache"
```

---

## Task 4: Implement Open-Meteo Historical Forecast Fetcher

**Objective:** Fetch hourly 2-meter temperature forecasts from Open-Meteo archive API for all 8 station lat/lons for 2020-2024, cache as Parquet.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/openmeteo.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_openmeteo.py`

**Step 1: Write failing test**

```python
def test_build_openmeteo_url():
    from dfw_temp_model.data.openmeteo import build_url
    url = build_url(32.897, -97.038, "2024-06-01", "2024-06-03")
    assert "latitude=32.897" in url
    assert "longitude=-97.038" in url
    assert "hourly=temperature_2m" in url
```

Run:

```bash
/opt/hermes/.venv/bin/python -m pytest tests/test_openmeteo.py::test_build_openmeteo_url -v
```

Expected: FAIL.

**Step 2: Implement URL builder and fetcher**

```python
import requests
import pandas as pd

API = "https://archive-api.open-meteo.com/v1/archive"


def build_url(lat: float, lon: float, start: str, end: str) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    r = requests.Request("GET", API, params=params).prepare()
    return r.url


def fetch_hourly_temp(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = build_url(lat, lon, start, end)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame({"valid": pd.to_datetime(data["time"]), "fcst_temp_f": data["temperature_2m"]})
    df["lat"] = lat
    df["lon"] = lon
    return df


def fetch_all_stations(stations, start: str, end: str, cache_path=None) -> pd.DataFrame:
    frames = []
    for st in stations:
        df = fetch_hourly_temp(st.lat, st.lon, start, end)
        df["station"] = st.icao
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if cache_path:
        out.to_parquet(cache_path)
    return out
```

**Step 3: Run test**

```bash
/opt/hermes/.venv/bin/python -m pytest tests/test_openmeteo.py::test_build_openmeteo_url -v
```

Expected: PASS.

**Step 4: Add smoke test**

```python
def test_fetch_hourly_temp_smoke():
    from dfw_temp_model.data.openmeteo import fetch_hourly_temp
    df = fetch_hourly_temp(32.897, -97.038, "2024-06-01", "2024-06-03")
    assert not df.empty
    assert "fcst_temp_f" in df.columns
```

Run. Expected: PASS.

**Step 5: Commit**

```bash
git add dfw_temp_model/data/openmeteo.py tests/test_openmeteo.py
git commit -m "feat: add Open-Meteo historical forecast fetcher"
```

---

## Task 5: Build Observation + Forecast Alignment Pipeline

**Objective:** From raw ASOS and Open-Meteo data, produce a clean merged table with daily observed highs and forecast highs per station.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/build_dataset.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_build_dataset.py`

**Step 1: Write failing test**

```python
def test_compute_daily_highs():
    import pandas as pd
    from dfw_temp_model.data.build_dataset import compute_daily_highs
    df = pd.DataFrame({
        "station": ["KDFW", "KDFW", "KDFW"],
        "valid": pd.to_datetime(["2024-06-01 12:00:00+00:00", "2024-06-01 18:00:00+00:00", "2024-06-02 12:00:00+00:00"]),
        "tmpf": [80.0, 85.0, 82.0],
    })
    out = compute_daily_highs(df)
    assert len(out) == 2
    assert out.loc["2024-06-01", "KDFW"] == 85.0
```

Run:

```bash
/opt/hermes/.venv/bin/python -m pytest tests/test_build_dataset.py::test_compute_daily_highs -v
```

Expected: FAIL.

**Step 2: Implement daily-high aggregation**

```python
import pandas as pd


def compute_daily_highs(obs_df: pd.DataFrame) -> pd.DataFrame:
    df = obs_df[["station", "valid", "tmpf"]].copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["date"] = df["valid"].dt.tz_localize(None).dt.date.astype(str)
    daily = df.groupby(["date", "station"])["tmpf"].max().reset_index()
    return daily.pivot(index="date", columns="station", values="tmpf")
```

Run test. Expected: PASS.

**Step 3: Implement forecast daily highs**

```python
def compute_forecast_daily_highs(fcst_df: pd.DataFrame) -> pd.DataFrame:
    df = fcst_df[["station", "valid", "fcst_temp_f"]].copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["date"] = df["valid"].dt.tz_localize(None).dt.date.astype(str)
    daily = df.groupby(["date", "station"])["fcst_temp_f"].max().reset_index()
    return daily.pivot(index="date", columns="station", values="fcst_temp_f")
```

Test:

```python
def test_compute_forecast_daily_highs():
    import pandas as pd
    from dfw_temp_model.data.build_dataset import compute_forecast_daily_highs
    df = pd.DataFrame({
        "station": ["KDFW", "KDFW"],
        "valid": pd.to_datetime(["2024-06-01 12:00:00+00:00", "2024-06-01 18:00:00+00:00"]),
        "fcst_temp_f": [80.0, 86.0],
    })
    out = compute_forecast_daily_highs(df)
    assert out.loc["2024-06-01", "KDFW"] == 86.0
```

Run. Expected: PASS.

**Step 4: Implement merge + residual computation**

```python
def build_residual_table(obs_daily: pd.DataFrame, fcst_daily: pd.DataFrame, stations) -> pd.DataFrame:
    merged = obs_daily.join(fcst_daily, how="inner", rsuffix="_fcst")
    residuals = pd.DataFrame(index=merged.index)
    for st in stations:
        if st.icao in obs_daily.columns and f"{st.icao}_fcst" in merged.columns:
            residuals[st.icao] = merged[st.icao] - merged[f"{st.icao}_fcst"]
    return residuals
```

Test with synthetic data. Expected: PASS.

**Step 5: End-to-end smoke test**

Write a test that fetches one month of real data (2024-06) for all stations, builds the residual table, and asserts:
- At least 25 non-null days.
- KDFW residual column exists.
- All neighbor columns exist.

Run. Expected: PASS (slow, network).

**Step 6: Commit**

```bash
git add dfw_temp_model/data/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: add daily-high alignment and residual table builder"
```

---

## Task 6: Implement Geometry Utilities

**Objective:** Convert station lat/lon into a local Cartesian grid centered on KDFW, compute distances, bearings, and upwind flags.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/features/geometry.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_geometry.py`

**Step 1: Write failing test**

```python
def test_local_coordinates():
    from dfw_temp_model.features.geometry import local_xy
    from dfw_temp_model.config import STATIONS
    target = next(s for s in STATIONS if s.icao == "KDFW")
    x, y = local_xy(target.lat, target.lon, target.lat, target.lon)
    assert x == 0.0
    assert y == 0.0
```

Run. Expected: FAIL.

**Step 2: Implement geometry functions**

```python
import math
import pyproj


def make_projection(lat0: float, lon0: float):
    return pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0, units="m")


def local_xy(lat: float, lon: float, lat0: float, lon0: float):
    proj = make_projection(lat0, lon0)
    return proj(lon, lat)


def distance_m(x1, y1, x2, y2) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def bearing_deg(x1, y1, x2, y2) -> float:
    dx = x2 - x1
    dy = y2 - y1
    angle = math.degrees(math.atan2(dx, dy))
    return angle % 360


def smallest_angle_diff(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return abs(diff)
```

Run test. Expected: PASS.

**Step 3: Add station geometry table**

```python
def station_geometry_table(stations):
    target = next(s for s in stations if s.icao == "KDFW")
    rows = []
    for s in stations:
        x, y = local_xy(s.lat, s.lon, target.lat, target.lon)
        dist = distance_m(0, 0, x, y)
        bearing = bearing_deg(0, 0, x, y)
        rows.append({
            "icao": s.icao,
            "x_m": x,
            "y_m": y,
            "dist_m": dist,
            "dist_km": dist / 1000,
            "bearing_from_target_deg": bearing,
            "elevation_diff_ft": s.elevation_ft - target.elevation_ft,
            "role": s.role,
        })
    return pd.DataFrame(rows)
```

Test with all stations; assert KDFW has dist_km == 0 and bearing == 0 (or 360). Expected: PASS.

**Step 4: Commit**

```bash
git add dfw_temp_model/features/geometry.py tests/test_geometry.py
git commit -m "feat: add local Cartesian geometry and station table"
```

---

## Task 7: Implement Baseline Inverse-Distance Model

**Objective:** A simple model that predicts KDFW residual as inverse-distance-weighted average of neighbor residuals.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/models/baseline.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_baseline.py`

**Step 1: Write failing test**

```python
def test_idw_baseline():
    import pandas as pd
    from dfw_temp_model.models.baseline import inverse_distance_predict
    residuals = pd.DataFrame({
        "KDFW": [1.0, 2.0],
        "KDAL": [2.0, 4.0],
        "KADS": [0.0, 0.0],
    }, index=["2024-06-01", "2024-06-02"])
    geom = pd.DataFrame({
        "icao": ["KDAL", "KADS"],
        "dist_km": [10.0, 20.0],
    })
    pred = inverse_distance_predict(residuals, geom, target_col="KDFW", p=2)
    assert "predicted_residual" in pred.columns
    # closer KDAL should dominate
    assert pred.loc["2024-06-01", "predicted_residual"] > 1.0
```

Run. Expected: FAIL.

**Step 2: Implement inverse-distance predictor**

```python
import pandas as pd
import numpy as np


def inverse_distance_predict(residuals: pd.DataFrame, geom: pd.DataFrame, target_col: str = "KDFW", p: float = 2.0) -> pd.DataFrame:
    neighbors = [c for c in residuals.columns if c != target_col]
    geom = geom.set_index("icao").loc[neighbors]
    weights = 1.0 / (geom["dist_km"].values ** p)
    weights = weights / weights.sum()
    pred = residuals[neighbors].values @ weights
    out = residuals[[target_col]].copy()
    out["predicted_residual"] = pred
    out["corrected_temp_f"] = out[target_col] - out["predicted_residual"]
    return out
```

Run test. Expected: PASS.

**Step 3: Commit**

```bash
git add dfw_temp_model/models/baseline.py tests/test_baseline.py
git commit -m "feat: add inverse-distance-weighted baseline model"
```

---

## Task 8: Implement Wind-Advection Model

**Objective:** Predict KDFW residual using wind-direction-weighted neighbor residuals, advection time, and front detection.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/models/advection.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_advection.py`

**Step 1: Write failing test for upwind weight**

```python
def test_upwind_weight():
    from dfw_temp_model.models.advection import upwind_weight
    # station directly upwind (bearing from target to station = 180, wind from 180 -> wind_dir=180)
    # Actually: if wind_dir=0 (north wind), upwind station is south of target, bearing from target to station = 180
    w = upwind_weight(180, wind_dir=0, half_width=45, boost=3.0)
    assert w > 1.0
```

Run. Expected: FAIL.

**Step 2: Implement upwind weight and advection functions**

```python
import math


def smallest_angle_diff(a, b):
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def upwind_weight(bearing_from_target: float, wind_dir: float, half_width: float = 45.0, boost: float = 3.0) -> float:
    diff = smallest_angle_diff(bearing_from_target, wind_dir)
    if diff > half_width:
        return 1.0
    return 1.0 + (boost - 1.0) * math.cos(math.radians(diff * (90.0 / half_width)))
```

Run test. Expected: PASS.

**Step 3: Implement advection predictor**

```python
import pandas as pd
import numpy as np
from dfw_temp_model.features.geometry import smallest_angle_diff


def advection_predict(residuals, geom, wind_df, target_col="KDFW", p=2.0, half_width=45.0, boost=3.0, l_adv_km=50.0) -> pd.DataFrame:
    neighbors = [c for c in residuals.columns if c != target_col]
    geom = geom.set_index("icao").loc[neighbors]
    dists = geom["dist_km"].values
    bearings = geom["bearing_from_target_deg"].values

    out = residuals[[target_col]].copy()
    preds = []
    for date in residuals.index:
        wind_dir = wind_df.loc[date, "wind_dir_deg"]
        wind_speed_kts = wind_df.loc[date, "wind_speed_kts"]
        wind_speed_ms = wind_speed_kts * 0.514444

        weights = 1.0 / (dists ** p)
        uw = np.array([upwind_weight(b, wind_dir, half_width, boost) for b in bearings])
        adv_decay = np.exp(-dists / l_adv_km)
        weights = weights * uw * adv_decay
        weights = weights / weights.sum()
        pred = residuals.loc[date, neighbors].values @ weights
        preds.append(pred)

    out["predicted_residual"] = preds
    out["corrected_temp_f"] = out[target_col] - out["predicted_residual"]
    return out
```

**Step 4: Test with synthetic data**

Create a test where wind is from the north, the southern station has a large residual, and assert the advection model shifts KDFW correction toward that southern residual more than the baseline does.

Run. Expected: PASS.

**Step 5: Commit**

```bash
git add dfw_temp_model/models/advection.py tests/test_advection.py
git commit -m "feat: add wind-advection-weighted correction model"
```

---

## Task 9: Implement Evaluation Framework

**Objective:** Compute RMSE, MAE, and bucket hit rate for corrected daily high temperatures.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/evaluation/metrics.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_metrics.py`

**Step 1: Write failing test**

```python
def test_rmse_mae():
    import pandas as pd
    from dfw_temp_model.evaluation.metrics import rmse, mae
    df = pd.DataFrame({
        "obs": [80.0, 82.0, 84.0],
        "pred": [81.0, 82.0, 83.0],
    })
    assert rmse(df["obs"], df["pred"]) == 1.0
    assert mae(df["obs"], df["pred"]) == 2.0 / 3.0
```

Run. Expected: FAIL.

**Step 2: Implement metrics**

```python
import numpy as np


def rmse(obs, pred):
    return np.sqrt(np.mean((obs - pred) ** 2))


def mae(obs, pred):
    return np.mean(np.abs(obs - pred))


def bucket_hit_rate(obs, pred, bucket_width=1.0):
    rounded_obs = np.round(obs / bucket_width) * bucket_width
    rounded_pred = np.round(pred / bucket_width) * bucket_width
    return np.mean(rounded_obs == rounded_pred)
```

Run test. Expected: PASS.

**Step 3: Add model comparison function**

```python
def evaluate_correction(obs, fcst, corrected) -> dict:
    return {
        "raw_rmse": rmse(obs, fcst),
        "raw_mae": mae(obs, fcst),
        "corrected_rmse": rmse(obs, corrected),
        "corrected_mae": mae(obs, corrected),
        "rmse_improvement": rmse(obs, fcst) - rmse(obs, corrected),
        "bucket_hit_rate": bucket_hit_rate(obs, corrected),
    }
```

Test with synthetic data. Expected: PASS.

**Step 4: Commit**

```bash
git add dfw_temp_model/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add evaluation metrics RMSE MAE bucket hit rate"
```

---

## Task 10: Implement Train / Validation / Test Split

**Objective:** Split residual table by date for walk-forward validation.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/training/splits.py`
- Create: `/opt/data/stock-research/dfw_temp_model/tests/test_splits.py`

**Step 1: Write failing test**

```python
def test_time_split():
    import pandas as pd
    from dfw_temp_model.training.splits import time_based_split
    df = pd.DataFrame(index=pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]).date)
    train, val, test = time_based_split(df, "2023-12-31", "2024-12-31")
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
```

Run. Expected: FAIL.

**Step 2: Implement split**

```python
import pandas as pd


def time_based_split(df: pd.DataFrame, train_end: str, val_end: str):
    dates = pd.to_datetime(df.index).to_series().dt.date
    train = df[dates <= pd.to_datetime(train_end).date()]
    val = df[(dates > pd.to_datetime(train_end).date()) & (dates <= pd.to_datetime(val_end).date())]
    test = df[dates > pd.to_datetime(val_end).date()]
    return train, val, test
```

Run test. Expected: PASS.

**Step 3: Commit**

```bash
git add dfw_temp_model/training/splits.py tests/test_splits.py
git commit -m "feat: add time-based train/val/test split"
```

---

## Task 11: Run First End-to-End Experiment

**Objective:** Download 2020-2023 data, fit baseline and advection models on train+val, evaluate on 2024 holdout, and print a comparison report.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/scripts/run_first_experiment.py`

**Step 1: Create experiment script**

The script should:
1. Fetch/cache ASOS obs for 2020-2024 (or read from cache if exists).
2. Fetch/cache Open-Meteo forecasts for 2020-2024.
3. Build residual table.
4. Split into train (2020-2023), val (2024), and test (2025 if available, else skip).
5. Fit baseline IDW on train+val.
6. Fit advection model on train+val using KDFW wind from observations.
7. Evaluate both on 2024 holdout.
8. Print metrics and save a CSV of results.

**Step 2: Run the script**

```bash
cd /opt/data/stock-research/dfw_temp_model
/opt/hermes/.venv/bin/python scripts/run_first_experiment.py
```

Expected output:
- Raw RMSE vs corrected RMSE for both models.
- Advection model RMSE <= baseline RMSE (or a clear explanation if not).
- A `data/results/2024_holdout_comparison.csv` file.

**Step 3: Inspect output**

Verify the CSV has columns: `date`, `obs`, `raw_fcst`, `baseline_corrected`, `advection_corrected`, `baseline_error`, `advection_error`.

**Step 4: Commit**

```bash
git add scripts/run_first_experiment.py data/results/2024_holdout_comparison.csv
git commit -m "feat: add first end-to-end experiment comparing baseline and advection models"
```

---

## Task 12: Parameter Sweep on Validation Set

**Objective:** Use the 2024 validation set to tune `p`, `boost`, `half_width`, and `l_adv_km` with a simple grid search.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/scripts/tune_advection.py`
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/training/tune.py`

**Step 1: Implement grid search**

```python
from itertools import product
from dfw_temp_model.models.advection import advection_predict
from dfw_temp_model.evaluation.metrics import rmse


def grid_search_advection(residuals, geom, wind_df, val_idx, target_col="KDFW"):
    best = None
    for p, boost, half_width, l_adv in product([1.5, 2.0, 3.0], [2.0, 3.0, 5.0], [30, 45, 60], [30, 50, 80]):
        pred = advection_predict(residuals, geom, wind_df, target_col=target_col, p=p, boost=boost, half_width=half_width, l_adv_km=l_adv)
        score = rmse(pred.loc[val_idx, target_col], pred.loc[val_idx, "corrected_temp_f"])
        if best is None or score < best["rmse"]:
            best = {"rmse": score, "p": p, "boost": boost, "half_width": half_width, "l_adv_km": l_adv}
    return best
```

**Step 2: Run tune script**

```bash
/opt/hermes/.venv/bin/python scripts/tune_advection.py
```

Expected: prints best parameter set and saves `data/results/best_params.json`.

**Step 3: Re-run holdout with best params**

Update `run_first_experiment.py` to load `best_params.json` and re-evaluate on test set. Compare vs baseline.

**Step 4: Commit**

```bash
git add dfw_temp_model/training/tune.py scripts/tune_advection.py data/results/best_params.json
git commit -m "feat: add advection model parameter grid search"
```

---

## Task 13: Add Front-Detection Branch

**Objective:** Detect strong residual gradients and switch to front-aware prediction during those days.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/models/advection.py`
- Modify: `/opt/data/stock-research/dfw_temp_model/tests/test_advection.py`

**Step 1: Add gradient-based front detector**

```python
def residual_gradient(residuals_row, geom):
    neighbors = [c for c in residuals_row.index if c != "KDFW"]
    X = geom.loc[neighbors, ["x_m", "y_m"]].values
    y = residuals_row[neighbors].values
    # weighted least squares with inverse distance weights
    w = 1.0 / geom.loc[neighbors, "dist_km"].values ** 2
    W = np.diag(w)
    beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return beta  # [a, b]
```

**Step 2: Add front-aware branch in predictor**

When `sqrt(a^2 + b^2) > G_front` (start with 0.08 °F/km), use the mean residual of upwind stations only.

**Step 3: Test**

Create a synthetic residual field with a north-south gradient and assert front detection triggers and the predictor shifts toward the upwind cluster.

**Step 4: Commit**

```bash
git add dfw_temp_model/models/advection.py tests/test_advection.py
git commit -m "feat: add front-detection branch to advection model"
```

---

## Task 14: Document Results and Save to GitHub Pages Repo

**Objective:** Summarize the first experiment in a research note, push to the GitHub Pages repo with the direct GitHub URL.

**Files:**
- Create: `/opt/data/DAlvarez101.HermesStocks.io/2026-06-16-dfw-model-first-experiment.md`

**Step 1: Write summary note**

Include:
- Data sources and date range.
- Baseline RMSE vs advection RMSE on 2024 holdout.
- Best parameters found.
- Front detection hit rate.
- Limitations.
- Cited sources.

**Step 2: Commit and push**

```bash
cd /opt/data/DAlvarez101.HermesStocks.io
git add 2026-06-16-dfw-model-first-experiment.md
git commit -m "docs: add DFW advection model first experiment results"
git push origin main
```

---

## Risks and Tradeoffs

| Risk | Mitigation |
|---|---|
| IEM ASOS API changes or is down | Cache all fetched data as Parquet; write fallback readers |
| Open-Meteo archive missing some days | Validate non-null counts per station/date; fill small gaps with linear interpolation if needed |
| Wind data itself is wrong (METAR errors) | Use median of neighbor wind reports, not just KDFW |
| Daily high timing differs between obs and forecast | Both use UTC-day max; note any timezone issues in docs |
| Overfitting to 2024 validation set | Use simple grid search, not a huge search space; reserve 2025 as final test |
| Heavy data downloads take too long | Start with one year (2024) for pipeline validation, then expand to 2020-2024 |

## Open Questions

1. Should the first experiment use all years 2020-2024, or start with 2024 only to validate the pipeline faster?
2. Is Open-Meteo archive sufficient, or should we invest in HRRR analysis immediately?
3. Should we predict the daily high, daily low, or both first?
4. How will we handle missing station days — drop the row or impute from neighbors?
5. What bucket width matches the actual Polymarket market structure for KDFW highs?

---

**Plan complete and saved. Ready to execute using subagent-driven-development — I'll dispatch a fresh subagent per task with two-stage review (spec compliance then code quality). Shall I proceed?**
