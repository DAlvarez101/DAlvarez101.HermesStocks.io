# Blended Forecast Dashboard Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add a bias-corrected 18-hour temperature forecast to the live weather dashboard by blending HRRR model output with recent METAR observations, designed so new forecast models (GFS, NAM) can be added later without touching the blending logic.

**Architecture:** A new `blending/` subpackage implements a provider-agnostic bias correction. A `ForecastProvider` protocol abstracts model sources. The `BiasCorrector` computes an exponentially-weighted rolling bias from METAR-HRRR overlaps. A new dashboard chart shows HRRR raw, bias-corrected, observed METAR points, and an uncertainty band. A dropdown lets the user switch between recent HRRR cycles to compare how earlier forecasts held up against observations.

**Tech Stack:** Python 3.13, pandas, numpy (already installed), plotly (already installed), SQLite (stdlib), zoneinfo (stdlib). No new dependencies.

---

## Current Context

### Database
SQLite at `data/cache/db/weather_observations.db`:
- `metar_observations`: 312 rows, 8 stations, columns: id, fetched_at, source, station, valid, lat, lon, tmpf, dewpf, drct, sknt, skyc1, mslp, p01i
- `hrrr_forecasts`: 1384 rows, 8 stations, 23 cycles, columns: id, fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf
- The `source` column in `hrrr_forecasts` is currently always `'hrrr-aws'` but is designed for multi-model use

### Data alignment
- METAR observations have `valid` timestamps like `2026-06-17T22:53:00+00:00`. An observation at :53 is valid for the 23Z hour (it reports conditions in the preceding hour). We floor to the hour for matching: `22:53 → 22:00`.
- HRRR forecasts have `valid_dt` timestamps at the top of each hour: `2026-06-17T23:00:00+00:00`.
- Merging on floored hour gives 81 overlapping rows for KDAL, with a measurable bias: mean(obs-fcst) = +0.71°F, std = 1.34°F.

### Existing code that must NOT be disrupted
- `scripts/generate_dashboard.py` — the existing weather dashboard generator (417 lines). We will add a new chart function and one template insertion, not rewrite existing functions.
- `scripts/ingest_live_metars.py` — the cron ingestion script. Untouched.
- `dfw_temp_model/data/hrrr.py` — the HRRR fetcher. Untouched.
- `dfw_temp_model/storage/obs_db.py` — the DB layer. Untouched.
- `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` — the cron script. Untouched (it already calls `generate_dashboard.py`, which will produce the new chart).

### Existing models code (reference, not modified)
- `dfw_temp_model/models/advection.py` — wind-advection-weighted residual model. Designed for daily highs, not hourly. Future enhancement, not part of this plan.
- `dfw_temp_model/models/baseline.py` — inverse-distance prediction. Same — daily resolution, future use.

### METAR-HRRR matching detail
The user noted: "data coming in at 22:53 would technically be valid for the 23UTC hour." This is the standard METAR convention — a METAR at :53 reports conditions for the upcoming hour. However, for bias correction purposes, the observation at 22:53 represents the temperature *at* 22:53, and the HRRR forecast valid at 23:00 represents the forecast *at* 23:00. The 7-minute offset is negligible for temperature bias purposes. We floor both to the same hour for matching. This is the same approach NWS uses for MOS verification.

### Design constraints from the user
1. "Don't disrupt much of the existing code" — additive only, no rewrites
2. "Dashboard should be easy to understand" — clear legend, labeled lines
3. "METAR points should line up with the appropriate hour of the HRRR forecast" — floor to hour, overlay on the same x-axis
4. "Consider a dropdown to show different runtimes of the HRRR" — Plotly dropdown to switch between recent complete cycles
5. "Code should be straightforward and robust" — no clever tricks, clear function names
6. "Minimize dependencies" — only use what's already installed (pandas, numpy, plotly)
7. "Ensure code is well formatted so new models can be swapped in" — provider protocol with HRRR as the first implementation

---

## Plan Overview

The plan has two workstreams that converge in the final task:

**Workstream A: Blending subpackage** (Tasks 1-4) — the core logic: provider protocol, bias corrector, blended forecast function, all with TDD tests.

**Workstream B: Dashboard chart** (Tasks 5-6) — the visualization: new Plotly chart with HRRR raw, bias-corrected, METAR overlay, uncertainty band, and cycle dropdown. Inserted into the existing dashboard.

**Task 7** — smoke test the full pipeline end-to-end.

---

## Workstream A: Blending Subpackage

### Task 1: Create the provider protocol and HRRR provider

**Objective:** Define the abstract interface that all forecast models implement, and wrap the existing HRRR data access as the first provider.

**Files:**
- Create: `dfw_temp_model/blending/__init__.py`
- Create: `dfw_temp_model/blending/providers.py`
- Test: `tests/test_blending_providers.py`

**Step 1: Create the blending package init**

`dfw_temp_model/blending/__init__.py`:
```python
"""Bias-correction blending for multi-model temperature forecasts."""
```

**Step 2: Write failing test**

`tests/test_blending_providers.py`:
```python
"""Tests for the forecast provider interface and HRRR provider."""
import pandas as pd
import pytest
import sqlite3
import tempfile
import os

from dfw_temp_model.blending.providers import ForecastProvider, HRRRProvider


def test_protocol_is_abstract():
    """ForecastProvider is a Protocol — any class with the right methods conforms."""
    # A simple class with fetch_forecast and recent_cycles should satisfy the protocol.
    class FakeProvider:
        def fetch_forecast(self, conn, station, init_dt, forecast_hours=18):
            return pd.DataFrame()
        def recent_cycles(self, conn, station, min_hours=18):
            return []
    # Protocol check — should not raise
    provider = FakeProvider()
    assert hasattr(provider, "fetch_forecast")
    assert hasattr(provider, "recent_cycles")


def test_hrrr_provider_returns_forecast():
    """HRRRProvider reads from the SQLite DB and returns a DataFrame."""
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
            (1, "2026-01-01T00:00:00Z", "hrrr-aws", "KDAL",
             "2026-01-01T00:00:00Z", 1, "2026-01-01T01:00:00Z", 32.0, -96.0, 80.0),
            (2, "2026-01-01T00:00:00Z", "hrrr-aws", "KDAL",
             "2026-01-01T00:00:00Z", 2, "2026-01-01T02:00:00Z", 32.0, -96.0, 82.0),
        ],
    )
    conn.commit()

    provider = HRRRProvider()
    df = provider.fetch_forecast(conn, "KDAL", "2026-01-01T00:00:00Z", forecast_hours=2)
    assert len(df) == 2
    assert "valid_dt" in df.columns
    assert "tmpf" in df.columns
    assert "forecast_hour" in df.columns
    conn.close()


def test_hrrr_provider_recent_cycles():
    """recent_cycles returns init_dt strings sorted newest first."""
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
    # Two cycles, one complete (18 hours) and one partial (8 hours)
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "hrrr-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    for fh in range(1, 9):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "hrrr-aws", "KDAL", "2026-01-01T06:00:00Z", fh, "t", 0, 0, 80),
        )
    conn.commit()

    provider = HRRRProvider()
    cycles = provider.recent_cycles(conn, "KDAL", min_hours=18)
    assert "2026-01-01T12:00:00Z" in cycles
    # The partial cycle should not appear when min_hours=18
    assert "2026-01-01T06:00:00Z" not in cycles
    conn.close()
```

**Step 3: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfw_temp_model.blending'`

**Step 4: Write minimal implementation**

`dfw_temp_model/blending/providers.py`:
```python
"""Forecast provider interface and implementations.

Each provider wraps a model source (HRRR, GFS, NAM, etc.) behind a common
interface so the blending logic can treat all models uniformly. The DB
schema already has a ``source`` column in ``hrrr_forecasts``; new models
will store their rows with a different source tag.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd
import sqlite3


@runtime_checkable
class ForecastProvider(Protocol):
    """Abstract interface for a forecast model source.

    Implementations read from the SQLite DB (or fetch live) and return
    DataFrames with at minimum: valid_dt, tmpf, forecast_hour, init_dt.
    """

    def fetch_forecast(
        self,
        conn: sqlite3.Connection,
        station: str,
        init_dt: str,
        forecast_hours: int = 18,
    ) -> pd.DataFrame:
        """Return forecast rows for one model cycle at one station."""
        ...

    def recent_cycles(
        self,
        conn: sqlite3.Connection,
        station: str,
        min_hours: int = 18,
    ) -> list[str]:
        """Return init_dt strings (newest first) with at least min_hours frames."""
        ...


class HRRRProvider:
    """Reads HRRR forecasts from the SQLite ``hrrr_forecasts`` table.

    This is a thin wrapper around the existing storage queries. The actual
    HRRR fetching (downloading GRIB2 from AWS) lives in
    ``dfw_temp_model.data.hrrr`` and is not duplicated here.
    """

    SOURCE = "hrrr-aws"

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
            WHERE station = ? AND init_dt = ?
            ORDER BY forecast_hour ASC
            """,
            conn,
            params=[station, init_dt],
        )
        return df

    def recent_cycles(
        self,
        conn: sqlite3.Connection,
        station: str,
        min_hours: int = 18,
    ) -> list[str]:
        """Return init_dt strings that have at least min_hours of frames.

        Sorted newest-first. Only includes complete cycles.
        """
        df = pd.read_sql_query(
            """
            SELECT init_dt, COUNT(*) AS n
            FROM hrrr_forecasts
            WHERE station = ?
            GROUP BY init_dt
            HAVING n >= ?
            ORDER BY init_dt DESC
            """,
            conn,
            params=[station, min_hours],
        )
        if df.empty:
            return []
        return df["init_dt"].tolist()
```

**Step 5: Run test to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_providers.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/blending/__init__.py dfw_temp_model/blending/providers.py tests/test_blending_providers.py
git commit -m "feat: add ForecastProvider protocol and HRRRProvider for model-agnostic blending"
```

---

### Task 2: Create the bias corrector

**Objective:** Implement an exponentially-weighted rolling bias estimator that computes the correction from recent METAR-HRRR overlaps.

**Files:**
- Create: `dfw_temp_model/blending/bias.py`
- Test: `tests/test_blending_bias.py`

**Step 1: Write failing test**

`tests/test_blending_bias.py`:
```python
"""Tests for the rolling bias corrector."""
import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.blending.bias import (
    compute_rolling_bias,
    apply_bias_correction,
)


def test_compute_rolling_bias_basic():
    """Bias = observed - forecast, returned as a rolling exponentially-weighted mean."""
    obs = pd.DataFrame({
        "valid_hour": pd.to_datetime([
            "2026-06-17T18:00:00Z",
            "2026-06-17T19:00:00Z",
            "2026-06-17T20:00:00Z",
        ], utc=True),
        "tmpf_obs": [89.0, 90.0, 91.0],
    })
    fcst = pd.DataFrame({
        "valid_hour": pd.to_datetime([
            "2026-06-17T18:00:00Z",
            "2026-06-17T19:00:00Z",
            "2026-06-17T20:00:00Z",
        ], utc=True),
        "tmpf_fcst": [88.0, 89.0, 90.0],
    })
    result = compute_rolling_bias(obs, fcst, halflife_hours=6.0)
    assert "bias" in result.columns
    assert "n_matches" in result.columns
    # Each bias = obs - fcst = 1.0
    assert result["bias"].iloc[-1] == pytest.approx(1.0, abs=0.01)
    assert result["n_matches"].iloc[-1] == 3


def test_compute_rolling_bias_empty():
    """No overlaps → empty result with correct columns."""
    obs = pd.DataFrame({"valid_hour": pd.to_datetime([], utc=True), "tmpf_obs": []})
    fcst = pd.DataFrame({"valid_hour": pd.to_datetime([], utc=True), "tmpf_fcst": []})
    result = compute_rolling_bias(obs, fcst, halflife_hours=6.0)
    assert result.empty
    assert "bias" in result.columns
    assert "bias_std" in result.columns


def test_apply_bias_correction():
    """Corrected = raw forecast + rolling bias."""
    forecast = pd.DataFrame({
        "valid_dt": pd.to_datetime([
            "2026-06-17T21:00:00Z",
            "2026-06-17T22:00:00Z",
            "2026-06-17T23:00:00Z",
        ], utc=True),
        "tmpf": [88.0, 87.0, 86.0],
        "forecast_hour": [1, 2, 3],
    })
    bias_df = pd.DataFrame({
        "valid_hour": [pd.Timestamp("2026-06-17T20:00:00Z", tz="UTC")],
        "bias": [1.5],
        "bias_std": [0.5],
        "n_matches": [5],
    })
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=1.0)
    assert "tmpf_corrected" in result.columns
    assert "uncertainty_low" in result.columns
    assert "uncertainty_high" in result.columns
    # Corrected = 88 + 1.5 = 89.5
    assert result["tmpf_corrected"].iloc[0] == pytest.approx(89.5, abs=0.01)
    # Uncertainty band = ±bias_std
    assert result["uncertainty_low"].iloc[0] == pytest.approx(89.0, abs=0.01)
    assert result["uncertainty_high"].iloc[0] == pytest.approx(90.0, abs=0.01)


def test_apply_bias_correction_no_bias():
    """If no bias data, corrected = raw, uncertainty from raw spread."""
    forecast = pd.DataFrame({
        "valid_dt": pd.to_datetime(["2026-06-17T21:00:00Z"], utc=True),
        "tmpf": [88.0],
        "forecast_hour": [1],
    })
    bias_df = pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=1.0)
    # No bias → corrected = raw
    assert result["tmpf_corrected"].iloc[0] == pytest.approx(88.0, abs=0.01)
    # Uncertainty should be some default (not NaN)
    assert not np.isnan(result["uncertainty_low"].iloc[0])
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_bias.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfw_temp_model.blending.bias'`

**Step 3: Write minimal implementation**

`dfw_temp_model/blending/bias.py`:
```python
"""Rolling bias correction for model-agnostic forecast blending.

Computes the exponentially-weighted rolling mean of (observed - forecast)
at each valid hour, then applies that bias as an additive correction to
future forecast hours. The bias is provider-specific: HRRR has its own
bias, GFS would have its own, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _floor_to_hour(ts: pd.Series) -> pd.Series:
    """Floor a datetime Series to the top of the hour."""
    return pd.to_datetime(ts, utc=True).dt.floor("h")


def compute_rolling_bias(
    obs_df: pd.DataFrame,
    fcst_df: pd.DataFrame,
    halflife_hours: float = 6.0,
) -> pd.DataFrame:
    """Compute the rolling bias (observed - forecast) from matched hours.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Must have columns ``valid_hour`` (datetime, UTC) and ``tmpf_obs`` (float).
    fcst_df : pd.DataFrame
        Must have columns ``valid_hour`` (datetime, UTC) and ``tmpf_fcst`` (float).
    halflife_hours : float
        Half-life of the exponential decay in hours. Recent observations
        weigh more. Default 6 hours means an observation 6 hours old has
        half the weight of the current one.

    Returns
    -------
    pd.DataFrame
        Columns: ``valid_hour`` (datetime UTC), ``bias`` (float, the EWMA of
        obs - fcst), ``bias_std`` (float, rolling std for uncertainty),
        ``n_matches`` (int, cumulative count of matched hours).
    """
    if obs_df.empty or fcst_df.empty:
        return pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])

    obs = obs_df.copy()
    fcst = fcst_df.copy()
    obs["valid_hour"] = _floor_to_hour(obs["valid_hour"])
    fcst["valid_hour"] = _floor_to_hour(fcst["valid_hour"])

    # Merge on valid_hour (many-to-one if multiple cycles match the same obs hour)
    merged = obs.merge(fcst, on="valid_hour", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])

    # If multiple forecast cycles match the same obs hour, take the mean.
    merged["error"] = merged["tmpf_obs"] - merged["tmpf_fcst"]
    hourly = merged.groupby("valid_hour").agg(
        error_mean=("error", "mean"),
        error_std=("error", "std"),
        n=("error", "count"),
    ).reset_index()
    hourly = hourly.sort_values("valid_hour")

    # Exponentially-weighted moving average of the bias.
    # Use adjust=False so the EWMA is recursive (favors recent data).
    halflife_td = pd.Timedelta(hours=halflife_hours)
    # Convert halflife to span for ewm
    # span = 2 * halflife (in number of samples, assuming 1-hour spacing)
    span = max(1, int(2 * halflife_hours))

    hourly["bias"] = hourly["error_mean"].ewm(
        span=span, adjust=False, min_periods=1
    ).mean()
    # Rolling std (expanding, with at least 2 samples)
    hourly["bias_std"] = hourly["error_std"].fillna(0.0)
    # If only 1 sample, use a default uncertainty of 1.0°F
    hourly.loc[hourly["n"] == 1, "bias_std"] = 1.0
    hourly["n_matches"] = hourly["n"].cumsum()

    return hourly[["valid_hour", "bias", "bias_std", "n_matches"]]


def apply_bias_correction(
    forecast: pd.DataFrame,
    bias_df: pd.DataFrame,
    uncertainty_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Apply the latest rolling bias to a forecast and add uncertainty bands.

    Parameters
    ----------
    forecast : pd.DataFrame
        Must have ``valid_dt`` (datetime UTC) and ``tmpf`` (float).
    bias_df : pd.DataFrame
        Output of ``compute_rolling_bias``. The *latest* bias value is used
        as a constant correction for all future forecast hours.
    uncertainty_multiplier : float
        Multiplier for the bias_std to form the uncertainty band. 1.0 = ±1σ.

    Returns
    -------
    pd.DataFrame
        Copy of ``forecast`` with added columns: ``tmpf_corrected``,
        ``uncertainty_low``, ``uncertainty_high``, ``bias_applied``.
    """
    result = forecast.copy()
    result["valid_dt"] = pd.to_datetime(result["valid_dt"], utc=True)

    if bias_df.empty:
        # No bias data: corrected = raw, default uncertainty
        result["tmpf_corrected"] = result["tmpf"]
        result["bias_applied"] = 0.0
        default_unc = 2.0  # 2°F default when we have no bias estimate
        result["uncertainty_low"] = result["tmpf_corrected"] - default_unc
        result["uncertainty_high"] = result["tmpf_corrected"] + default_unc
        return result

    # Use the latest bias value as a constant for all future hours.
    latest = bias_df.iloc[-1]
    bias = float(latest["bias"])
    bias_std = float(latest["bias_std"]) * uncertainty_multiplier

    result["tmpf_corrected"] = result["tmpf"] + bias
    result["bias_applied"] = bias
    result["uncertainty_low"] = result["tmpf_corrected"] - bias_std
    result["uncertainty_high"] = result["tmpf_corrected"] + bias_std
    return result
```

**Step 4: Run test to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_bias.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/blending/bias.py tests/test_blending_bias.py
git commit -m "feat: add rolling bias corrector with exponentially-weighted EWMA"
```

---

### Task 3: Create the blended forecast orchestrator

**Objective:** A single function that reads from the DB, matches METAR to HRRR, computes the bias, and returns a blended corrected forecast — all provider-agnostic.

**Files:**
- Create: `dfw_temp_model/blending/blend.py`
- Test: `tests/test_blending_blend.py`

**Step 1: Write failing test**

`tests/test_blending_blend.py`:
```python
"""Tests for the blended forecast orchestrator."""
import sqlite3
import pandas as pd
import pytest

from dfw_temp_model.blending.blend import blended_forecast, list_recent_cycles
from dfw_temp_model.blending.providers import HRRRProvider


def _make_db():
    """Create an in-memory DB with METAR + HRRR data for testing."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE metar_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT, valid TEXT,
            lat REAL, lon REAL, tmpf REAL, dewpf REAL, drct REAL,
            sknt REAL, skyc1 TEXT, mslp REAL, p01i REAL,
            UNIQUE(source, station, valid)
        );
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station)
        );
    """)

    # METAR observations for KDAL: hours 12-20Z, temps 80-91°F
    for h in range(12, 21):
        conn.execute(
            "INSERT INTO metar_observations VALUES (NULL,'t','aviationweather','KDAL',?,?,32,-96,0,0,0,0,NULL,0,NULL)",
            (f"2026-06-17T{h:02d}:53:00+00:00", 80.0 + h),
        )

    # HRRR forecast cycle init at 18Z, f01-f18 → valid 19Z..12Z next day
    # For overlap with METAR, only hours 19-20Z will match (METAR has 12-20Z)
    for fh in range(1, 19):
        valid_h = (18 + fh) % 24
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (NULL,'t','hrrr-aws','KDAL','2026-06-17T18:00:00+00:00',?,?,32,-96,0)",
            (fh, f"2026-06-17T{valid_h:02d}:00:00+00:00", 79.0 + fh),
        )
    conn.commit()
    return conn


def test_blended_forecast_returns_correct_columns():
    """blended_forecast returns a DataFrame with all expected columns."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(conn, "KDAL", provider, init_dt="2026-06-17T18:00:00+00:00")
    assert "tmpf" in result.columns          # raw HRRR
    assert "tmpf_corrected" in result.columns  # bias-corrected
    assert "uncertainty_low" in result.columns
    assert "uncertainty_high" in result.columns
    assert "forecast_hour" in result.columns
    assert "valid_dt" in result.columns
    conn.close()


def test_blended_forecast_bias_is_nonzero():
    """With real METAR-HRRR overlap, the bias should be non-zero."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(conn, "KDAL", provider, init_dt="2026-06-17T18:00:00+00:00")
    # METAR at 19Z = 99°F, HRRR valid 19Z (f01) = 80°F → bias = 19°F
    # METAR at 20Z = 100°F, HRRR valid 20Z (f02) = 81°F → bias = 19°F
    # So the corrected forecast should be raw + ~19
    assert result["tmpf_corrected"].iloc[0] > result["tmpf"].iloc[0]
    conn.close()


def test_blended_forecast_no_overlap():
    """If no METAR data overlaps the forecast, corrected = raw."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE metar_observations (
            id INTEGER PRIMARY KEY, fetched_at TEXT, source TEXT,
            station TEXT, valid TEXT, lat REAL, lon REAL, tmpf REAL,
            dewpf REAL, drct REAL, sknt REAL, skyc1 TEXT, mslp REAL, p01i REAL,
            UNIQUE(source, station, valid)
        );
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY, fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station)
        );
    """)
    # HRRR only, no METARs
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (NULL,'t','hrrr','KDAL','2026-06-17T18:00:00Z',?,?,0,0,80)",
            (fh, f"2026-06-17T{(18+fh)%24:02d}:00:00Z"),
        )
    conn.commit()
    provider = HRRRProvider()
    result = blended_forecast(conn, "KDAL", provider, init_dt="2026-06-17T18:00:00Z")
    # No bias → corrected = raw
    assert result["tmpf_corrected"].iloc[0] == pytest.approx(80.0, abs=0.01)
    conn.close()


def test_list_recent_cycles():
    """list_recent_cycles returns available complete cycles."""
    conn = _make_db()
    provider = HRRRProvider()
    cycles = list_recent_cycles(conn, "KDAL", provider, min_hours=18)
    assert len(cycles) >= 1
    assert "2026-06-17T18:00:00+00:00" in cycles
    conn.close()
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_blend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfw_temp_model.blending.blend'`

**Step 3: Write minimal implementation**

`dfw_temp_model/blending/blend.py`:
```python
"""Orchestrator: read DB, match METAR to forecast, compute bias, correct.

This is the top-level entry point for the blending pipeline. It is
provider-agnostic: pass in any ForecastProvider and it will read that
provider's forecasts from the DB, match against METAR observations,
compute the rolling bias, and return a corrected forecast.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from dfw_temp_model.blending.bias import (
    apply_bias_correction,
    compute_rolling_bias,
)
from dfw_temp_model.blending.providers import ForecastProvider


def _load_metar_for_station(conn: sqlite3.Connection, station: str) -> pd.DataFrame:
    """Load METAR observations for a station, floored to the hour."""
    df = pd.read_sql_query(
        """
        SELECT valid, tmpf
        FROM metar_observations
        WHERE station = ?
        ORDER BY valid
        """,
        conn,
        params=[station],
    )
    if df.empty:
        return pd.DataFrame(columns=["valid_hour", "tmpf_obs"])
    df["valid_hour"] = pd.to_datetime(df["valid"], utc=True).dt.floor("h")
    # If multiple obs in the same hour, take the latest one
    df = df.sort_values("valid").groupby("valid_hour").tail(1)
    df = df.rename(columns={"tmpf": "tmpf_obs"})
    return df[["valid_hour", "tmpf_obs"]]


def _load_forecast_for_matching(
    conn: sqlite3.Connection,
    provider: ForecastProvider,
    station: str,
    cycles: list[str],
) -> pd.DataFrame:
    """Load forecast rows from multiple cycles for bias matching.

    We use all recent cycles (not just the latest) so the bias estimate
    benefits from more data points. Each cycle contributes its own
    (forecast, observation) pairs at matching valid hours.
    """
    frames = []
    for init_dt in cycles:
        df = provider.fetch_forecast(conn, station, init_dt)
        if df.empty:
            continue
        df["valid_hour"] = pd.to_datetime(df["valid_dt"], utc=True).dt.floor("h")
        df = df.rename(columns={"tmpf": "tmpf_fcst"})
        frames.append(df[["valid_hour", "tmpf_fcst"]])
    if not frames:
        return pd.DataFrame(columns=["valid_hour", "tmpf_fcst"])
    return pd.concat(frames, ignore_index=True)


def blended_forecast(
    conn: sqlite3.Connection,
    station: str,
    provider: ForecastProvider,
    init_dt: str | None = None,
    halflife_hours: float = 6.0,
    uncertainty_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Compute a bias-corrected forecast for a station.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection to the weather DB.
    station : str
        ICAO code (e.g. ``"KDAL"``).
    provider : ForecastProvider
        The model provider (HRRR, GFS, etc.).
    init_dt : str, optional
        The model cycle to correct. If None, uses the latest complete cycle.
    halflife_hours : float
        Half-life of the exponential bias decay. Recent observations
        matter more.
    uncertainty_multiplier : float
        Multiplier for the bias std to form the uncertainty band.

    Returns
    -------
    pd.DataFrame
        One row per forecast hour with columns: ``valid_dt``, ``tmpf``
        (raw), ``tmpf_corrected``, ``uncertainty_low``, ``uncertainty_high``,
        ``forecast_hour``, ``bias_applied``, ``init_dt``.
    """
    # Determine which cycle to correct
    if init_dt is None:
        cycles = provider.recent_cycles(conn, station, min_hours=18)
        if not cycles:
            return pd.DataFrame()
        init_dt = cycles[0]  # newest first

    # Load the forecast to correct
    forecast = provider.fetch_forecast(conn, station, init_dt)
    if forecast.empty:
        return pd.DataFrame()

    # Load all recent cycles for bias matching (more data = better bias)
    all_cycles = provider.recent_cycles(conn, station, min_hours=1)
    if not all_cycles:
        all_cycles = [init_dt]

    # Load METAR observations
    obs_df = _load_metar_for_station(conn, station)

    # Load all forecast data for matching (from all recent cycles)
    fcst_for_matching = _load_forecast_for_matching(conn, provider, station, all_cycles)

    # Compute rolling bias
    bias_df = compute_rolling_bias(obs_df, fcst_for_matching, halflife_hours=halflife_hours)

    # Apply bias correction
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=uncertainty_multiplier)

    return result


def list_recent_cycles(
    conn: sqlite3.Connection,
    station: str,
    provider: ForecastProvider,
    min_hours: int = 18,
) -> list[str]:
    """Convenience wrapper: list available complete forecast cycles."""
    return provider.recent_cycles(conn, station, min_hours=min_hours)
```

**Step 4: Run test to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_blend.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/blending/blend.py tests/test_blending_blend.py
git commit -m "feat: add blended_forecast orchestrator for provider-agnostic bias correction"
```

---

### Task 4: Verify the local package imports in the venv

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "import dfw_temp_model.blending; print('OK')"`
Expected: `OK`

If it fails, run: `cd /opt/data/stock-research/dfw_temp_model && uv pip install -e . && .venv/bin/python -c "import dfw_temp_model.blending; print('OK')"`

---

## Workstream B: Dashboard Chart

### Task 5: Create the blended forecast chart function

**Objective:** Add a new Plotly chart to `generate_dashboard.py` that shows HRRR raw, bias-corrected, METAR observations, uncertainty band, and a dropdown to switch between recent HRRR cycles.

**Files:**
- Modify: `scripts/generate_dashboard.py` (add one new function ~120 lines, add one line to the HTML template, add one line to the `generate_dashboard()` format call)

**Step 1: Add the chart function**

Insert the following function into `scripts/generate_dashboard.py` after the existing `hrrr_forecast_chart` function (around line 274):

```python
def blended_forecast_chart(conn) -> str:
    """Interactive Plotly chart: HRRR raw vs bias-corrected vs METAR observations.

    Includes a dropdown to switch between recent complete HRRR cycles.
    METAR observations are overlaid at their floored-hour positions so they
    visually align with the HRRR forecast at the same valid hour.
    """
    from dfw_temp_model.blending.blend import blended_forecast, list_recent_cycles
    from dfw_temp_model.blending.providers import HRRRProvider

    provider = HRRRProvider()
    cycles = list_recent_cycles(conn, TARGET_ICAO, provider, min_hours=18)
    if not cycles:
        return "<p>No complete HRRR forecast cycles available</p>"

    # Limit to the 5 most recent cycles for the dropdown
    cycles = cycles[:5]

    # Load METAR observations for overlay
    metar_df = pd.read_sql_query(
        "SELECT valid, tmpf FROM metar_observations WHERE station = ? ORDER BY valid",
        conn,
        params=[TARGET_ICAO],
    )
    if not metar_df.empty:
        metar_df["valid"] = pd.to_datetime(metar_df["valid"], utc=True)
        metar_df["valid_hour"] = metar_df["valid"].dt.floor("h")
        # Take the latest observation per hour
        metar_hourly = metar_df.sort_values("valid").groupby("valid_hour").tail(1)
        metar_hourly["ct_label"] = metar_hourly["valid"].apply(
            lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")
        )
    else:
        metar_hourly = pd.DataFrame()

    # Build one trace-set per cycle for the dropdown
    fig = go.Figure()

    # Add METAR observations (always visible, shared across all dropdown views)
    if not metar_hourly.empty:
        fig.add_trace(go.Scatter(
            x=metar_hourly["valid_hour"],
            y=metar_hourly["tmpf"],
            mode="markers",
            name="METAR observed",
            marker={"size": 8, "color": "#38bdf8", "symbol": "circle"},
            hovertemplate=(
                "<b>METAR</b><br>"
                "%{x|%Y-%m-%d %H:%M UTC}<br>"
                f"%{{customdata}}<br>"
                "Temp: %{y:.1f}°F<extra></extra>"
            ),
            customdata=metar_hourly.get("ct_label", ""),
            visible=True,
        ))

    # For each cycle, add: raw HRRR, corrected, uncertainty band
    # We toggle visibility via dropdown buttons
    n_metar = 1 if not metar_hourly.empty else 0
    dropdown_buttons = []

    for i, cycle_dt in enumerate(cycles):
        blended = blended_forecast(conn, TARGET_ICAO, provider, init_dt=cycle_dt)
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

        # Raw HRRR
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"],
            y=blended["tmpf"],
            mode="lines+markers",
            name=f"HRRR raw (cycle {i+1})",
            line={"color": "#f59e0b", "width": 2, "dash": "dot"},
            marker={"size": 5, "color": "#f59e0b"},
            hovertemplate=(
                f"<b>HRRR raw</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                f"%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>"
                f"Cycle: {init_label}<extra></extra>"
            ),
            customdata=ct_labels,
            visible=(i == 0),
        ))

        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"].tolist() + blended["valid_dt"].tolist()[::-1],
            y=blended["uncertainty_high"].tolist() + blended["uncertainty_low"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.12)",
            line={"color": "rgba(34, 197, 94, 0)", "width": 0},
            name=f"Uncertainty (cycle {i+1})",
            hoverinfo="skip",
            visible=(i == 0),
            showlegend=False,
        ))

        # Bias-corrected
        bias_val = blended["bias_applied"].iloc[0] if "bias_applied" in blended.columns else 0
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"],
            y=blended["tmpf_corrected"],
            mode="lines+markers",
            name=f"Corrected (cycle {i+1})",
            line={"color": "#22c55e", "width": 2.5},
            marker={"size": 6, "color": "#22c55e"},
            hovertemplate=(
                f"<b>Corrected</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                f"%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>"
                f"Bias: {bias_val:+.1f}°F<br>"
                f"Cycle: {init_label} · {init_ct}<extra></extra>"
            ),
            customdata=ct_labels,
            visible=(i == 0),
        ))

        # Build visibility list for this dropdown option
        # n_metar traces are always visible, then 3 per cycle (raw, band, corrected)
        n_traces_per_cycle = 3
        total_traces = n_metar + len(cycles) * n_traces_per_cycle
        visibility = [True] * n_metar  # METAR always on
        for j in range(len(cycles)):
            if j == i:
                visibility.extend([True, True, True])
            else:
                visibility.extend([False, False, False])

        dropdown_buttons.append(dict(
            label=init_ts.strftime("%m/%d %H:00Z"),
            method="update",
            args=[{"visible": visibility}],
        ))

    # Set initial visibility to the first cycle
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
        title=f"Bias-Corrected Forecast — {TARGET_ICAO}<br><sup>HRRR raw (orange) vs corrected (green) vs METAR (blue)</sup>",
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

**Step 2: Add the chart to the HTML template**

In the `HTML_TEMPLATE` string, add after the existing HRRR forecast chart section (after line `{hrrr_chart}`):

Find the line:
```html
    <h2>HRRR 18-hour forecast ({TARGET_ICAO})</h2>
    {hrrr_chart}
```

Add after it:
```html

    <h2>Bias-Corrected Forecast ({TARGET_ICAO})</h2>
    {blended_chart}
```

**Step 3: Add the blended chart to the generate_dashboard function call**

In the `generate_dashboard()` function, find the `html = HTML_TEMPLATE.format(...)` call and add:

```python
        blended_chart=blended_forecast_chart(get_db(db_path)),
```

after the existing `hrrr_chart=hrrr_forecast_chart(get_db(db_path)),` line.

**Step 4: Run the dashboard generator and verify**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/generate_dashboard.py --db data/cache/db/weather_observations.db --output-dir /tmp/dfw-blend-test 2>&1`

Expected: `Dashboard written to: /tmp/dfw-blend-test/index.html`

Run: `grep -c 'blended' /tmp/dfw-blend-test/index.html && grep -c 'Corrected' /tmp/dfw-blend-test/index.html && grep -c 'updatemenus' /tmp/dfw-blend-test/index.html`

Expected: All > 0 (the chart is present, the dropdown is present)

**Step 5: Verify visually (if browser available)**

Open `/tmp/dfw-blend-test/index.html` in a browser and check:
- Orange dotted line: HRRR raw forecast
- Green solid line: bias-corrected forecast
- Blue circles: METAR observations (aligned to the hour)
- Green shaded band: uncertainty
- Dropdown: switch between recent cycles
- METAR points line up with the correct hour on the x-axis

**Step 6: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add scripts/generate_dashboard.py
git commit -m "feat: add bias-corrected forecast chart with METAR overlay and cycle dropdown"
```

---

### Task 6: Run the full pipeline via the cron script

**Objective:** Verify the cron script generates the dashboard with the new chart and pushes it to GitHub Pages.

**Step 1: Run the cron script**

Run: `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh 2>&1 | tail -15`

Expected: Output includes "Generating dashboard ...", "Dashboard written to: ...", git commit, git push, "Done."

**Step 2: Verify the live page**

Wait ~30 seconds for GitHub Pages to propagate, then:

Run: `curl -sI https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/ | head -3`

Expected: `HTTP/2 200`

Run: `curl -sL https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/ | grep -c 'Corrected'`

Expected: > 0 (the blended chart is live)

**Step 3: Commit all remaining changes**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add -A
git commit -m "feat: bias-corrected blended forecast dashboard with cycle dropdown"
git push origin main
```

---

## Task 7: Full test suite and smoke test

**Step 1: Run all blending tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_*.py -v`

Expected: All passed (11 tests total)

**Step 2: Run non-network test suite**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow" --tb=short`

Expected: 0 failed

**Step 3: Dry-run the trading bot (regression check)**

Run: `cd /opt/data/stock-research/dfw_temp_model && POLYMARKET_PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000001 .venv/bin/python scripts/run_polymarket_bot.py --dry-run 2>&1 | head -3`

Expected: JSON output starting with `{`

---

## Files Changed Summary

| File | Action | Description |
|------|--------|-------------|
| `dfw_temp_model/blending/__init__.py` | Create | Package init |
| `dfw_temp_model/blending/providers.py` | Create | ForecastProvider protocol + HRRRProvider |
| `dfw_temp_model/blending/bias.py` | Create | Rolling bias corrector (EWMA) |
| `dfw_temp_model/blending/blend.py` | Create | Orchestrator: DB → bias → corrected forecast |
| `tests/test_blending_providers.py` | Create | Provider tests (3 tests) |
| `tests/test_blending_bias.py` | Create | Bias corrector tests (4 tests) |
| `tests/test_blending_blend.py` | Create | Orchestrator tests (4 tests) |
| `scripts/generate_dashboard.py` | Modify | Add `blended_forecast_chart()` function, 1 template line, 1 format call line |

## What is NOT touched

- `scripts/ingest_live_metars.py` — untouched
- `scripts/generate_db_viewer.py` — untouched
- `dfw_temp_model/data/hrrr.py` — untouched
- `dfw_temp_model/storage/obs_db.py` — untouched
- `dfw_temp_model/models/advection.py` — untouched (future enhancement)
- `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` — untouched (already calls generate_dashboard.py)
- All trading code — untouched

## How to add a new model later

To add GFS as a second forecast source:

1. Create `dfw_temp_model/blending/gfs_provider.py` with a `GFSProvider` class that implements the `ForecastProvider` protocol (same methods: `fetch_forecast`, `recent_cycles`). Store rows in the DB with `source='gfs'`.
2. In the dashboard chart function, create a second `GFSProvider()` and call `blended_forecast()` with it. Add its traces to the Plotly chart.
3. The bias correction is automatic — each provider gets its own bias from its own forecast rows.
4. For multi-model blending (weighted average of HRRR + GFS), add a `MultiModelBlender` class that calls `blended_forecast()` for each provider and weights the results by inverse RMSE. This is a future task, not part of this plan.

## Risks and Mitigations

1. **Too few METAR-HRRR overlap points** — Currently 81 matched rows for KDAL. The EWMA with a 6-hour halflife weights recent hours heavily, so even 3-4 matched hours give a usable bias. If no overlap exists, `apply_bias_correction` falls back to raw forecast with a 2°F default uncertainty.

2. **METAR at :53 vs HRRR at :00** — The 7-minute offset is negligible for temperature. Both are floored to the same hour. This is the same convention NWS uses.

3. **Multiple forecast cycles matching the same observation** — Handled: `compute_rolling_bias` groups by `valid_hour` and takes the mean error, so if 5 different HRRR cycles all forecast the 20Z hour, the bias uses the mean of all 5 errors at that hour.

4. **Plotly chart size** — The blended chart adds another Plotly div. The page already loads Plotly via CDN (from the existing HRRR chart). The new chart uses `include_plotlyjs=False` to avoid loading it twice.

5. **Cron script** — No changes needed. The cron script already calls `generate_dashboard.py`, which will now produce the blended chart automatically.

## Open Questions

- Should the DB viewer also show the blended forecast? Left out for now — the DB viewer is read-only raw data. The blended forecast is a derived product that belongs on the main dashboard.
- Should the blended forecast be stored in the DB for the trading bot to use? Yes, eventually — but that's a separate task. The trading bot's `signal.py` can call `blended_forecast()` directly in the meantime.