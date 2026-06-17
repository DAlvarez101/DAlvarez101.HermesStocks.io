# AviationWeather.gov METAR JSON Fetcher — Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. Combine TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add a live, free, no-API-key AviationWeather.gov METAR JSON fetcher to the DFW temperature model so the pipeline can ingest current airport observations for all 8 stations.

**Architecture:** Implement a thin `dfw_temp_model.data.aviationweather` module that calls the AviationWeather.gov `/api/data/metar` JSON endpoint for a list of ICAO stations and a recent hour window. Parse temperature, dewpoint, wind direction/speed, sky cover, and observation time into the same DataFrame schema used by the IEM ASOS fetcher. Cache results to Parquet and add a small CLI/script entry point for live polling.

**Tech Stack:** Python 3.13, pandas, requests (already installed). No new dependencies.

---

## Current context

The project already has:
- `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/iem_asos.py` — historical hourly ASOS fetcher.
- `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/openmeteo.py` — historical forecast fetcher.
- `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/config.py` — `STATIONS`, `TARGET_ICAO`, `CACHE_DIR`.
- `/opt/data/stock-research/dfw_temp_model/scripts/run_first_experiment.py` — historical experiment runner.
- Tests live under `/opt/data/stock-research/dfw_temp_model/tests/` and run with `pytest`.

AviationWeather.gov endpoint to use:
```
https://aviationweather.gov/api/data/metar?ids=KDFW,KDAL,KADS,KAFW,KDTO,KGKY,KACT,KTYR&format=json&hours=2
```
Returns a JSON array of METAR objects, each containing `icaoId`, `obsTime`, `temp`, `dewp`, `wdir`, `wspd`, `cldCvg`, etc.

---

## Step-by-step plan

### Task 1: Add `aviationweather.py` data fetcher

**Objective:** Create a module that fetches and parses live METAR JSON from AviationWeather.gov.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/data/aviationweather.py`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_aviationweather.py`

**Step 1: Write failing tests**

In `tests/test_aviationweather.py`:
- `test_build_aviationweather_url`: assert URL contains all station IDs, `format=json`, and `hours=2`.
- `test_parse_metar_json_empty`: empty list returns empty DataFrame with expected columns.
- `test_parse_metar_json_sample`: synthetic JSON object with KDFW 30°C, 10 kt wind, few clouds → DataFrame row with `station=KDFW`, `tmpf=86.0`, `drct=0.0`, `sknt=10.0`, valid timestamp.
- `test_fetch_aviationweather_smoke`: real network call, assert non-empty result for the 8 stations, cached to a temp Parquet file.

**Step 2: Run test to verify failure**

```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python -m pytest tests/test_aviationweather.py -v
```
Expected: `ModuleNotFoundError` / assertion failures.

**Step 3: Implement `aviationweather.py`**

Functions:
- `build_aviationweather_url(stations, hours=2) -> str`
- `parse_metar_json(payload: list[dict]) -> pd.DataFrame`
  Map fields:
  - `icaoId` → `station`
  - `obsTime` (ISO 8601) → `valid` (UTC datetime)
  - `temp` (°C) → `tmpf` (°F): `temp * 9/5 + 32`
  - `dewp` (°C) → `dwpf` (°F)
  - `wdir` (°) → `drct`
  - `wspd` (kts) → `sknt`
  - `cldCvg` first element or `cldCvg1` → `skyc1` as string if present
  - Add constant `lat`, `lon` per station from `config.STATIONS` lookup (AviationWeather does not return coords in this endpoint).
- `fetch_aviationweather(stations, hours=2, cache_path=None, timeout=30) -> pd.DataFrame`
  Call URL, parse JSON, convert schema, optionally cache to Parquet.

**Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_aviationweather.py -v
```
Expected: 4 passed.

**Step 5: Commit**

```bash
git add dfw_temp_model/data/aviationweather.py tests/test_aviationweather.py
git commit -m "feat: add AviationWeather.gov live METAR JSON fetcher"
```

---

### Task 2: Add a live-polling script entry point

**Objective:** Provide a simple command a non-technical user can ask Hermes to run to get current conditions.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/scripts/fetch_live_metars.py`

**Step 1: Write failing test (if needed) / smoke test**

- `tests/test_live_script.py`:
  - `test_live_script_runs`: run the script with project venv, assert it prints current METAR rows for KDFW and saves to `data/cache/live_metars.parquet`.

**Step 2: Run test to verify failure**

```bash
.venv/bin/python -m pytest tests/test_live_script.py -v
```

**Step 3: Implement `fetch_live_metars.py`**

```python
"""Fetch live METARs for the DFW station network and print a summary."""
import argparse
from pathlib import Path
import sys

from dfw_temp_model.config import CACHE_DIR, STATIONS, TARGET_ICAO
from dfw_temp_model.data.aviationweather import fetch_aviationweather


def main():
    parser = argparse.ArgumentParser(description="Fetch live METARs")
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--cache", type=str, default=str(Path(CACHE_DIR) / "live_metars.parquet"))
    args = parser.parse_args()

    df = fetch_aviationweather(STATIONS, hours=args.hours, cache_path=args.cache)
    if df.empty:
        print("No live METARs returned.", file=sys.stderr)
        sys.exit(1)

    latest = df.sort_values("valid").groupby("station").last().reset_index()
    print(latest[["station", "valid", "tmpf", "drct", "sknt", "skyc1"]].to_string(index=False))
    print(f"\nCached to: {args.cache}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify pass**

```bash
.venv/bin/python -m pytest tests/test_live_script.py -v
.venv/bin/python scripts/fetch_live_metars.py
```

Expected: prints a table of 8 stations with current temps/wind/clouds.

**Step 5: Commit**

```bash
git add scripts/fetch_live_metars.py tests/test_live_script.py
git commit -m "feat: add live METAR polling script"
```

---

### Task 3: Add `AviationWeather` as a data-source option in `run_first_experiment.py`

**Objective:** Allow the historical experiment script to optionally use AviationWeather for recent data instead of IEM ASOS for the last ~24 hours.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/scripts/run_first_experiment.py`

**Step 1: Write failing test**

Add a test in `tests/test_first_experiment.py`:
- `test_load_or_fetch_aviationweather_option`: run the script with `--obs-source aviationweather --start-date 2024-06-01 --end-date 2024-06-02`, assert it completes and produces a metrics JSON.

**Step 2: Run test to verify failure**

```bash
.venv/bin/python -m pytest tests/test_first_experiment.py::test_load_or_fetch_aviationweather_option -v
```

**Step 3: Implement the `--obs-source` flag**

- Add argument `--obs-source {iem,aviationweather}` defaulting to `iem`.
- In `load_or_fetch_asos`, if source is `aviationweather`, call `fetch_aviationweather` with `hours` derived from date range (up to 24 hours because AviationWeather only supports recent data).
- Fallback to IEM if date range is older than ~24 hours or AviationWeather returns empty.

**Step 4: Run test to verify pass**

```bash
.venv/bin/python -m pytest tests/test_first_experiment.py -v
```

**Step 5: Commit**

```bash
git add scripts/run_first_experiment.py tests/test_first_experiment.py
git commit -m "feat: allow AviationWeather as recent obs source in experiment"
```

---

### Task 4: Documentation and smoke test

**Objective:** Verify the live fetcher works with real data and document how a non-technical user runs it.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/docs/live_data.md`
- Modify: `/opt/data/stock-research/dfw_temp_model/README.md` (if it exists; create if not)

**Step 1: Smoke test**

```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python scripts/fetch_live_metars.py
```
Expected: live table of 8 stations printed from the network.

**Step 2: Update README / add live_data.md**

```markdown
# Live data

Fetch current METARs for the DFW network:

    .venv/bin/python scripts/fetch_live_metars.py

This uses AviationWeather.gov's free JSON API (no key required) and caches the result to `data/cache/live_metars.parquet`.
```

**Step 3: Full test suite**

```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: all tests still pass (count will be 70 + new tests).

**Step 4: Commit**

```bash
git add docs/live_data.md README.md
git commit -m "docs: add live METAR fetcher usage"
```

---

## Risks, tradeoffs, and open questions

1. **AviationWeather is hourly + SPECIs, not true 5-minute.** It will not capture sub-hourly fronts as fast as Synoptic, but it is free and official.
2. **The JSON endpoint may change field names.** The parser should tolerate missing keys and use `.get()` / `pd.json_normalize` safely.
3. **Live data is only recent.** The `/api/data/metar` endpoint supports `hours=N`; older data should still come from IEM.
4. **Station coords are not returned.** We must merge lat/lon from `config.STATIONS` by `icaoId` so downstream geometry code works.
5. **Temperature in JSON is °C.** Must convert to °F to match the rest of the pipeline.
6. **No API key needed**, but rate-limit unknown — implement polite delay and retry.
7. **Should the daily high target still come from IEM `daily.py`?** Yes. AviationWeather is only for live feature observations; the official daily high target remains IEM daily or NWS climate report.

---

## Validation checklist

- [ ] `tests/test_aviationweather.py` passes.
- [ ] `tests/test_live_script.py` passes.
- [ ] `scripts/fetch_live_metars.py` prints live data from the network.
- [ ] Full suite still passes.
- [ ] No new non-optional dependencies added.

---

## Definition of done

A user can ask Hermes: *“Fetch live METARs for DFW”* and get a current temperature/wind table back within seconds, sourced directly from AviationWeather.gov with no API key.
