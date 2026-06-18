# Test Fix + DB Viewer Dashboard Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Fix the one failing pre-existing test, then add a read-only static HTML database viewer page to the GitHub Pages dashboard so the SQLite weather database is browseable from a browser without any third-party tools.

**Architecture:** Two independent workstreams. Workstream A is a 1-line test fix. Workstream B generates a self-contained static HTML page (snapshot of the SQLite DB) alongside the existing dashboard, pushed to GitHub Pages by the existing cron job. No new dependencies, no server, no database viewer frameworks — just a Python script that reads the SQLite DB and writes a static HTML file with sortable tables and charts. The existing cron script gets one line added to call the new generator.

**Tech Stack:** Python 3.13, pandas, plotly (already installed), SQLite (stdlib), GitHub Pages static HTML.

---

## Current Context

### The failing test
`tests/test_build_dataset.py::test_end_to_end_smoke` (line 128) asserts:
```python
for col in NEIGHBOR_ICAOS:
    assert col in target.columns
```
`NEIGHBOR_ICAOS` is `[s.icao for s in STATIONS if s.icao != TARGET_ICAO]` where `TARGET_ICAO = "KDAL"`.
So it expects `"KDFW"` to be in `target.columns`.

But `build_target_table()` in `dfw_temp_model/data/build_dataset.py:31-38` creates:
- `target["kdfw_obs"]` and `target["kdfw_fcst"]` (lowercase, not `KDFW`)
- `target["residual_target"]`
- neighbor columns from `residuals.columns` (which are the neighbor ICAOs, NOT KDFW since `build_residual_table` skips `KDFW` at line 24)

So `KDFW` is intentionally absent — it was split into `kdfw_obs`/`kdfw_fcst` when KDFW became the observation source (not a neighbor). The test is stale. It needs to check for the neighbor ICAOs minus KDFW, or check for `kdfw_obs`/`kdfw_fcst` separately.

### The database
SQLite at `data/cache/db/weather_observations.db`:
- `metar_observations`: 312 rows, columns: id, fetched_at, source, station, valid, lat, lon, tmpf, dewpf, drct, sknt, skyc1, mslp, p01i
- `hrrr_forecasts`: 1384 rows, columns: id, fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf

### The existing dashboard
- `scripts/generate_dashboard.py` generates `dfw-live-dashboard/index.html` — a dark-themed weather dashboard with Plotly charts.
- The GitHub Pages repo root `index.html` is a "dashboard of dashboards" that auto-discovers `.html` files.
- The cron job at `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` runs hourly: ingest → generate dashboard → git commit → git push.
- The existing dashboard is public-facing. The DB viewer should be a separate page to keep the weather dashboard clean.

### Design decision: separate page vs. embedded in weather dashboard
The user explicitly said the weather dashboard is public-facing and to keep changes to a minimum. So the DB viewer will be a **separate static HTML page** at `dfw-data-viewer/index.html` in the GitHub Pages repo. The root index.html auto-discovers HTML files in subdirectories via the GitHub API, so it will show up automatically in the dashboard-of-dashboards listing.

### Design decision: snapshot vs. live
Since this is GitHub Pages (static hosting), the HTML file is a **snapshot** of the DB at generation time. The cron job regenerates it hourly, so it stays fresh. No server, no API calls needed — all data is embedded in the HTML.

---

## Workstream A: Fix the failing test

### Task 1: Fix test_end_to_end_smoke column assertions

**Objective:** Update the stale test to match the current build_dataset column naming.

**Files:**
- Modify: `tests/test_build_dataset.py:127-128`

**Step 1: Read the current test to confirm exact lines**

The failing assertion is at lines 127-128:
```python
    for col in NEIGHBOR_ICAOS:
        assert col in target.columns
```

`NEIGHBOR_ICAOS` includes `"KDFW"` (line 10) because it filters out `TARGET_ICAO` which is `"KDAL"`, not `"KDFW"`.

**Step 2: Fix the assertion**

Replace lines 126-128 with:
```python
    # KDFW is the observation source, not a neighbor — it appears as
    # kdfw_obs / kdfw_fcst, not as a standalone KDFW column.
    assert "kdfw_obs" in target.columns
    assert "kdfw_fcst" in target.columns
    # All neighbor columns exist (neighbors = all stations except KDFW and KDAL).
    neighbor_icaos = [s.icao for s in STATIONS if s.icao not in ("KDFW", "KDAL")]
    for col in neighbor_icaos:
        assert col in target.columns
```

**Step 3: Run the test (may be skipped if network marks prevent it)**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_build_dataset.py::test_end_to_end_smoke -v -m "not network"`

Note: This test is marked `@pytest.mark.network` and `@pytest.mark.slow`, so it may be skipped by default. If skipped, verify the assertion logic by running a focused unit test instead:

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "
from dfw_temp_model.config import STATIONS, TARGET_ICAO
neighbor_icaos = [s.icao for s in STATIONS if s.icao not in ('KDFW', 'KDAL')]
print('Neighbors:', neighbor_icaos)
assert 'KDFW' not in neighbor_icaos
assert 'KDAL' not in neighbor_icaos
print('OK')
"`

Expected: `OK`

**Step 4: Run full non-network test suite**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow" --tb=short`

Expected: All passed, 0 failed.

**Step 5: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add tests/test_build_dataset.py
git commit -m "fix: update test_end_to_end_smoke for KDFW-as-observation-source column naming"
```

---

## Workstream B: DB Viewer Dashboard

### Task 2: Create the DB viewer generator script

**Objective:** Write a Python script that reads the SQLite DB and produces a self-contained static HTML page with sortable tables and summary charts.

**Files:**
- Create: `scripts/generate_db_viewer.py`

**Step 1: Write the script**

The script should:
- Accept `--db` and `--output-dir` arguments (same pattern as `generate_dashboard.py`)
- Read `metar_observations` and `hrrr_forecasts` tables from SQLite
- Produce a single `index.html` with:
  - Summary cards (row counts, date ranges, station counts)
  - METAR observations table (all rows, sortable via simple HTML `<table>` with a small JS sort function)
  - HRRR forecasts table (all rows, sortable)
  - A Plotly chart of temperature trends per station (METAR)
  - A Plotly chart of HRRR forecast curve for the target station (latest cycle)
  - Dark theme matching the existing dashboard (`#0f172a` background)
  - All CSS/JS inline or via CDN (Plotly via CDN, same as existing dashboard)
  - Central Time conversion on all timestamps (reuse the `zoneinfo` approach from `generate_dashboard.py`)

Key implementation details:
- Use `pandas.read_sql_query` to load tables
- Use `plotly.offline.plot` with `include_plotlyjs='cdn'` for charts (same as existing dashboard)
- Use a lightweight vanilla-JS table sorter (~20 lines) — no external library
- Round floats to 2 decimal places for display
- Limit table display to last 500 rows (most recent) with a note if truncated, to keep HTML size reasonable. Full data available in the SQLite DB itself.
- Timestamps displayed in both UTC and CT (12-hour) like the weather dashboard

```python
#!/usr/bin/env python3
"""Generate a static HTML database viewer page from the SQLite weather DB."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

# Ensure project is importable when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dfw_temp_model.config import CACHE_DIR, TARGET_ICAO

_CT = ZoneInfo("America/Chicago")


def _dt_to_ct(dt, fmt="%m/%d %I:%M %p CT"):
    """Convert a timezone-aware datetime to Central Time string."""
    ct = dt.tz_convert(_CT)
    return ct.strftime(fmt)


def _table_html(df: pd.DataFrame, max_rows: int = 500) -> str:
    """Convert a DataFrame to an HTML table with inline CSS and JS sorting."""
    display = df.head(max_rows).copy()
    truncated = len(df) > max_rows

    # Round floats
    for col in display.select_dtypes(include=["float"]).columns:
        display[col] = display[col].round(2)

    html = display.to_html(index=False, classes="data-table", border=0, escape=False)
    if truncated:
        note = f'<p class="note">Showing latest {max_rows} of {len(df)} rows</p>'
    else:
        note = ""
    return note + html


def _metar_trend_chart(df: pd.DataFrame) -> str:
    """Plotly chart: temperature trend per station from METAR observations."""
    df = df.copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    stations = df["station"].unique()

    fig = go.Figure()
    for st in sorted(stations):
        sub = df[df["station"] == st].sort_values("valid")
        fig.add_trace(go.Scatter(
            x=sub["valid"],
            y=sub["tmpf"],
            mode="lines+markers",
            name=st,
            hovertemplate=f"<b>{st}</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>Temp: %{{y:.1f}}°F<extra></extra>",
        ))

    fig.update_layout(
        title="METAR Temperature Trend — All Stations",
        xaxis_title="Time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        hovermode="x unified",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs="cdn", config={"displayModeBar": False})


def _hrrr_forecast_chart(df: pd.DataFrame) -> str:
    """Plotly chart: latest HRRR cycle forecast for target station."""
    target_df = df[df["station"] == TARGET_ICAO].copy()
    if target_df.empty:
        return "<p>No HRRR data for target station</p>"

    target_df["valid_dt"] = pd.to_datetime(target_df["valid_dt"], utc=True)
    target_df["init_dt"] = pd.to_datetime(target_df["init_dt"], utc=True)

    # Pick the latest init cycle with the most forecast hours
    latest_init = target_df["init_dt"].max()
    cycle_df = target_df[target_df["init_dt"] == latest_init].sort_values("forecast_hour")

    if cycle_df.empty:
        return "<p>No complete HRRR cycle available</p>"

    fig = go.Figure(data=[go.Scatter(
        x=cycle_df["valid_dt"],
        y=cycle_df["tmpf"],
        mode="lines+markers",
        name=f"HRRR {TARGET_ICAO}",
        line={"color": "#f59e0b", "width": 2},
        marker={"size": 6, "color": "#f59e0b"},
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.15)",
        hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M UTC}}</b><br>Temp: %{{y:.1f}}°F<br>f%{{text}}<extra></extra>",
        text=cycle_df["forecast_hour"].astype(int),
    )])

    init_label = latest_init.strftime("%Y-%m-%d %H:%M UTC")
    ct_label = _dt_to_ct(latest_init)

    fig.update_layout(
        title=f"HRRR Forecast — {TARGET_ICAO} 2m temp<br><sup>Latest cycle {init_label} · {ct_label}</sup>",
        xaxis_title="Valid time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        showlegend=False,
        hovermode="x unified",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False})


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DFW Weather Database Viewer</title>
    <style>
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #38bdf8; margin-bottom: 0.25rem; }}
        h2 {{ color: #7dd3fc; margin-top: 2rem; }}
        p.updated {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0; max-width: 900px; }}
        .card {{ background: #1e293b; padding: 1rem; border-radius: 0.5rem; }}
        .card h3 {{ margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.03em; }}
        .card p {{ margin: 0; font-size: 1.4rem; font-weight: 600; color: #38bdf8; }}
        .card small {{ display: block; color: #64748b; margin-top: 0.25rem; font-size: 0.75rem; }}
        .note {{ color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0; }}
        table.data-table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; max-width: 1200px; font-size: 0.82rem; }}
        table.data-table th, table.data-table td {{ border: 1px solid #334155; padding: 0.35rem 0.6rem; text-align: left; cursor: pointer; }}
        table.data-table th {{ background: #1e293b; position: sticky; top: 0; }}
        table.data-table th:hover {{ background: #334155; }}
        table.data-table tr:nth-child(even) {{ background: #162032; }}
        table.data-table tr:hover {{ background: #1e3a5f; }}
        .table-wrap {{ max-height: 500px; overflow-y: auto; margin: 1rem 0; max-width: 1200px; }}
        .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.85rem; max-width: 900px; }}
        a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>DFW Weather Database Viewer</h1>
    <p class="updated">Generated {generated_at} UTC · {generated_at_ct} CT · Source: {db_path}<br>This is a read-only snapshot refreshed hourly by the cron job.</p>

    <div class="stats">
        <div class="card"><h3>METAR rows</h3><p>{metar_count}</p></div>
        <div class="card"><h3>HRRR rows</h3><p>{hrrr_count}</p></div>
        <div class="card"><h3>Stations</h3><p>{station_count}</p></div>
        <div class="card"><h3>Date range</h3><p>{date_range}</p></div>
    </div>

    <h2>METAR Temperature Trends</h2>
    {metar_chart}

    <h2>HRRR Forecast — Latest Cycle ({TARGET_ICAO})</h2>
    {hrrr_chart}

    <h2>METAR Observations</h2>
    <div class="table-wrap">
    {metar_table}
    </div>

    <h2>HRRR Forecasts</h2>
    <div class="table-wrap">
    {hrrr_table}
    </div>

    <div class="footer">
        <a href="../dfw-live-dashboard/">← Back to Weather Dashboard</a><br>
        <a href="https://github.com/DAlvarez101/DAlvarez101.HermesStocks.io">View repo →</a>
    </div>

    <script>
    // Lightweight table sorter — click a header to sort by that column.
    document.querySelectorAll('table.data-table th').forEach((th, colIdx) => {{
        th.addEventListener('click', () => {{
            const table = th.closest('table');
            const tbody = table.querySelector('tbody') || table;
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const numeric = !isNaN(parseFloat(rows[0]?.querySelector(`td:nth-child(${{colIdx + 1}})`)?.textContent));
            const asc = th.dataset.sort !== 'asc';
            th.dataset.sort = asc ? 'asc' : 'desc';
            rows.sort((a, b) => {{
                let av = a.querySelector(`td:nth-child(${{colIdx + 1}})`)?.textContent ?? '';
                let bv = b.querySelector(`td:nth-child(${{colIdx + 1}})`)?.textContent ?? '';
                if (numeric) {{
                    av = parseFloat(av) || 0;
                    bv = parseFloat(bv) || 0;
                }}
                return asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
            }});
            rows.forEach(r => tbody.appendChild(r));
        }});
    }});
    </script>
</body>
</html>
"""


def generate_db_viewer(db_path: str, output_dir: str) -> Path:
    import sqlite3

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    metar_df = pd.read_sql_query("SELECT * FROM metar_observations ORDER BY valid DESC", conn)
    hrrr_df = pd.read_sql_query("SELECT * FROM hrrr_forecasts ORDER BY valid_dt DESC", conn)
    conn.close()

    # Stats
    metar_count = len(metar_df)
    hrrr_count = len(hrrr_df)
    station_count = metar_df["station"].nunique() if not metar_df.empty else 0

    if not metar_df.empty:
        first_valid = pd.to_datetime(metar_df["valid"].min(), utc=True)
        last_valid = pd.to_datetime(metar_df["valid"].max(), utc=True)
        date_range = f"{first_valid.strftime('%m/%d')} – {last_valid.strftime('%m/%d')}"
    else:
        date_range = "—"

    # Charts
    metar_chart = _metar_trend_chart(metar_df) if not metar_df.empty else "<p>No METAR data</p>"
    hrrr_chart = _hrrr_forecast_chart(hrrr_df) if not hrrr_df.empty else "<p>No HRRR data</p>"

    # Tables (prepare display columns)
    metar_display = metar_df[["station", "valid", "tmpf", "dewpf", "drct", "sknt", "mslp", "p01i"]].copy()
    metar_display.columns = ["Station", "Valid (UTC)", "Temp °F", "Dewpt °F", "Dir °", "Speed kt", "MSLP mb", "Precip in"]

    hrrr_display = hrrr_df[["station", "init_dt", "forecast_hour", "valid_dt", "tmpf"]].copy()
    hrrr_display.columns = ["Station", "Init (UTC)", "Fhr", "Valid (UTC)", "Temp °F"]

    now_utc = datetime.now(timezone.utc)
    now_ct = now_utc.astimezone(_CT)

    html = HTML_TEMPLATE.format(
        generated_at=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        generated_at_ct=now_ct.strftime("%I:%M %p"),
        db_path=db_path,
        metar_count=metar_count,
        hrrr_count=hrrr_count,
        station_count=station_count,
        date_range=date_range,
        metar_chart=metar_chart,
        hrrr_chart=hrrr_chart,
        metar_table=_table_html(metar_display),
        hrrr_table=_table_html(hrrr_display),
        TARGET_ICAO=TARGET_ICAO,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate static DB viewer HTML")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer",
        help="Directory to write index.html",
    )
    args = parser.parse_args()

    path = generate_db_viewer(args.db, args.output_dir)
    print(f"DB viewer written to: {path}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the script runs**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/generate_db_viewer.py --db data/cache/db/weather_observations.db --output-dir /tmp/dfw-test-viewer 2>&1`

Expected: `DB viewer written to: /tmp/dfw-test-viewer/index.html`

**Step 3: Verify the HTML output**

Run: `ls -la /tmp/dfw-test-viewer/index.html && grep -c 'data-table' /tmp/dfw-test-viewer/index.html && grep -c 'plotly' /tmp/dfw-test-viewer/index.html`

Expected: File exists, contains `data-table` class and plotly references.

**Step 4: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add scripts/generate_db_viewer.py
git commit -m "feat: add static HTML database viewer generator script"
```

---

### Task 3: Add DB viewer generation to the cron script

**Objective:** Add one line to the existing cron script so the DB viewer is regenerated alongside the weather dashboard each hour. This is the ONLY change to the cron script — everything else stays untouched.

**Files:**
- Modify: `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` (add 3 lines after the dashboard generation line)
- Also sync: `/opt/data/scripts/dfw_live_metar_hourly.sh` (keep in sync)

**Step 1: Read the current cron script**

The script currently has these lines after METAR ingestion:
```bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating dashboard ..."
"$PYTHON" scripts/generate_dashboard.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/${DASHBOARD_SUBDIR}"

cd "$PAGES_DIR"
```

**Step 2: Add DB viewer generation after the dashboard generation**

Insert after the `generate_dashboard.py` line and before `cd "$PAGES_DIR"`:
```bash

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating DB viewer ..."
"$PYTHON" scripts/generate_db_viewer.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/dfw-data-viewer"
```

So the section becomes:
```bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating dashboard ..."
"$PYTHON" scripts/generate_dashboard.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/${DASHBOARD_SUBDIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating DB viewer ..."
"$PYTHON" scripts/generate_db_viewer.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/dfw-data-viewer"

cd "$PAGES_DIR"
```

The `dfw-data-viewer/` directory will be picked up by the git commit + push logic since the script already does `git add "${DASHBOARD_SUBDIR}/"` — but we need to also add `dfw-data-viewer/`. Update the git add line:

Change:
```bash
    git add "${DASHBOARD_SUBDIR}/"
```
To:
```bash
    git add "${DASHBOARD_SUBDIR}/" "dfw-data-viewer/"
```

And update the diff check to include the new directory:
```bash
if ! git diff --quiet -- "${DASHBOARD_SUBDIR}/" "dfw-data-viewer/" || ! git diff --cached --quiet -- "${DASHBOARD_SUBDIR}/" "dfw-data-viewer/"; then
```

**Step 3: Apply the changes to both copies of the cron script**

Apply to `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` (the one the scheduler uses).
Apply to `/opt/data/scripts/dfw_live_metar_hourly.sh` (the source copy).

**Step 4: Test the full cron script manually**

Run: `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh 2>&1 | tail -20`

Expected: Output includes "Generating DB viewer ..." and "DB viewer written to: ..." and the git push succeeds.

**Step 5: Verify the HTML file was created and pushed**

Run: `ls -la /opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer/index.html`

Expected: File exists, non-zero size.

Run: `cd /opt/data/DAlvarez101.HermesStocks.io && git log --oneline -3`

Expected: Latest commit includes `dfw-data-viewer/index.html`.

**Step 6: Commit the cron script changes to the repo**

```bash
cd /opt/data/stock-research/dfw_temp_model
# The cron scripts are outside the repo, but the repo version scripts/cron_update_dashboard.sh
# should also be updated for consistency if it exists.
git add scripts/cron_update_dashboard.sh  # if it was modified
git commit -m "feat: add DB viewer generation to cron script"
```

---

### Task 4: Verify the DB viewer page renders correctly

**Objective:** Confirm the generated HTML page is valid and displays correctly in a browser.

**Step 1: Check HTML structure**

Run: `grep -c '<table' /opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer/index.html`

Expected: At least 2 (METAR + HRRR tables).

Run: `grep -c 'Plotly.newPlot' /opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer/index.html`

Expected: At least 2 (METAR trend + HRRR forecast charts).

Run: `grep 'CT' /opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer/index.html | head -5`

Expected: CT timestamps present.

**Step 2: Open the page in the browser tool to visually verify**

Use `browser_navigate` to open `https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-data-viewer/` and verify:
- Dark theme renders
- Summary cards show row counts
- Tables are visible and sortable
- Charts render (Plotly)
- Back link to weather dashboard works
- CT timestamps are present

**Step 3: Verify it appears on the root dashboard**

Check `https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/` — the root index.html auto-discovers HTML files. The `dfw-data-viewer/` subdirectory should appear as a link card.

---

### Task 5: Run full test suite and confirm no regressions

**Objective:** Verify everything passes.

**Step 1: Run non-network tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow" --tb=short`

Expected: All passed, 0 failed.

**Step 2: Run the trading-specific tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_trading_*.py -v`

Expected: All passed.

**Step 3: Run a dry-run of the trading bot**

Run: `cd /opt/data/stock-research/dfw_temp_model && POLYMARKET_PRIVATE_KEY=0x0000000000000000000000000000000000000000000000000000000000000001 .venv/bin/python scripts/run_polymarket_bot.py --dry-run 2>&1 | head -5`

Expected: JSON output starting with `{`.

---

## Files Changed Summary

| File | Action | Description |
|------|--------|-------------|
| `tests/test_build_dataset.py` | Modify | Fix stale column assertion (Task 1) |
| `scripts/generate_db_viewer.py` | Create | New script: SQLite → static HTML viewer (Task 2) |
| `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` | Modify | Add DB viewer generation to cron (Task 3) |
| `/opt/data/scripts/dfw_live_metar_hourly.sh` | Modify | Keep source copy in sync (Task 3) |
| `/opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer/index.html` | Auto-generated | Output of the viewer script |

## What is NOT touched

- `scripts/generate_dashboard.py` — the existing weather dashboard is not modified
- `dfw_temp_model/trading/*` — no trading code changes
- `dfw_temp_model/data/*` — no data pipeline changes
- `dfw_temp_model/storage/*` — no storage changes
- The cron job schedule, timing, or core logic — only adds one more generation step

## Risks and Mitigations

1. **HTML file size** — 312 METAR rows + 1384 HRRR rows could produce a large HTML file. Mitigated by capping table display at 500 rows. The full data stays in SQLite.

2. **GitHub Pages repo growth** — Each hourly push regenerates the HTML. Since it's a single file replaced each time (not appended), git history grows by one diff per hour. This is the same pattern as the existing dashboard. No additional risk.

3. **Cron script modification** — The user explicitly asked to not touch working code, especially the cron job. The change is minimal (3 lines added, 2 lines modified for git add) and additive. The dashboard generation line is untouched. If the DB viewer script fails, `set -euo pipefail` will abort the cron job — but this is the same behavior as if the dashboard generation failed. To be safe, we could wrap the DB viewer call in a `|| true` to prevent it from blocking the dashboard push if it fails. Actually, let's do that:

   ```bash
   echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating DB viewer ..."
   "$PYTHON" scripts/generate_db_viewer.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/dfw-data-viewer" || echo "WARNING: DB viewer generation failed, continuing ..."
   ```

   This way, if the DB viewer script has a bug, the dashboard still gets pushed. This is the safest approach.

4. **Privacy** — The DB viewer is public on GitHub Pages. It contains weather data only (temperatures, wind, pressure). No personal data, no trading data, no API keys. This is the same data already shown on the existing weather dashboard, just in a more detailed table format.

## Open Questions

- Should the DB viewer also show the `fetched_at` column (ingestion timestamp)? It's currently excluded from display for brevity but could be useful for debugging. Left out for now — can add later.
- Should there be a download link for the raw SQLite DB? This would expose the full database publicly. Not included for now — the user can decide if they want this later.