# Append-Only SQLite Observation Archive + Hourly Cron + GitHub Pages Dashboard

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. Combine TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Build a self-contained, append-only SQLite database inside the DFW temperature model project that records live METAR observations every hour; add a simple static HTML dashboard pushed to the GitHub Pages repo so progress is visible without exposing ports or running a separate server.

**Architecture:** A Python module `dfw_temp_model.storage.obs_db` manages an SQLite database at `data/db/weather_observations.db`. A CLI script `scripts/ingest_live_metars.py` fetches the latest AviationWeather.gov METARs and inserts new rows with `INSERT OR IGNORE` so repeated runs never duplicate data. A second script `scripts/generate_dashboard.py` reads the database and writes a static HTML/PNG report to the GitHub Pages repo. A Hermes `cronjob` runs the ingest script hourly and the dashboard generator once per day. Everything lives inside the existing project directory in the Hostinger Docker VPS and persists on the mounted `/opt/data` volume.

**Tech Stack:** Python 3.13, SQLite (stdlib), pandas, requests (already installed), matplotlib for charts. No separate database server.

---

## Current context

- Project root: `/opt/data/stock-research/dfw_temp_model`
- Virtual environment: `.venv/bin/python`
- Live fetcher: `dfw_temp_model/data/aviationweather.py` returns a DataFrame with columns `station`, `valid`, `lat`, `lon`, `tmpf`, `dewpf`, `drct`, `sknt`, `skyc1`, `mslp`, `p01i`.
- GitHub Pages repo: `/opt/data/DAlvarez101.HermesStocks.io/` (already configured for pushes).
- The container runs on Hostinger; `/opt/data` is persistent.
- The user wants the database to grow forever and never be deleted from.

---

## Step-by-step plan

### Task 1: Add SQLite observation database module

**Objective:** Create a reusable module that opens/creates an append-only SQLite database with a `metar_observations` table.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/storage/__init__.py`
- Create: `/opt/data/stock-research/dfw_temp_model/dfw_temp_model/storage/obs_db.py`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_obs_db.py`

**Step 1: Write failing tests**

In `tests/test_obs_db.py`:
- `test_get_db_creates_file`: call `get_db(path)` on a temp path, assert the file exists and returns a sqlite3.Connection.
- `test_ensure_schema_creates_table`: call `ensure_schema(conn)`, query `sqlite_master`, assert `metar_observations` exists.
- `test_insert_observations`: insert two sample rows, assert table has 2 rows.
- `test_insert_or_ignore_prevents_duplicates`: insert the same unique `(source, station, valid)` twice, assert table still has 2 rows.
- `test_latest_observations_per_station`: insert rows for KDFW and KDAL at different times, assert `latest_by_station` returns the newest row per station.

**Step 2: Run tests to verify failure**

```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python -m pytest tests/test_obs_db.py -v
```
Expected: `ModuleNotFoundError` or assertion failures.

**Step 3: Implement `obs_db.py`**

```python
"""Append-only SQLite storage for METAR observations."""
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metar_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL,
    station TEXT NOT NULL,
    valid TEXT NOT NULL,
    lat REAL,
    lon REAL,
    tmpf REAL,
    dewpf REAL,
    drct REAL,
    sknt REAL,
    skyc1 TEXT,
    mslp REAL,
    p01i REAL,
    UNIQUE(source, station, valid)
);

CREATE INDEX IF NOT EXISTS idx_metar_station_valid
    ON metar_observations(station, valid);

CREATE INDEX IF NOT EXISTS idx_metar_fetched_at
    ON metar_observations(fetched_at);
"""


def get_db(db_path: str) -> sqlite3.Connection:
    """Open or create the SQLite database and ensure schema exists."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the observation table and indexes if they do not exist."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_observations(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source: str,
    fetched_at: Optional[str] = None,
) -> int:
    """Insert rows from a DataFrame using INSERT OR IGNORE.

    Returns the number of newly inserted rows.
    """
    if df.empty:
        return 0

    if fetched_at is None:
        from datetime import datetime, timezone
        fetched_at = datetime.now(timezone.utc).isoformat()

    columns = [
        "fetched_at", "source", "station", "valid", "lat", "lon",
        "tmpf", "dewpf", "drct", "sknt", "skyc1", "mslp", "p01i",
    ]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            fetched_at,
            source,
            row.get("station"),
            row.get("valid"),
            row.get("lat"),
            row.get("lon"),
            row.get("tmpf"),
            row.get("dewpf"),
            row.get("drct"),
            row.get("sknt"),
            row.get("skyc1"),
            row.get("mslp"),
            row.get("p01i"),
        ))

    cursor = conn.cursor()
    cursor.executemany(
        f"""
        INSERT OR IGNORE INTO metar_observations (
            {', '.join(columns)}
        ) VALUES ({', '.join('?' for _ in columns)})
        """,
        rows,
    )
    conn.commit()
    return cursor.rowcount


def read_all(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the entire table as a DataFrame."""
    return pd.read_sql_query(
        "SELECT * FROM metar_observations ORDER BY valid", conn
    )


def latest_by_station(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the most recent row per station."""
    return pd.read_sql_query(
        """
        SELECT m.*
        FROM metar_observations m
        INNER JOIN (
            SELECT station, MAX(valid) AS max_valid
            FROM metar_observations
            GROUP BY station
        ) t ON m.station = t.station AND m.valid = t.max_valid
        ORDER BY m.station
        """,
        conn,
    )


def row_count(conn: sqlite3.Connection) -> int:
    """Return total row count."""
    return conn.execute("SELECT COUNT(*) FROM metar_observations").fetchone()[0]
```

**Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_obs_db.py -v
```
Expected: 5 passed.

**Step 5: Commit**

```bash
git add dfw_temp_model/storage/__init__.py dfw_temp_model/storage/obs_db.py tests/test_obs_db.py
git commit -m "feat: add append-only SQLite observation database"
```

---

### Task 2: Add hourly ingestion script

**Objective:** Build a CLI that fetches live METARs and appends them to the database.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/scripts/ingest_live_metars.py`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_ingest_script.py`

**Step 1: Write failing tests**

In `tests/test_ingest_script.py`:
- `test_ingest_script_smoke`: run the script with a temp database path, assert it reports non-negative inserted rows and the database file exists.
- `test_ingest_script_idempotent`: run the script twice with the same temp database, assert the second run inserts 0 or very few new rows (because of `INSERT OR IGNORE`).

**Step 2: Run tests to verify failure**

```bash
.venv/bin/python -m pytest tests/test_ingest_script.py -v
```
Expected: failures.

**Step 3: Implement `ingest_live_metars.py`**

```python
"""Hourly ingestion script: fetch live METARs and append to SQLite."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dfw_temp_model.config import CACHE_DIR, STATIONS
from dfw_temp_model.data.aviationweather import fetch_aviationweather
from dfw_temp_model.storage.obs_db import get_db, insert_observations


def main():
    parser = argparse.ArgumentParser(description="Ingest live METARs into SQLite")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=2,
        help="Hours back to fetch",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="aviationweather",
        help="Source label to store in the database",
    )
    args = parser.parse_args()

    fetched_at = datetime.now(timezone.utc).isoformat()
    print(f"[{fetched_at}] Fetching live METARs ({args.hours}h back) ...")

    df = fetch_aviationweather(STATIONS, hours=args.hours)
    if df.empty:
        print("No METARs returned; nothing to ingest.", file=sys.stderr)
        sys.exit(1)

    conn = get_db(args.db)
    inserted = insert_observations(conn, df, source=args.source, fetched_at=fetched_at)
    total = conn.execute("SELECT COUNT(*) FROM metar_observations").fetchone()[0]
    conn.close()

    print(f"Inserted {inserted} new rows. Total rows in database: {total}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/test_ingest_script.py -v
.venv/bin/python scripts/ingest_live_metars.py --db /tmp/test_ingest.db
```

**Step 5: Commit**

```bash
git add scripts/ingest_live_metars.py tests/test_ingest_script.py
git commit -m "feat: add hourly METAR ingestion script"
```

---

### Task 3: Add dashboard report generator

**Objective:** Read the database and generate a static HTML dashboard with simple charts and summary tables, then copy it to the GitHub Pages repo.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/scripts/generate_dashboard.py`
- Test: `/opt/data/stock-research/dfw_temp_model/tests/test_generate_dashboard.py`
- Add dependency: `matplotlib` to `pyproject.toml`

**Step 1: Write failing tests**

In `tests/test_generate_dashboard.py`:
- `test_generate_dashboard_creates_html`: run the generator with a populated temp DB and temp output dir, assert an `index.html` is created.
- `test_summary_stats`: verify `summary_stats(conn)` returns total rows, station count, date range.

**Step 2: Run tests to verify failure**

```bash
.venv/bin/python -m pytest tests/test_generate_dashboard.py -v
```

**Step 3: Implement dashboard generator**

```python
"""Generate a static HTML dashboard from the observation database."""
import argparse
import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dfw_temp_model.config import CACHE_DIR, TARGET_ICAO
from dfw_temp_model.storage.obs_db import get_db, latest_by_station

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DFW Live Weather Dashboard</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #38bdf8; }}
        table {{ border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ border: 1px solid #334155; padding: 0.5rem 1rem; text-align: left; }}
        th {{ background: #1e293b; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }}
        .card {{ background: #1e293b; padding: 1rem; border-radius: 0.5rem; }}
        .card h3 {{ margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #94a3b8; }}
        .card p {{ margin: 0; font-size: 1.5rem; font-weight: 600; color: #38bdf8; }}
        img {{ max-width: 100%; border-radius: 0.5rem; margin: 1rem 0; }}
        .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <h1>DFW Live Weather Dashboard</h1>
    <p>Updated at {updated_at} UTC</p>
    <div class="stats">
        <div class="card"><h3>Total observations</h3><p>{total_rows}</p></div>
        <div class="card"><h3>Stations</h3><p>{station_count}</p></div>
        <div class="card"><h3>First observation</h3><p>{first_obs}</p></div>
        <div class="card"><h3>Latest observation</h3><p>{last_obs}</p></div>
    </div>
    <h2>Latest readings per station</h2>
    {latest_table}
    <h2>Temperature trend (target: KDFW)</h2>
    <img src="data:image/png;base64,{kdfw_chart}" alt="KDFW temperature trend">
    <h2>Hourly row count</h2>
    <img src="data:image/png;base64,{hourly_chart}" alt="Hourly ingestion volume">
    <div class="footer">
        Source: AviationWeather.gov METAR JSON. Database path: {db_path}<br>
        This page is generated automatically and pushed to GitHub Pages.
    </div>
</body>
</html>
"""


def summary_stats(conn) -> dict:
    df = pd.read_sql_query("SELECT * FROM metar_observations", conn)
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return {
        "total_rows": len(df),
        "station_count": df["station"].nunique(),
        "first_obs": df["valid"].min().strftime("%Y-%m-%d %H:%M UTC") if not df.empty else "—",
        "last_obs": df["valid"].max().strftime("%Y-%m-%d %H:%M UTC") if not df.empty else "—",
    }


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def kdfw_temperature_chart(conn) -> str:
    df = pd.read_sql_query(
        f"SELECT valid, tmpf FROM metar_observations WHERE station = '{TARGET_ICAO}' ORDER BY valid",
        conn,
    )
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No KDFW data yet", ha="center", va="center")
        return _to_base64(fig)
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["valid"], df["tmpf"], color="#38bdf8", linewidth=1.5)
    ax.set_title(f"{TARGET_ICAO} Temperature")
    ax.set_ylabel("Temperature (°F)")
    ax.set_xlabel("UTC")
    ax.grid(True, alpha=0.3)
    return _to_base64(fig)


def hourly_count_chart(conn) -> str:
    df = pd.read_sql_query(
        "SELECT substr(valid, 1, 13) AS hour FROM metar_observations",
        conn,
    )
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        return _to_base64(fig)
    counts = df.groupby("hour").size().reset_index(name="rows")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts["hour"], counts["rows"], color="#38bdf8")
    ax.set_title("Observations ingested per hour")
    ax.set_ylabel("Row count")
    ax.set_xlabel("Hour (UTC)")
    ax.tick_params(axis="x", rotation=45)
    return _to_base64(fig)


def generate_dashboard(db_path: str, output_dir: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_db(db_path)
    stats = summary_stats(conn)
    latest = latest_by_station(conn)
    conn.close()

    html = HTML_TEMPLATE.format(
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        total_rows=stats["total_rows"],
        station_count=stats["station_count"],
        first_obs=stats["first_obs"],
        last_obs=stats["last_obs"],
        latest_table=latest.to_html(index=False, classes="table"),
        kdfw_chart=kdfw_temperature_chart(get_db(db_path)),
        hourly_chart=hourly_count_chart(get_db(db_path)),
        db_path=db_path,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate static weather dashboard")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/data/DAlvarez101.HermesStocks.io/dfw-live-dashboard",
        help="Directory to write index.html",
    )
    args = parser.parse_args()

    path = generate_dashboard(args.db, args.output_dir)
    print(f"Dashboard written to: {path}")


if __name__ == "__main__":
    main()
```

**Step 4: Add matplotlib dependency**

In `pyproject.toml`, append `"matplotlib>=3.8"` to `dependencies`.

**Step 5: Run tests to verify pass**

```bash
uv pip install -e . --python .venv/bin/python
.venv/bin/python -m pytest tests/test_generate_dashboard.py -v
.venv/bin/python scripts/generate_dashboard.py --db /tmp/test_dash.db --output-dir /tmp/test_dash
```

**Step 6: Commit**

```bash
git add pyproject.toml scripts/generate_dashboard.py tests/test_generate_dashboard.py
git commit -m "feat: add static HTML dashboard generator"
```

---

### Task 4: Wire ingestion + dashboard into a single command

**Objective:** Create a convenience script that ingests and regenerates the dashboard in one call.

**Files:**
- Create: `/opt/data/stock-research/dfw_temp_model/scripts/ingest_and_update_dashboard.py`

**Step 1: Implement**

```python
"""Run hourly ingestion and regenerate the dashboard."""
import argparse
import subprocess
import sys
from pathlib import Path

from dfw_temp_model.config import CACHE_DIR


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Ingest METARs and update dashboard")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/data/DAlvarez101.HermesStocks.io/dfw-live-dashboard",
    )
    parser.add_argument("--hours", type=int, default=2)
    args = parser.parse_args()

    python = sys.executable
    rc = run([
        python, "scripts/ingest_live_metars.py",
        "--db", args.db,
        "--hours", str(args.hours),
    ])
    if rc != 0:
        sys.exit(rc)

    rc = run([
        python, "scripts/generate_dashboard.py",
        "--db", args.db,
        "--output-dir", args.output_dir,
    ])
    sys.exit(rc)


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test**

```bash
.venv/bin/python scripts/ingest_and_update_dashboard.py --db /tmp/smoke.db --output-dir /tmp/smoke_dash
ls /tmp/smoke_dash
```

**Step 3: Commit**

```bash
git add scripts/ingest_and_update_dashboard.py
git commit -m "feat: add combined ingest + dashboard update script"
```

---

### Task 5: Create Hermes cronjobs

**Objective:** Schedule automatic hourly ingestion and daily dashboard push to GitHub Pages.

**Step 1: Hourly ingestion cronjob**

Create with Hermes cronjob tool:
- Schedule: `0 * * * *` (top of every hour)
- Command: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/ingest_live_metars.py`
- No agent; pure shell (`no_agent=True`).
- Deliver to origin chat.

**Step 2: Daily dashboard + push cronjob**

- Schedule: `0 6 * * *` (6 AM UTC daily)
- Command:
  ```bash
  export HERMES_HOME=/opt/data && export HERMES_DOCKER_EXEC_AS_ROOT=1 && \
  cd /opt/data/stock-research/dfw_temp_model && \
  .venv/bin/python scripts/ingest_and_update_dashboard.py && \
  cd /opt/data/DAlvarez101.HermesStocks.io && \
  git add dfw-live-dashboard/ && \
  git commit -m "auto: update dfw live dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)" && \
  git push origin main
  ```
- `no_agent=True`.
- Deliver origin only on non-empty stdout (or use notify_on_complete).

**Step 3: Verify cronjobs are listed**

```bash
hermes cronjob list
```
Expected: two new jobs appear.

**Step 4: Commit any supporting wrapper scripts**

If a wrapper shell script is created to simplify the cron command, commit it under `scripts/cron_update_dashboard.sh`.

---

### Task 6: Initial manual run and GitHub Pages push

**Objective:** Populate the database once and publish the first dashboard.

**Step 1: Initial ingestion**

```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python scripts/ingest_live_metars.py --hours 2
```

**Step 2: Generate dashboard and push**

```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python scripts/ingest_and_update_dashboard.py
cd /opt/data/DAlvarez101.HermesStocks.io
git add dfw-live-dashboard/
git commit -m "feat: add initial DFW live weather dashboard"
git push origin main
```

**Step 3: Verify URL**

After push, the dashboard should be available at:
```
https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/
```

---

## Tests / validation summary

- `tests/test_obs_db.py` — schema creation, inserts, idempotency, latest-by-station.
- `tests/test_ingest_script.py` — script runs and is idempotent.
- `tests/test_generate_dashboard.py` — HTML report is generated from real rows.
- `tests/test_aviationweather.py` — live fetcher already exists; re-run full suite.

Final full-suite run:
```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: all tests pass.

---

## Risks, tradeoffs, and open questions

1. **SQLite on a shared volume.** SQLite handles concurrent reads well but not concurrent writes. The cron runs once per hour, so write contention is unlikely. If future scaling is needed, migrate to PostgreSQL later.
2. **AviationWeather rate limits unknown.** A 2-hour backfill every hour is polite; if rate-limit errors appear, increase the per-request delay.
3. **Database will grow forever.** With 8 stations and ~26 reports per 2-hour window, expect ~300 rows/day, ~9K/month, ~110K/year. SQLite will handle this for years without pruning.
4. **Dashboard page is rebuilt daily, but data is ingested hourly.** If the user wants near-real-time dashboard updates, change the dashboard cron to hourly too.
5. **Git auto-push requires the repo to be in a clean state.** The cron command commits the `dfw-live-dashboard/` directory. If other uncommitted changes exist in the Pages repo, the commit may include them. Consider using a dedicated GitHub Pages publishing branch or a separate repo if this becomes messy.
6. **No authentication on the dashboard URL.** The GitHub Pages dashboard is public. No sensitive data is exposed (only public METAR observations).
7. **Should we also archive IEM daily highs?** Yes, but as a separate future task. This plan only archives live METAR observations. Daily high targets can still be pulled from IEM `daily.py` on demand.

---

## Definition of done

- `data/db/weather_observations.db` exists and grows hourly.
- `scripts/ingest_live_metars.py` can be run manually or by cron.
- `scripts/generate_dashboard.py` produces a dark-themed static HTML page with stats, latest readings, and two charts.
- A Hermes cronjob runs ingestion every hour and pushes the dashboard daily.
- The dashboard is browsable from the GitHub Pages URL.
- All tests pass.

---

## Execution handoff

Plan complete and saved at:
`/opt/data/stock-research/dfw_temp_model/.hermes/plans/2026-06-16_180000-append-only-observation-database-and-dashboard.md`

Ready to execute using subagent-driven-development or direct TDD implementation. Shall I proceed?
