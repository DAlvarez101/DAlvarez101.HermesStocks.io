"""Tests for the dashboard generator."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dfw_temp_model.storage.obs_db import get_db, insert_hrrr_forecasts, insert_observations


@pytest.fixture
def populated_db(tmp_path):
    db_path = tmp_path / "dash.db"
    conn = get_db(str(db_path))
    df = pd.DataFrame([
        {
            "station": "KDFW",
            "valid": "2026-06-16T16:00:00+00:00",
            "lat": 32.897,
            "lon": -97.038,
            "tmpf": 80.0,
            "dewpf": 65.0,
            "drct": 10.0,
            "sknt": 5.0,
            "skyc1": "FEW",
            "mslp": None,
            "p01i": None,
        },
        {
            "station": "KDFW",
            "valid": "2026-06-16T17:00:00+00:00",
            "lat": 32.897,
            "lon": -97.038,
            "tmpf": 84.0,
            "dewpf": 69.0,
            "drct": 0.0,
            "sknt": 3.0,
            "skyc1": "SCT",
            "mslp": None,
            "p01i": None,
        },
        {
            "station": "KDAL",
            "valid": "2026-06-16T17:00:00+00:00",
            "lat": 32.848,
            "lon": -96.851,
            "tmpf": 84.0,
            "dewpf": 70.0,
            "drct": 120.0,
            "sknt": 4.0,
            "skyc1": "SCT",
            "mslp": None,
            "p01i": None,
        },
    ])
    insert_observations(conn, df, source="test")
    hrrr_df = pd.DataFrame([
        {
            "station": "KDAL",
            "init_dt": "2026-06-16T16:00:00+00:00",
            "forecast_hour": 1,
            "valid_dt": "2026-06-16T17:00:00+00:00",
            "lat": 32.848,
            "lon": -96.851,
            "tmpf": 85.5,
        }
    ])
    insert_hrrr_forecasts(conn, hrrr_df, source="test")
    conn.close()
    return str(db_path)


def test_summary_stats(populated_db):
    conn = get_db(populated_db)
    # Import via importlib since scripts/ is not a package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = mod.summary_stats(conn)
    conn.close()
    assert stats["total_rows"] == 3
    assert stats["station_count"] == 2
    assert "2026-06-16" in stats["first_obs"]


def test_generate_dashboard_creates_html(populated_db, tmp_path):
    output_dir = tmp_path / "dash"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dashboard.py",
            "--db",
            populated_db,
            "--output-dir",
            str(output_dir),
        ],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, result.stderr
    index_path = output_dir / "index.html"
    assert index_path.exists()
    html = index_path.read_text(encoding="utf-8")
    assert "DFW Live Weather Dashboard" in html
    assert "KDAL" in html
    assert "METAR vs HRRR" in html
    assert "85.5°F" in html
    assert "+1.5°F" in html
    # Two matplotlib base64 chart images remain; HRRR is an interactive Plotly chart.
    assert html.count("data:image/png;base64,") >= 2
    assert "plotly" in html.lower()
