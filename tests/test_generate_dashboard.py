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
    hrrr_rows = []
    for fh in range(1, 19):
        hrrr_rows.append(
            {
                "station": "KDAL",
                "init_dt": "2026-06-16T16:00:00+00:00",
                "forecast_hour": fh,
                "valid_dt": (pd.Timestamp("2026-06-16T16:00:00+00:00") + pd.Timedelta(hours=fh)).isoformat(),
                "lat": 32.848,
                "lon": -96.851,
                "tmpf": 85.5 + (fh - 1) * 0.1,
            }
        )
    hrrr_df = pd.DataFrame(hrrr_rows)
    insert_hrrr_forecasts(conn, hrrr_df, source="hrrr-aws")

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


def test_compute_cycle_stats():
    """_compute_cycle_stats extracts tile values from blended result + bias trace."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    blended = pd.DataFrame({
        "valid_dt": pd.to_datetime(["2026-06-19T13:00:00Z", "2026-06-19T14:00:00Z"], utc=True),
        "forecast_hour": [1, 2],
        "tmpf": [80.2, 78.8],
        "tmpf_corrected": [74.7, 73.2],
        "bias_applied": [-5.5, -5.5],
        "trend_correction": [0.04, -0.02],
        "tmpf_trend_adjusted": [74.7, 73.2],
        "uncertainty_low": [71.0, 69.5],
        "uncertainty_high": [78.4, 76.9],
    })
    bias_df = pd.DataFrame({
        "valid_hour": pd.to_datetime(["2026-06-19T12:00:00Z", "2026-06-19T13:00:00Z"], utc=True),
        "bias": [-0.06, -1.71],
        "bias_std": [2.86, 1.03],
        "n_matches": [196, 296],
        "error_mean": [-8.13, -10.80],
        "obs_mean": [73.1, 70.7],
        "fcst_mean": [81.3, 81.5],
    })
    latest_obs = pd.Series({"valid": pd.Timestamp("2026-06-19T14:20:00Z", tz="UTC"), "tmpf": 71.6})

    stats = mod._compute_cycle_stats(blended, bias_df, latest_obs, halflife_hours=2.0, trend_weight=0.15)

    assert stats["bias_applied"] == pytest.approx(-5.5, abs=0.01)
    assert stats["trend_min"] == pytest.approx(-0.02, abs=0.01)
    assert stats["trend_max"] == pytest.approx(0.04, abs=0.01)
    assert stats["n_matched_pairs"] == 296
    assert stats["n_matched_hours"] == 2
    assert stats["latest_obs_tmpf"] == pytest.approx(71.6, abs=0.01)
    assert "latest_obs_time" in stats
    assert "corrected_at_obs_hour" in stats
    assert "hindsight_error" in stats
    assert stats["halflife_hours"] == 2.0
    assert stats["trend_weight"] == 0.15


def test_build_stat_tiles_html():
    """_build_stat_tiles_html returns a div with stat tile cards."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stats = {
        "bias_applied": -5.5,
        "trend_min": -0.19,
        "trend_max": 0.13,
        "n_matched_pairs": 321,
        "n_matched_hours": 6,
        "uncertainty_plus": 3.6,
        "max_correction": -5.7,
        "latest_obs_tmpf": 71.6,
        "latest_obs_time": pd.Timestamp("2026-06-19T14:20:00Z", tz="UTC"),
        "corrected_at_obs_hour": 73.2,
        "raw_at_obs_hour": 78.8,
        "hindsight_error": 1.6,
        "hindsight_raw_error": 7.2,
        "halflife_hours": 2.0,
        "trend_weight": 0.15,
    }
    html = mod._build_stat_tiles_html(stats, cycle_idx=0, visible=True)
    assert 'id="stats-0"' in html
    assert "-5.5" in html
    assert "71.6" in html
    assert "73.2" in html
    assert "1.6" in html
    assert "2h" in html
    assert "15%" in html


def test_build_bias_table_html():
    """_build_bias_table_html returns a details element with per-hour rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bias_df = pd.DataFrame({
        "valid_hour": pd.to_datetime(["2026-06-19T12:00:00Z", "2026-06-19T13:00:00Z"], utc=True),
        "bias": [-0.06, -1.71],
        "bias_std": [2.86, 1.03],
        "n_matches": [196, 296],
        "error_mean": [-8.13, -10.80],
        "obs_mean": [73.1, 70.7],
        "fcst_mean": [81.3, 81.5],
    })
    html = mod._build_bias_table_html(bias_df, cycle_idx=0, visible=True)
    assert 'id="bias-table-0"' in html
    assert "<table" in html
    assert "73.1" in html
    assert "81.3" in html
    assert "<details" in html


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
    # New: stat tiles and dropdown
    assert 'id="cycle-selector"' in html
    assert "cycle-stats" in html
    assert "stat-tile" in html
    assert "Bias Applied" in html
    assert "Hindsight Error" in html
    assert "bias-table" in html
    assert "switchBlendedCycle" in html
