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

    # METAR observations for KDAL: hours 12-20Z
    for h in range(12, 21):
        conn.execute(
            "INSERT INTO metar_observations (fetched_at, source, station, valid, lat, lon, tmpf) VALUES ('t','aviationweather','KDAL',?,32,-96,?)",
            (f"2026-06-17T{h:02d}:53:00+00:00", 80.0 + h),
        )

    # HRRR forecast cycle init at 18Z, f01-f18
    for fh in range(1, 19):
        valid_h = (18 + fh) % 24
        conn.execute(
            "INSERT INTO hrrr_forecasts (fetched_at, source, station, init_dt, forecast_hour, valid_dt, lat, lon, tmpf) VALUES ('t','hrrr-aws','KDAL','2026-06-17T18:00:00+00:00',?,?,32,-96,?)",
            (fh, f"2026-06-17T{valid_h:02d}:00:00+00:00", 79.0 + fh),
        )
    conn.commit()
    return conn


def test_blended_forecast_returns_correct_columns():
    """blended_forecast returns a DataFrame with all expected columns."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(conn, "KDAL", provider, init_dt="2026-06-17T18:00:00+00:00")
    assert "tmpf" in result.columns
    assert "tmpf_corrected" in result.columns
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
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (NULL,'t','hrrr','KDAL','2026-06-17T18:00:00Z',?,?,0,0,80)",
            (fh, f"2026-06-17T{(18+fh)%24:02d}:00:00Z"),
        )
    conn.commit()
    provider = HRRRProvider()
    result = blended_forecast(conn, "KDAL", provider, init_dt="2026-06-17T18:00:00Z")
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


def test_blended_forecast_has_trend_correction():
    """blended_forecast with trend_weight > 0 returns trend columns."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(
        conn, "KDAL", provider,
        init_dt="2026-06-17T18:00:00+00:00",
        trend_weight=0.15,
    )
    assert "trend_correction" in result.columns
    assert "tmpf_trend_adjusted" in result.columns
    conn.close()


def test_blended_forecast_trend_weight_zero():
    """When trend_weight=0, trend_adjusted = bias_corrected (no trend change)."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(
        conn, "KDAL", provider,
        init_dt="2026-06-17T18:00:00+00:00",
        trend_weight=0.0,
    )
    assert (result["tmpf_trend_adjusted"] == result["tmpf_corrected"]).all()
    conn.close()