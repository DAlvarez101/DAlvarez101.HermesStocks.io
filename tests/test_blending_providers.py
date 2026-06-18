"""Tests for the forecast provider interface and HRRR provider."""
import pandas as pd
import pytest
import sqlite3

from dfw_temp_model.blending.providers import ForecastProvider, HRRRProvider, NBMProvider


def test_protocol_is_abstract():
    """ForecastProvider is a Protocol — any class with the right methods conforms."""
    class FakeProvider:
        def fetch_forecast(self, conn, station, init_dt, forecast_hours=18):
            return pd.DataFrame()
        def recent_cycles(self, conn, station, min_hours=18):
            return []
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
            UNIQUE(init_dt, forecast_hour, station, source)
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
            UNIQUE(init_dt, forecast_hour, station, source)
        );
    """)
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
    assert "2026-01-01T06:00:00Z" not in cycles
    conn.close()


def test_nbm_provider_returns_forecast():
    """NBMProvider reads from the SQLite DB filtered by source='nbm-aws'."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station, source)
        );
    """)
    conn.executemany(
        "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "2026-01-01T00:00:00Z", "nbm-aws", "KDAL",
             "2026-01-01T00:00:00Z", 1, "2026-01-01T01:00:00Z", 32.0, -96.0, 80.0),
            (2, "2026-01-01T00:00:00Z", "hrrr-aws", "KDAL",
             "2026-01-01T00:00:00Z", 1, "2026-01-01T01:00:00Z", 32.0, -96.0, 82.0),
        ],
    )
    conn.commit()

    provider = NBMProvider()
    df = provider.fetch_forecast(conn, "KDAL", "2026-01-01T00:00:00Z", forecast_hours=2)
    # Should only return the nbm-aws row, not the hrrr-aws row
    assert len(df) == 1
    assert df.iloc[0]["source"] == "nbm-aws"
    assert "valid_dt" in df.columns
    conn.close()


def test_nbm_provider_recent_cycles():
    """recent_cycles returns init_dt strings sorted newest first, filtered by source."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station, source)
        );
    """)
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "nbm-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    for fh in range(1, 9):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "hrrr-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    conn.commit()

    provider = NBMProvider()
    cycles = provider.recent_cycles(conn, "KDAL", min_hours=18)
    assert "2026-01-01T12:00:00Z" in cycles
    conn.close()


def test_hrrr_and_nbm_providers_are_independent():
    """HRRR and NBM providers return disjoint sets when both sources exist."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE hrrr_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT, source TEXT, station TEXT,
            init_dt TEXT, forecast_hour INTEGER, valid_dt TEXT,
            lat REAL, lon REAL, tmpf REAL,
            UNIQUE(init_dt, forecast_hour, station, source)
        );
    """)
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "hrrr-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 80),
        )
    for fh in range(1, 19):
        conn.execute(
            "INSERT INTO hrrr_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (None, "t", "nbm-aws", "KDAL", "2026-01-01T12:00:00Z", fh, "t", 0, 0, 82),
        )
    conn.commit()

    hrrr_provider = HRRRProvider()
    nbm_provider = NBMProvider()
    hrrr_df = hrrr_provider.fetch_forecast(conn, "KDAL", "2026-01-01T12:00:00Z")
    nbm_df = nbm_provider.fetch_forecast(conn, "KDAL", "2026-01-01T12:00:00Z")
    assert len(hrrr_df) == 18
    assert len(nbm_df) == 18
    assert hrrr_df["tmpf"].iloc[0] == 80.0
    assert nbm_df["tmpf"].iloc[0] == 82.0
    conn.close()