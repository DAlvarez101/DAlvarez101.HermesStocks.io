"""Tests for the append-only SQLite observation database."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from dfw_temp_model.storage.obs_db import (
    ensure_schema,
    get_db,
    insert_observations,
    latest_by_station,
    read_all,
    row_count,
    station_count,
    time_range,
)


@pytest.fixture
def empty_conn(tmp_path):
    db_path = tmp_path / "test.db"
    return get_db(str(db_path))


def sample_df():
    return pd.DataFrame([
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
            "mslp": 101230.0,
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


def test_get_db_creates_file(tmp_path):
    db_path = tmp_path / "weather.db"
    conn = get_db(str(db_path))
    assert db_path.exists()
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "metar_observations" in tables
    conn.close()


def test_ensure_schema_idempotent(empty_conn):
    ensure_schema(empty_conn)
    ensure_schema(empty_conn)
    cur = empty_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    assert "metar_observations" in {r[0] for r in cur.fetchall()}


def test_insert_observations(empty_conn):
    inserted = insert_observations(empty_conn, sample_df(), source="aviationweather")
    assert inserted == 2
    assert row_count(empty_conn) == 2


def test_insert_or_ignore_prevents_duplicates(empty_conn):
    df = sample_df()
    insert_observations(empty_conn, df, source="aviationweather")
    insert_observations(empty_conn, df, source="aviationweather")
    assert row_count(empty_conn) == 2


def test_latest_by_station(empty_conn):
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
    insert_observations(empty_conn, df, source="aviationweather")
    latest = latest_by_station(empty_conn)
    assert len(latest) == 2
    kdfw = latest[latest["station"] == "KDFW"].iloc[0]
    assert kdfw["tmpf"] == 84.0


def test_time_range_and_station_count(empty_conn):
    assert time_range(empty_conn) == (None, None)
    assert station_count(empty_conn) == 0
    insert_observations(empty_conn, sample_df(), source="aviationweather")
    first, last = time_range(empty_conn)
    assert first is not None and last is not None
    assert station_count(empty_conn) == 2
