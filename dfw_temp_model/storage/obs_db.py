"""Append-only SQLite storage for METAR observations."""
import sqlite3
from datetime import datetime, timezone
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
        fetched_at = datetime.now(timezone.utc).isoformat()

    columns = [
        "fetched_at", "source", "station", "valid", "lat", "lon",
        "tmpf", "dewpf", "drct", "sknt", "skyc1", "mslp", "p01i",
    ]
    rows = []
    for _, row in df.iterrows():
        valid = row.get("valid")
        if isinstance(valid, pd.Timestamp):
            valid = valid.isoformat()
        rows.append((
            fetched_at,
            source,
            row.get("station"),
            valid,
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


def time_range(conn: sqlite3.Connection) -> tuple[Optional[str], Optional[str]]:
    """Return (first_obs_iso, last_obs_iso) or (None, None)."""
    row = conn.execute(
        "SELECT MIN(valid), MAX(valid) FROM metar_observations"
    ).fetchone()
    return (row[0], row[1])


def station_count(conn: sqlite3.Connection) -> int:
    """Return number of distinct stations."""
    return conn.execute(
        "SELECT COUNT(DISTINCT station) FROM metar_observations"
    ).fetchone()[0]
