"""Append-only SQLite storage for METAR observations and HRRR forecasts."""
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

CREATE TABLE IF NOT EXISTS hrrr_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'hrrr-aws',
    station TEXT NOT NULL,
    init_dt TEXT NOT NULL,
    forecast_hour INTEGER NOT NULL,
    valid_dt TEXT NOT NULL,
    lat REAL,
    lon REAL,
    tmpf REAL,
    UNIQUE(init_dt, forecast_hour, station)
);

CREATE INDEX IF NOT EXISTS idx_hrrr_station_valid
    ON hrrr_forecasts(station, valid_dt);

CREATE INDEX IF NOT EXISTS idx_hrrr_fetched_at
    ON hrrr_forecasts(fetched_at);
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


def insert_hrrr_forecasts(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    source: str = "hrrr-aws",
    fetched_at: Optional[str] = None,
) -> int:
    """Insert HRRR forecast rows using INSERT OR IGNORE.

    Returns the number of newly inserted rows.
    """
    if df.empty:
        return 0
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).isoformat()

    columns = [
        "fetched_at", "source", "station", "init_dt", "forecast_hour",
        "valid_dt", "lat", "lon", "tmpf",
    ]
    rows = []
    for _, row in df.iterrows():
        init_dt = row.get("init_dt")
        valid_dt = row.get("valid_dt")
        if isinstance(init_dt, pd.Timestamp):
            init_dt = init_dt.isoformat()
        if isinstance(valid_dt, pd.Timestamp):
            valid_dt = valid_dt.isoformat()
        forecast_hour = row.get("forecast_hour", 0)
        rows.append((
            fetched_at, source, row.get("station"), init_dt,
            int(forecast_hour) if forecast_hour is not None else 0,
            valid_dt,
            row.get("lat"), row.get("lon"), row.get("tmpf"),
        ))

    cursor = conn.cursor()
    cursor.executemany(
        f"""
        INSERT OR IGNORE INTO hrrr_forecasts (
            {', '.join(columns)}
        ) VALUES ({', '.join('?' for _ in columns)})
        """,
        rows,
    )
    conn.commit()
    return cursor.rowcount


def read_all(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the entire METAR table as a DataFrame."""
    return pd.read_sql_query(
        "SELECT * FROM metar_observations ORDER BY valid", conn
    )


def latest_by_station(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the most recent METAR row per station."""
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


def latest_hrrr_forecast(conn: sqlite3.Connection, station: Optional[str] = None) -> pd.DataFrame:
    """Return the most recent HRRR forecast row (optionally filtered by station)."""
    if station:
        query = """
            SELECT *
            FROM hrrr_forecasts
            WHERE station = ?
            ORDER BY init_dt DESC, forecast_hour DESC
            LIMIT 1
        """
        return pd.read_sql_query(query, conn, params=[station])
    return pd.read_sql_query(
        """
        SELECT *
        FROM hrrr_forecasts
        ORDER BY init_dt DESC, forecast_hour DESC
        LIMIT 1
        """,
        conn,
    )


def hrrr_for_valid_hour(
    conn: sqlite3.Connection,
    station: str,
    valid_hour: datetime,
) -> pd.DataFrame:
    """Return HRRR forecasts for a station whose valid hour matches the given hour."""
    start = valid_hour.strftime("%Y-%m-%dT%H:00:00")
    end = (valid_hour + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:00:00")
    return pd.read_sql_query(
        """
        SELECT *
        FROM hrrr_forecasts
        WHERE station = ? AND valid_dt >= ? AND valid_dt < ?
        ORDER BY init_dt DESC
        """,
        conn,
        params=[station, start, end],
    )


def hrrr_forecast_range(
    conn: sqlite3.Connection,
    station: str,
    from_dt: Optional[str] = None,
    limit: int = 18,
) -> pd.DataFrame:
    """Return HRRR forecast rows for *station* ordered by valid_dt ascending.

    If *from_dt* is given (ISO string), only rows with valid_dt >= from_dt
    are returned.  Otherwise all rows are returned (useful when you know the
    DB only contains one fresh cycle).  *limit* caps the number of rows.
    """
    if from_dt:
        query = """
            SELECT * FROM hrrr_forecasts
            WHERE station = ? AND valid_dt >= ?
            ORDER BY valid_dt ASC
            LIMIT ?
        """
        return pd.read_sql_query(query, conn, params=[station, from_dt, limit])
    query = """
        SELECT * FROM hrrr_forecasts
        WHERE station = ?
        ORDER BY valid_dt ASC
        LIMIT ?
    """
    return pd.read_sql_query(query, conn, params=[station, limit])


def latest_complete_hrrr_cycle(
    conn: sqlite3.Connection, station: str, required_hours: int = 18
) -> Optional[str]:
    """Return the latest init_dt (ISO string) that has >= required_hours frames."""
    df = pd.read_sql_query(
        """
        SELECT init_dt, COUNT(*) AS n
        FROM hrrr_forecasts
        WHERE station = ?
        GROUP BY init_dt
        HAVING n >= ?
        ORDER BY init_dt DESC
        LIMIT 1
        """,
        conn,
        params=[station, required_hours],
    )
    if df.empty:
        return None
    return str(df.iloc[0]["init_dt"])


def hrrr_forecast_for_cycle(
    conn: sqlite3.Connection, station: str, init_dt: str
) -> pd.DataFrame:
    """Return every forecast hour for a given station and model cycle."""
    return pd.read_sql_query(
        """
        SELECT * FROM hrrr_forecasts
        WHERE station = ? AND init_dt = ?
        ORDER BY forecast_hour ASC
        """,
        conn,
        params=[station, init_dt],
    )


def row_count(conn: sqlite3.Connection) -> int:
    """Return total METAR row count."""
    return conn.execute("SELECT COUNT(*) FROM metar_observations").fetchone()[0]


def hrrr_row_count(conn: sqlite3.Connection) -> int:
    """Return total HRRR forecast row count."""
    return conn.execute("SELECT COUNT(*) FROM hrrr_forecasts").fetchone()[0]


def time_range(conn: sqlite3.Connection) -> tuple[Optional[str], Optional[str]]:
    """Return (first_obs_iso, last_obs_iso) or (None, None)."""
    row = conn.execute(
        "SELECT MIN(valid), MAX(valid) FROM metar_observations"
    ).fetchone()
    return (row[0], row[1])


def station_count(conn: sqlite3.Connection) -> int:
    """Return number of distinct METAR stations."""
    return conn.execute(
        "SELECT COUNT(DISTINCT station) FROM metar_observations"
    ).fetchone()[0]
