"""Forecast provider interface and implementations.

Each provider wraps a model source (HRRR, GFS, NAM, etc.) behind a common
interface so the blending logic can treat all models uniformly. The DB
schema already has a ``source`` column in ``hrrr_forecasts``; new models
will store their rows with a different source tag.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd
import sqlite3


@runtime_checkable
class ForecastProvider(Protocol):
    """Abstract interface for a forecast model source.

    Implementations read from the SQLite DB (or fetch live) and return
    DataFrames with at minimum: valid_dt, tmpf, forecast_hour, init_dt.
    """

    def fetch_forecast(
        self,
        conn: sqlite3.Connection,
        station: str,
        init_dt: str,
        forecast_hours: int = 18,
    ) -> pd.DataFrame:
        """Return forecast rows for one model cycle at one station."""
        ...

    def recent_cycles(
        self,
        conn: sqlite3.Connection,
        station: str,
        min_hours: int = 18,
    ) -> list[str]:
        """Return init_dt strings (newest first) with at least min_hours frames."""
        ...


class HRRRProvider:
    """Reads HRRR forecasts from the SQLite ``hrrr_forecasts`` table.

    This is a thin wrapper around the existing storage queries. The actual
    HRRR fetching (downloading GRIB2 from AWS) lives in
    ``dfw_temp_model.data.hrrr`` and is not duplicated here.
    """

    SOURCE = "hrrr-aws"

    def fetch_forecast(
        self,
        conn: sqlite3.Connection,
        station: str,
        init_dt: str,
        forecast_hours: int = 18,
    ) -> pd.DataFrame:
        """Return all forecast hours for a given station and init cycle."""
        df = pd.read_sql_query(
            """
            SELECT init_dt, forecast_hour, valid_dt, tmpf, lat, lon, station, source
            FROM hrrr_forecasts
            WHERE station = ? AND init_dt = ?
            ORDER BY forecast_hour ASC
            """,
            conn,
            params=[station, init_dt],
        )
        return df

    def recent_cycles(
        self,
        conn: sqlite3.Connection,
        station: str,
        min_hours: int = 18,
    ) -> list[str]:
        """Return init_dt strings that have at least min_hours of frames.

        Sorted newest-first. Only includes complete cycles.
        """
        df = pd.read_sql_query(
            """
            SELECT init_dt, COUNT(*) AS n
            FROM hrrr_forecasts
            WHERE station = ?
            GROUP BY init_dt
            HAVING n >= ?
            ORDER BY init_dt DESC
            """,
            conn,
            params=[station, min_hours],
        )
        if df.empty:
            return []
        return df["init_dt"].tolist()