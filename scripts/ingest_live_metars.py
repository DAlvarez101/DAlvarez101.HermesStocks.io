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
