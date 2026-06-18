"""Fetch 5-minute observations from the NWS API for all stations and store in SQLite."""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make dfw_temp_model importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dfw_temp_model.config import STATIONS, CACHE_DIR
from dfw_temp_model.data.nws_api import fetch_nws_observations
from dfw_temp_model.storage.obs_db import get_db, insert_observations


def main():
    parser = argparse.ArgumentParser(description="Ingest 5-minute NWS API observations")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Max observations per station (default 25 = ~2 hours of 5-min data)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="nws-api",
        help="Source label to store in the database",
    )
    args = parser.parse_args()

    fetched_at = datetime.now(timezone.utc).isoformat()
    conn = get_db(args.db)
    total_inserted = 0

    for station in STATIONS:
        icao = station.icao
        try:
            df = fetch_nws_observations(icao, limit=args.limit)
            if df.empty:
                print(f"  {icao}: no observations returned", file=sys.stderr)
                continue
            inserted = insert_observations(conn, df, source=args.source, fetched_at=fetched_at)
            total_inserted += inserted
            print(f"  {icao}: inserted {inserted} rows ({len(df)} fetched)")
        except Exception as exc:
            # Don't let one station failure stop the others.
            print(f"  {icao}: ERROR - {exc}", file=sys.stderr)
            continue
        # Brief pause between stations to be polite to the API.
        time.sleep(0.5)

    total = conn.execute("SELECT COUNT(*) FROM metar_observations").fetchone()[0]
    print(f"[{fetched_at}] Inserted {total_inserted} NWS API rows. Total METAR rows: {total}")
    conn.close()


if __name__ == "__main__":
    main()