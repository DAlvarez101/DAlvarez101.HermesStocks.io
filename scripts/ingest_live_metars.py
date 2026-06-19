"""Hourly ingestion script: fetch live METARs and append to SQLite."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make dfw_temp_model importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dfw_temp_model.config import CACHE_DIR, STATIONS
from dfw_temp_model.data.aviationweather import fetch_aviationweather
from dfw_temp_model.data.hrrr import fetch_hrrr_forecast_range, fetch_latest_hrrr_2m_temp
from dfw_temp_model.storage.obs_db import get_db, insert_hrrr_forecasts, insert_observations


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
    parser.add_argument(
        "--hrrr",
        action="store_true",
        help="Also fetch HRRR 2m temperature forecast for the next 18 hours",
    )
    parser.add_argument(
        "--hrrr-hours",
        type=int,
        default=18,
        help="Number of HRRR forecast hours to fetch (default 18)",
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
    print(f"Inserted {inserted} METAR rows. Total rows in database: {total}")

    if args.hrrr:
        print(f"[{fetched_at}] Fetching HRRR 2m temp (f01-f{args.hrrr_hours:02d}) ...")
        hrrr_df = fetch_hrrr_forecast_range(
            STATIONS, max_forecast_hour=args.hrrr_hours, lookback_hours=6
        )
        if hrrr_df.empty:
            print("No HRRR forecast returned.", file=sys.stderr)
        else:
            hrrr_inserted = insert_hrrr_forecasts(
                conn, hrrr_df, source="hrrr-aws", fetched_at=fetched_at
            )
            hrrr_total = conn.execute("SELECT COUNT(*) FROM hrrr_forecasts").fetchone()[0]
            print(
                f"Inserted {hrrr_inserted} HRRR rows "
                f"({len(hrrr_df)} fetched across {hrrr_df['forecast_hour'].nunique()} hours). "
                f"Total HRRR rows: {hrrr_total}"
            )

    conn.close()


if __name__ == "__main__":
    main()
