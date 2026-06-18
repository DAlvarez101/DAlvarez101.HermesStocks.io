"""Fetch live METARs for the DFW station network and print a summary."""
import argparse
from pathlib import Path
import sys

# Make dfw_temp_model importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dfw_temp_model.config import CACHE_DIR, STATIONS
from dfw_temp_model.data.aviationweather import fetch_aviationweather


def main():
    parser = argparse.ArgumentParser(description="Fetch live METARs")
    parser.add_argument("--hours", type=int, default=2, help="Hours back to fetch")
    parser.add_argument(
        "--cache",
        type=str,
        default=str(Path(CACHE_DIR) / "live_metars.parquet"),
        help="Parquet cache path",
    )
    args = parser.parse_args()

    df = fetch_aviationweather(STATIONS, hours=args.hours, cache_path=args.cache)
    if df.empty:
        print("No live METARs returned.", file=sys.stderr)
        sys.exit(1)

    latest = df.sort_values("valid").groupby("station").last().reset_index()
    display = latest[["station", "valid", "tmpf", "dewpf", "drct", "sknt", "skyc1"]].copy()
    display.columns = ["Station", "Valid (UTC)", "Temp (F)", "Dewpt (F)", "Wind Dir", "Wind (kt)", "Sky"]
    print(display.to_string(index=False))
    print(f"\nCached to: {args.cache}")


if __name__ == "__main__":
    main()
