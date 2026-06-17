#!/usr/bin/env python3
"""Run the Polymarket weather-trading bot once."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Ensure the project package is importable when this script is executed directly.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dfw_temp_model.config import CACHE_DIR
from dfw_temp_model.trading.config import load_config
from dfw_temp_model.trading.bot import run_bot


def main():
    parser = argparse.ArgumentParser(description="Polymarket weather-trading bot")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to the existing weather SQLite DB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Override POLYMARKET_DRY_RUN to true",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.dry_run:
        cfg = replace(cfg, dry_run=True)

    if not cfg.dry_run:
        print("WARNING: running in LIVE trading mode", file=sys.stderr)
        print("Set POLYMARKET_DRY_RUN=true or use --dry-run to simulate.", file=sys.stderr)

    report = run_bot(cfg, args.db)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
