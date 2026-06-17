"""Combined ingest + dashboard update, callable as a Python script."""
import argparse
import subprocess
import sys
from pathlib import Path

from dfw_temp_model.config import CACHE_DIR


def run(cmd: list[str], cwd: str = ".") -> int:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Ingest METARs and update dashboard")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/data/DAlvarez101.HermesStocks.io/dfw-live-dashboard",
    )
    parser.add_argument("--hours", type=int, default=3)
    args = parser.parse_args()

    python = sys.executable
    rc = run([
        python, "scripts/ingest_live_metars.py",
        "--db", args.db,
        "--hours", str(args.hours),
    ])
    if rc != 0:
        sys.exit(rc)

    rc = run([
        python, "scripts/generate_dashboard.py",
        "--db", args.db,
        "--output-dir", args.output_dir,
    ])
    sys.exit(rc)


if __name__ == "__main__":
    main()
