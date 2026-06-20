#!/bin/bash
# Fetch HRRR 2m temperature forecast only (no METAR, no dashboard regen).
# Runs at :50 of every hour via Hermes cron. The next 5-minute NWS cron tick
# (within ~5 minutes) will regenerate the dashboard and pick up the newly
# ingested HRRR cycle automatically.
#
# Uses --hrrr-prefer-current so the current hour's cycle is tried first.
# If the current run is not yet fully published, the fetcher falls back to
# the previous hour's cycle (same as the existing :05 behaviour).
set -euo pipefail

export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1

PROJECT_DIR="/opt/data/stock-research/dfw_temp_model"
DB_PATH="${PROJECT_DIR}/data/cache/db/weather_observations.db"

cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Fetching HRRR 2m temp (hrrr-only, prefer-current) ..."
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hrrr-only --hrrr-prefer-current

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."