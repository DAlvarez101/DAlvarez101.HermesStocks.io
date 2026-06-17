#!/bin/bash
# Run the Polymarket weather-trading bot on its own schedule.
# This script is separate from cron_update_dashboard.sh so that trading failures
# never break the live dashboard ingestion pipeline.
set -euo pipefail

export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1

PROJECT_DIR="/opt/data/stock-research/dfw_temp_model"
DB_PATH="${PROJECT_DIR}/data/cache/db/weather_observations.db"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOG_FILE="${LOG_DIR}/polymarket_bot_$(date -u +%Y%m%d_%H%M%S).log"

# Ensure the project package is importable when running scripts directly.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Always run dry-run from cron by default; override with env if you want live.
DRY_RUN_FLAG="--dry-run"
if [ "${POLYMARKET_CRON_LIVE:-}" = "1" ]; then
    DRY_RUN_FLAG=""
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running Polymarket bot ..." | tee -a "$LOG_FILE"
"$PYTHON" "${PROJECT_DIR}/scripts/run_polymarket_bot.py" --db "$DB_PATH" $DRY_RUN_FLAG 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done." | tee -a "$LOG_FILE"
