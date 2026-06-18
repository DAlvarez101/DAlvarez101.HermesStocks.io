#!/bin/bash
# Fetch 5-minute NWS API observations and regenerate the dashboard.
# Runs every 5 minutes via Hermes cron. Does NOT fetch HRRR.
set -euo pipefail

export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1

PROJECT_DIR="/opt/data/stock-research/dfw_temp_model"
PAGES_DIR="/opt/data/DAlvarez101.HermesStocks.io"
DASHBOARD_SUBDIR="dfw-live-dashboard"
DB_PATH="${PROJECT_DIR}/data/cache/db/weather_observations.db"

cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Fetching NWS API 5-minute observations ..."
"$PYTHON" scripts/ingest_nws_observations.py --db "$DB_PATH" --limit 25

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Regenerating dashboard ..."
"$PYTHON" scripts/generate_dashboard.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/${DASHBOARD_SUBDIR}"

cd "$PAGES_DIR"

if ! git diff --quiet -- "${DASHBOARD_SUBDIR}/" || ! git diff --cached --quiet -- "${DASHBOARD_SUBDIR}/"; then
    git add "${DASHBOARD_SUBDIR}/"
    git commit -m "auto: 5-min NWS obs update $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushing to GitHub Pages ..."
    git pull origin main --no-rebase 2>/dev/null || true
    git push origin main
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No dashboard changes to push."
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."