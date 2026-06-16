#!/bin/bash
# Run hourly ingestion, regenerate dashboard, and push to GitHub Pages.
#
# This script is called by the Hermes cronjob. It is written as a bash wrapper
# because the cron scheduler runs shell commands more reliably than multi-line
# Python invocations, and it lets us export the Docker-root variables once.
set -euo pipefail

export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1

PROJECT_DIR="/opt/data/stock-research/dfw_temp_model"
PAGES_DIR="/opt/data/DAlvarez101.HermesStocks.io"
DASHBOARD_SUBDIR="dfw-live-dashboard"
DB_PATH="${PROJECT_DIR}/data/cache/db/weather_observations.db"
HOURS_BACK=3

cd "$PROJECT_DIR"

# Use project venv explicitly.
PYTHON="${PROJECT_DIR}/.venv/bin/python"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ingesting live METARs ..."
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating dashboard ..."
"$PYTHON" scripts/generate_dashboard.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/${DASHBOARD_SUBDIR}"

cd "$PAGES_DIR"

# Only commit/push if there are changes in the dashboard directory.
if ! git diff --quiet -- "${DASHBOARD_SUBDIR}/" || ! git diff --cached --quiet -- "${DASHBOARD_SUBDIR}/"; then
    git add "${DASHBOARD_SUBDIR}/"
    git commit -m "auto: update dfw live dashboard $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushing to GitHub Pages ..."
    git push origin main
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No dashboard changes to push."
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."
