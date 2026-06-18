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

# Ensure the project package is importable when running scripts directly.
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ingesting observations + HRRR + NBM forecasts ..."
# NWS API 5-minute observations (primary observation source, also runs every 5 min via separate cron)
"$PYTHON" scripts/ingest_nws_observations.py --db "$DB_PATH" --limit 25
# AviationWeather METAR + HRRR + NBM forecasts (both models update hourly; AviationWeather serves as cross-validation)
# To revert to AviationWeather-only: comment out the NWS line above and uncomment the line below
# "$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK" --hrrr
"$PYTHON" scripts/ingest_live_metars.py --db "$DB_PATH" --hours "$HOURS_BACK" --hrrr --nbm

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating dashboard ..."
"$PYTHON" scripts/generate_dashboard.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/${DASHBOARD_SUBDIR}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Generating DB viewer ..."
"$PYTHON" scripts/generate_db_viewer.py --db "$DB_PATH" --output-dir "${PAGES_DIR}/dfw-data-viewer" || echo "WARNING: DB viewer generation failed, continuing ..."

cd "$PAGES_DIR"

# Only commit/push if there are changes in the dashboard or viewer directories.
if ! git diff --quiet -- "${DASHBOARD_SUBDIR}/" "dfw-data-viewer/" || ! git diff --cached --quiet -- "${DASHBOARD_SUBDIR}/" "dfw-data-viewer/"; then
    git add "${DASHBOARD_SUBDIR}/" "dfw-data-viewer/"
    git commit -m "auto: update dfw dashboard + db viewer $(date -u +%Y-%m-%dT%H:%M:%SZ)" || true
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushing to GitHub Pages ..."
    git push origin main
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No dashboard changes to push."
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done."