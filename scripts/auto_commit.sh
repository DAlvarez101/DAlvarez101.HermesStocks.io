#!/usr/bin/env bash
# auto_commit.sh — auto-commit any untracked/modified source files.
# Run by cron jobs after dashboard regeneration, or manually.
# Only commits .py files (not data/ or __pycache__/).
set -e
cd "$(dirname "$0")/.."

# Stage all tracked + untracked .py files (not data, not pycache)
git add -- '*.py' '*.sh' '*.md' '*.txt' '*.toml' '*.cfg' '*.ini' '.gitignore' 2>/dev/null || true

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "[auto-commit] Nothing to commit."
    exit 0
fi

# Commit with timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
git commit -m "auto: source file snapshot ${TIMESTAMP}"
echo "[auto-commit] Committed at ${TIMESTAMP}"