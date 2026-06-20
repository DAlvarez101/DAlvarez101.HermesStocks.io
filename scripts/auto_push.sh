#!/usr/bin/env bash
# auto_push.sh — if a git remote is configured, push commits to it.
# Safe to run even if no remote exists (just exits quietly).
set -e
cd "$(dirname "$0")/.."

REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REMOTE" ]; then
    echo "[auto-push] No remote configured — skipping."
    exit 0
fi

# Push if we have unpushed commits
if git log --oneline @{u}..HEAD 2>/dev/null | grep -q .; then
    git push origin HEAD 2>&1 || echo "[auto-push] Push failed — will retry next run."
    echo "[auto-push] Pushed."
else
    echo "[auto-push] Already up to date."
fi