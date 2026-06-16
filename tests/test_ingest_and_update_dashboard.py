"""Test the combined ingest + dashboard script end-to-end."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.network
@pytest.mark.slow
def test_ingest_and_update_dashboard(tmp_path):
    db_path = tmp_path / "dash.db"
    output_dir = tmp_path / "dash"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_and_update_dashboard.py",
            "--db",
            str(db_path),
            "--output-dir",
            str(output_dir),
            "--hours",
            "2",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=120,
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, result.stderr
    assert db_path.exists()
    assert (output_dir / "index.html").exists()
    assert "Inserted" in result.stdout
    assert "Dashboard written to" in result.stdout
