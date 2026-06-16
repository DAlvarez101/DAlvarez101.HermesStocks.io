"""Tests for the hourly METAR ingestion script."""
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.network
@pytest.mark.slow
def test_ingest_script_smoke(tmp_path):
    db_path = tmp_path / "ingest.db"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_live_metars.py",
            "--db",
            str(db_path),
            "--hours",
            "2",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, result.stderr
    assert "Inserted" in result.stdout
    assert db_path.exists()

    df = pd.read_sql_query("SELECT * FROM metar_observations", f"sqlite:///{db_path}")
    assert not df.empty
    assert "KDFW" in df["station"].values


@pytest.mark.network
@pytest.mark.slow
def test_ingest_script_idempotent(tmp_path):
    db_path = tmp_path / "ingest.db"
    subprocess.run(
        [sys.executable, "scripts/ingest_live_metars.py", "--db", str(db_path), "--hours", "2"],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=60,
    )
    result2 = subprocess.run(
        [sys.executable, "scripts/ingest_live_metars.py", "--db", str(db_path), "--hours", "2"],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("STDOUT2:", result2.stdout)
    print("STDERR2:", result2.stderr)
    assert result2.returncode == 0, result2.stderr
    assert "Inserted 0 new rows" in result2.stdout
