"""Tests for the live METAR polling script."""
import subprocess
import sys

import pandas as pd
import pytest


@pytest.mark.network
@pytest.mark.slow
def test_live_script_runs(tmp_path):
    cache = tmp_path / "live_metars.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_live_metars.py",
            "--cache",
            str(cache),
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
    assert "KDFW" in result.stdout
    assert "Temp (F)" in result.stdout
    assert cache.exists()
    df = pd.read_parquet(cache)
    assert not df.empty
    assert "KDFW" in df["station"].values
