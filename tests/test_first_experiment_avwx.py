"""Tests for the --obs-source option in run_first_experiment.py."""
import subprocess
import sys

import pytest


@pytest.mark.network
@pytest.mark.slow
def test_first_experiment_aviationweather_option():
    """The experiment script can fetch live METARs and print a summary."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_first_experiment.py",
            "--obs-source",
            "aviationweather",
            "--start-date",
            "2026-06-16",
            "--end-date",
            "2026-06-16",
            "--aviationweather-hours",
            "2",
        ],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=90,
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, result.stderr
    assert "Live METAR summary (AviationWeather.gov):" in result.stdout
    assert "KDFW" in result.stdout
    assert "Historical daily-high experiments require --obs-source=iem" in result.stdout
