"""Tests for AviationWeather.gov METAR JSON fetcher."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS, TARGET_ICAO
from dfw_temp_model.data.aviationweather import (
    build_aviationweather_url,
    fetch_aviationweather,
    parse_metar_json,
)


@pytest.fixture
def sample_payload():
    return [
        {
            "icaoId": "KDFW",
            "obsTime": 1781629200,
            "reportTime": "2026-06-16T17:00:00.000Z",
            "temp": 28.9,
            "dewp": 22.2,
            "wdir": 180,
            "wspd": 7,
            "altim": 1012.3,
            "cover": "OVC",
            "clouds": [{"cover": "OVC", "base": 2500}],
        },
        {
            "icaoId": "KDAL",
            "obsTime": 1781629200,
            "reportTime": "2026-06-16T17:00:00.000Z",
            "temp": 28.3,
            "dewp": 21.7,
            "wdir": "VRB",
            "wspd": 3,
            "altim": 1011.9,
            "clouds": [{"cover": "SCT", "base": 3000}],
        },
    ]


def test_build_aviationweather_url():
    url = build_aviationweather_url(STATIONS[:3], hours=2)
    assert url.startswith("https://aviationweather.gov/api/data/metar")
    assert "ids=KDFW%2CKDAL%2CKADS" in url or "ids=KDFW,KDAL,KADS" in url
    assert "format=json" in url
    assert "hours=2" in url


def test_parse_metar_json_empty():
    df = parse_metar_json([], STATIONS)
    assert df.empty
    assert list(df.columns) == [
        "station",
        "valid",
        "lat",
        "lon",
        "tmpf",
        "dewpf",
        "drct",
        "sknt",
        "skyc1",
        "mslp",
        "p01i",
    ]


def test_parse_metar_json_sample(sample_payload):
    df = parse_metar_json(sample_payload, STATIONS)
    assert len(df) == 2

    kdfw = df[df["station"] == "KDFW"].iloc[0]
    assert kdfw["tmpf"] == pytest.approx(28.9 * 9.0 / 5.0 + 32.0, abs=0.01)
    assert kdfw["dewpf"] == pytest.approx(22.2 * 9.0 / 5.0 + 32.0, abs=0.01)
    assert kdfw["drct"] == 180.0
    assert kdfw["sknt"] == 7.0
    assert kdfw["skyc1"] == "OVC"
    assert kdfw["lat"] == 32.897
    assert kdfw["lon"] == -97.038

    kdal = df[df["station"] == "KDAL"].iloc[0]
    assert pd.isna(kdal["drct"])  # VRB becomes NaN
    assert kdal["skyc1"] == "SCT"

    expected_valid = datetime.fromtimestamp(1781629200, tz=timezone.utc)
    assert kdfw["valid"] == expected_valid


def test_parse_metar_json_vrb_and_zero_wind():
    payload = [
        {
            "icaoId": "KDTO",
            "obsTime": 1781629200,
            "temp": 20.0,
            "dewp": 10.0,
            "wdir": 0,
            "wspd": 0,
        }
    ]
    df = parse_metar_json(payload, STATIONS)
    row = df.iloc[0]
    assert row["drct"] == 0.0
    assert row["sknt"] == 0.0


@pytest.mark.network
@pytest.mark.slow
def test_fetch_aviationweather_smoke(tmp_path):
    """Real network smoke test: fetch live METARs for all 8 DFW stations."""
    cache = tmp_path / "live_metars.parquet"
    df = fetch_aviationweather(STATIONS, hours=2, cache_path=str(cache))

    assert not df.empty
    assert TARGET_ICAO in df["station"].values
    assert all(col in df.columns for col in ["station", "valid", "tmpf", "drct", "sknt"])

    # Temperature should be in Fahrenheit now.
    assert df["tmpf"].dropna().min() > -50
    assert df["tmpf"].dropna().max() < 150

    # The cache file should have been written.
    assert cache.exists()
    cached = pd.read_parquet(cache)
    assert not cached.empty
