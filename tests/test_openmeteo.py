import os

import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS
from dfw_temp_model.data.openmeteo import build_url, fetch_all_stations, fetch_hourly_temp


def test_build_openmeteo_url():
    url = build_url(32.897, -97.038, "2024-06-01", "2024-06-03")
    assert "latitude=32.897" in url
    assert "longitude=-97.038" in url
    assert "hourly=temperature_2m" in url
    assert "temperature_unit=fahrenheit" in url
    assert "timezone=UTC" in url


@pytest.mark.network
@pytest.mark.slow
def test_fetch_hourly_temp_smoke():
    df = fetch_hourly_temp(32.897, -97.038, "2024-06-01", "2024-06-03")
    assert not df.empty
    assert "valid" in df.columns
    assert "fcst_temp_f" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert isinstance(df["valid"].iloc[0], pd.Timestamp)


@pytest.mark.network
@pytest.mark.slow
def test_fetch_all_stations_smoke(tmp_path):
    cache_path = tmp_path / "test_openmeteo.parquet"
    df = fetch_all_stations(STATIONS, "2024-06-01", "2024-06-03", cache_path=str(cache_path))
    assert not df.empty
    assert "station" in df.columns
    assert "fcst_temp_f" in df.columns
    assert "valid" in df.columns
    assert os.path.exists(cache_path)
