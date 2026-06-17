import os

import pytest

from dfw_temp_model.config import STATIONS
from dfw_temp_model.data.iem_asos import build_iem_url, fetch_all_stations, fetch_asos_csv


def test_build_iem_url():
    url = build_iem_url("KDFW", "2022-01-01", "2022-01-02", ["tmpf", "drct", "sknt"])
    assert "station=KDFW" in url
    assert "data=tmpf" in url
    assert "year1=2022" in url


@pytest.mark.network
@pytest.mark.slow
def test_fetch_asos_csv_smoke():
    df = fetch_asos_csv("KDFW", "2024-06-01", "2024-06-03")
    assert not df.empty
    assert "tmpf" in df.columns
    assert "valid" in df.columns
    assert "station" in df.columns


@pytest.mark.network
@pytest.mark.slow
def test_fetch_all_stations(tmp_path):
    cache_path = tmp_path / "test_asos.parquet"
    df = fetch_all_stations("2024-06-01", "2024-06-03", STATIONS, cache_path=str(cache_path))
    assert not df.empty
    assert "station" in df.columns
    assert os.path.exists(cache_path)
