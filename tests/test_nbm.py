"""Tests for NBM 2-m temperature fetcher."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS, TARGET_ICAO
from dfw_temp_model.data.nbm import (
    _cog_url,
    _find_latest_cycle,
    _valid_dt,
    fetch_latest_nbm_2m_temp,
    fetch_nbm_forecast_range,
)


def test_cog_url_format():
    """COG URL follows the expected NBM S3 path pattern."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC").to_pydatetime()
    url = _cog_url(init, forecast_hour=1)
    assert url.startswith("https://noaa-nbm-pds.s3.amazonaws.com/")
    assert "blendv5.0/conus/2026/06/18/1800/temp/" in url
    assert "blendv5.0_conus_temp_2026-06-18T18:00_2026-06-18T19:00.tif" in url


def test_cog_url_forecast_hour_18():
    """Forecast hour 18 produces the correct valid time in the URL."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC").to_pydatetime()
    url = _cog_url(init, forecast_hour=18)
    assert "blendv5.0_conus_temp_2026-06-18T18:00_2026-06-19T12:00.tif" in url


def test_valid_dt_calculation():
    """valid_dt = init_dt + forecast_hour hours."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC")
    assert _valid_dt(init, 1) == pd.Timestamp("2026-06-18T19:00:00", tz="UTC")
    assert _valid_dt(init, 18) == pd.Timestamp("2026-06-19T12:00:00", tz="UTC")


def test_find_latest_cycle_returns_utc_timestamp():
    """_find_latest_cycle returns a tz-aware Timestamp and found=False when no cycle exists."""
    init, found = _find_latest_cycle(
        forecast_hour=1,
        lookback_hours=0,
        now=datetime(1990, 1, 1, tzinfo=timezone.utc),
    )
    assert isinstance(init, pd.Timestamp)
    assert init.tz is not None
    assert found is False


def test_fetch_latest_nbm_2m_temp_empty_when_no_cycle(mocker):
    """Returns empty DataFrame with correct columns when no cycle is found."""
    mocker.patch(
        "dfw_temp_model.data.nbm._find_latest_cycle",
        return_value=(pd.Timestamp("2026-06-18T18:00:00", tz="UTC"), False),
    )
    df = fetch_latest_nbm_2m_temp(STATIONS, forecast_hour=1, lookback_hours=0)
    assert df.empty
    assert list(df.columns) == [
        "station", "init_dt", "forecast_hour", "valid_dt", "lat", "lon", "tmpf",
    ]


def test_fetch_latest_nbm_2m_temp_with_mocked_pixel(mocker):
    """Returns one row per station with correct fields when a cycle is found."""
    init = pd.Timestamp("2026-06-18T18:00:00", tz="UTC")
    mocker.patch(
        "dfw_temp_model.data.nbm._find_latest_cycle",
        return_value=(init, True),
    )
    mocker.patch(
        "dfw_temp_model.data.nbm._read_temp_pixel",
        return_value=85.0,
    )
    df = fetch_latest_nbm_2m_temp(STATIONS[:1], forecast_hour=1, lookback_hours=0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["station"] == "KDFW"
    assert row["init_dt"] == init
    assert row["forecast_hour"] == 1
    assert row["valid_dt"] == pd.Timestamp("2026-06-18T19:00:00", tz="UTC")
    assert row["tmpf"] == 85.0


@pytest.mark.network
@pytest.mark.slow
def test_fetch_nbm_forecast_range_smoke():
    """Real network smoke test: fetch 3 NBM forecast hours for KDAL."""
    target = [s for s in STATIONS if s.icao == TARGET_ICAO][0]
    df = fetch_nbm_forecast_range(
        stations=[target],
        max_forecast_hour=3,
        lookback_hours=6,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "NBM fetch returned empty DataFrame (no cycle found)"
    assert TARGET_ICAO in df["station"].values
    assert len(df) == 3  # 3 forecast hours
    row = df.iloc[0]
    assert 20.0 <= row["tmpf"] <= 120.0
    assert pd.notna(row["init_dt"])
    assert row["forecast_hour"] == 1
    assert pd.notna(row["valid_dt"])
    assert row["valid_dt"] == row["init_dt"] + pd.Timedelta(hours=1)