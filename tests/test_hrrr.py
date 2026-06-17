"""Tests for HRRR 2-m temperature fetcher."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS, TARGET_ICAO
from dfw_temp_model.data.hrrr import (
    _cfgrib_subset,
    _find_latest_cycle,
    _grib_url,
    _nearest_temp,
    _valid_dt,
    fetch_latest_hrrr_2m_temp,
)


def test_grib_url_format():
    init = pd.Timestamp("2026-06-17T18:00:00", tz="UTC").to_pydatetime()
    url = _grib_url(init, forecast_hour=1)
    assert url.startswith("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/")
    assert "hrrr.20260617/conus/hrrr.t18z.wrfsfcf01.grib2" in url


def test_valid_dt_calculation():
    init = pd.Timestamp("2026-06-17T18:00:00", tz="UTC")
    assert _valid_dt(init, 1) == pd.Timestamp("2026-06-17T19:00:00", tz="UTC")
    assert _valid_dt(init, 3) == pd.Timestamp("2026-06-17T21:00:00", tz="UTC")


def test_find_latest_cycle_returns_utc_timestamp():
    # We avoid a real network call here; just ensure the function returns the
    # expected tuple shape when it cannot find a cycle.
    init, found = _find_latest_cycle(
        forecast_hour=1,
        lookback_hours=0,
        now=datetime(1990, 1, 1, tzinfo=timezone.utc),
    )
    assert isinstance(init, pd.Timestamp)
    assert init.tz is not None
    assert found is False


def test_fetch_latest_hrrr_2m_temp_empty_when_no_cycle(mocker):
    mocker.patch("dfw_temp_model.data.hrrr._find_latest_cycle", return_value=(pd.Timestamp("2026-06-17T18:00:00", tz="UTC"), False))
    df = fetch_latest_hrrr_2m_temp(STATIONS, forecast_hour=1, lookback_hours=0)
    assert df.empty
    assert list(df.columns) == [
        "station", "init_dt", "forecast_hour", "valid_dt", "lat", "lon", "tmpf",
    ]


def test_fetch_latest_hrrr_2m_temp_with_mocked_subset(mocker):
    init = pd.Timestamp("2026-06-17T18:00:00", tz="UTC")
    fake_values = np.array([[300.0]])
    fake_var = mocker.MagicMock(values=fake_values)
    fake_ds = mocker.MagicMock()
    fake_ds.data_vars = ["t2m"]
    fake_ds.latitude.values = np.array([[32.0]])
    fake_ds.longitude.values = np.array([[-97.0]])
    fake_ds.__getitem__ = mocker.MagicMock(return_value=fake_var)

    mocker.patch("dfw_temp_model.data.hrrr._find_latest_cycle", return_value=(init, True))
    mocker.patch("dfw_temp_model.data.hrrr._cfgrib_subset", return_value=fake_ds)

    df = fetch_latest_hrrr_2m_temp(STATIONS[:1], forecast_hour=1, lookback_hours=0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["station"] == "KDFW"
    assert row["init_dt"] == init
    assert row["forecast_hour"] == 1
    assert row["valid_dt"] == pd.Timestamp("2026-06-17T19:00:00", tz="UTC")
    # 300 K = 80.33 F
    assert row["tmpf"] == pytest.approx(80.33, abs=0.01)


@pytest.mark.network
@pytest.mark.slow
def test_fetch_latest_hrrr_2m_temp_smoke():
    """Real network smoke test: fetch the latest HRRR 2 m temperature for KDFW."""
    df = fetch_latest_hrrr_2m_temp(
        stations=[s for s in STATIONS if s.icao == TARGET_ICAO],
        forecast_hour=1,
        lookback_hours=6,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "HRRR fetch returned empty DataFrame (no cycle found)"
    assert TARGET_ICAO in df["station"].values

    row = df[df["station"] == TARGET_ICAO].iloc[0]
    assert 20.0 <= row["tmpf"] <= 120.0
    assert pd.notna(row["init_dt"])
    assert row["forecast_hour"] == 1
    assert pd.notna(row["valid_dt"])
    assert pd.notna(row["lat"])
    assert pd.notna(row["lon"])
    assert row["valid_dt"] == row["init_dt"] + pd.Timedelta(hours=1)
