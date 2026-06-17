import pandas as pd
import pytest

from dfw_temp_model.config import STATIONS, TARGET_ICAO
from dfw_temp_model.data import build_dataset
from dfw_temp_model.data.iem_asos import fetch_all_stations as fetch_asos_all
from dfw_temp_model.data.openmeteo import fetch_all_stations as fetch_openmeteo_all


NEIGHBOR_ICAOS = [s.icao for s in STATIONS if s.icao != TARGET_ICAO]


def test_compute_daily_highs():
    df = pd.DataFrame(
        {
            "station": ["KDFW", "KDFW", "KDFW", "KADS"],
            "valid": pd.to_datetime(
                [
                    "2024-06-01 12:00:00+00:00",
                    "2024-06-01 18:00:00+00:00",
                    "2024-06-02 12:00:00+00:00",
                    "2024-06-01 14:00:00+00:00",
                ]
            ),
            "tmpf": [80.0, 85.0, 82.0, 83.0],
        }
    )
    out = build_dataset.compute_daily_highs(df)
    assert len(out) == 2
    assert out.loc["2024-06-01", "KDFW"] == 85.0
    assert out.loc["2024-06-01", "KADS"] == 83.0
    assert out.loc["2024-06-02", "KDFW"] == 82.0


def test_compute_forecast_daily_highs():
    df = pd.DataFrame(
        {
            "station": ["KDFW", "KDFW", "KDFW"],
            "valid": pd.to_datetime(
                [
                    "2024-06-01 06:00:00+00:00",
                    "2024-06-01 18:00:00+00:00",
                    "2024-06-02 12:00:00+00:00",
                ]
            ),
            "fcst_temp_f": [78.0, 86.0, 84.0],
        }
    )
    out = build_dataset.compute_forecast_daily_highs(df)
    assert len(out) == 2
    assert out.loc["2024-06-01", "KDFW"] == 86.0
    assert out.loc["2024-06-02", "KDFW"] == 84.0


def test_build_residual_table():
    obs = pd.DataFrame(
        {
            "KDFW": [95.0, 96.0],
            "KDAL": [94.0, 97.0],
        },
        index=pd.Index(["2024-06-01", "2024-06-02"], name="date"),
    )
    fcst = pd.DataFrame(
        {
            "KDFW": [90.0, 92.0],
            "KDAL": [89.0, 93.0],
        },
        index=pd.Index(["2024-06-01", "2024-06-02"], name="date"),
    )
    residuals = build_dataset.build_residual_table(obs, fcst, STATIONS)
    assert list(residuals.columns) == ["KDAL"]
    assert residuals.loc["2024-06-01", "KDAL"] == 5.0
    assert residuals.loc["2024-06-02", "KDAL"] == 4.0


def test_build_target_table():
    obs = pd.DataFrame(
        {
            "KDFW": [95.0, 96.0],
            "KDAL": [94.0, 97.0],
        },
        index=pd.Index(["2024-06-01", "2024-06-02"], name="date"),
    )
    fcst = pd.DataFrame(
        {
            "KDFW": [90.0, 92.0],
            "KDAL": [89.0, 93.0],
        },
        index=pd.Index(["2024-06-01", "2024-06-02"], name="date"),
    )
    residuals = build_dataset.build_residual_table(obs, fcst, STATIONS)
    target = build_dataset.build_target_table(obs, fcst, residuals)
    assert target.index.name == "date"
    assert "kdfw_obs" in target.columns
    assert "kdfw_fcst" in target.columns
    assert "residual_target" in target.columns
    assert "KDAL" in target.columns
    assert target.loc["2024-06-01", "residual_target"] == 5.0
    assert target.loc["2024-06-02", "residual_target"] == 4.0


@pytest.mark.network
@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    start, end = "2024-06-01", "2024-06-30"
    obs = fetch_asos_all(start, end, STATIONS)
    assert not obs.empty
    assert "tmpf" in obs.columns
    assert "station" in obs.columns

    fcst = fetch_openmeteo_all(STATIONS, start, end)
    assert not fcst.empty
    assert "fcst_temp_f" in fcst.columns
    assert "station" in fcst.columns

    obs_daily = build_dataset.compute_daily_highs(obs)
    fcst_daily = build_dataset.compute_forecast_daily_highs(fcst)

    residuals = build_dataset.build_residual_table(obs_daily, fcst_daily, STATIONS)
    target = build_dataset.build_target_table(obs_daily, fcst_daily, residuals)

    # At least 25 non-null days for target residual.
    assert target["residual_target"].notna().sum() >= 25
    # KDFW residual column (target) exists.
    assert "residual_target" in target.columns
    # All neighbor columns exist.
    for col in NEIGHBOR_ICAOS:
        assert col in target.columns
