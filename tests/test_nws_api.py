"""Tests for the NWS API observation fetcher."""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from dfw_temp_model.data.nws_api import parse_nws_observations, fetch_nws_observations


def _sample_nws_payload():
    """Minimal NWS API GeoJSON payload for testing."""
    return {
        "features": [
            {
                "id": "https://api.weather.gov/stations/KDAL/observations/2026-06-18T17:45:00+00:00",
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-96.85, 32.85]},
                "properties": {
                    "timestamp": "2026-06-18T17:45:00+00:00",
                    "stationId": "KDAL",
                    "temperature": {"value": 32.0, "unitCode": "wmoUnit:degC"},
                    "dewpoint": {"value": 24.0, "unitCode": "wmoUnit:degC"},
                    "windDirection": {"value": 130.0, "unitCode": "wmoUnit:degree_(angle)"},
                    "windSpeed": {"value": 18.504, "unitCode": "wmoUnit:km_h-1"},
                    "windGust": {"value": None, "unitCode": "wmoUnit:km_h-1"},
                    "barometricPressure": {"value": 100575.74, "unitCode": "wmoUnit:Pa"},
                    "relativeHumidity": {"value": 62.7, "unitCode": "wmoUnit:percent"},
                    "visibility": {"value": 16093.44, "unitCode": "wmoUnit:m"},
                },
            },
            {
                "id": "https://api.weather.gov/stations/KDAL/observations/2026-06-18T17:40:00+00:00",
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-96.85, 32.85]},
                "properties": {
                    "timestamp": "2026-06-18T17:40:00+00:00",
                    "stationId": "KDAL",
                    "temperature": {"value": 31.8, "unitCode": "wmoUnit:degC"},
                    "dewpoint": {"value": 24.0, "unitCode": "wmoUnit:degC"},
                    "windDirection": {"value": 130.0, "unitCode": "wmoUnit:degree_(angle)"},
                    "windSpeed": {"value": 18.504, "unitCode": "wmoUnit:km_h-1"},
                    "windGust": {"value": None, "unitCode": "wmoUnit:km_h-1"},
                    "barometricPressure": {"value": 100575.74, "unitCode": "wmoUnit:Pa"},
                    "relativeHumidity": {"value": 62.7, "unitCode": "wmoUnit:percent"},
                    "visibility": {"value": 16093.44, "unitCode": "wmoUnit:m"},
                },
            },
        ]
    }


def test_parse_nws_observations_basic():
    """Parse a minimal NWS API payload into the project schema."""
    payload = _sample_nws_payload()
    df = parse_nws_observations(payload)
    assert len(df) == 2
    assert "station" in df.columns
    assert "valid" in df.columns
    assert "tmpf" in df.columns
    assert df.iloc[0]["station"] == "KDAL"
    # 32.0 C = 89.6 F
    assert df.iloc[0]["tmpf"] == pytest.approx(89.6, abs=0.1)
    # 31.8 C = 89.24 F
    assert df.iloc[1]["tmpf"] == pytest.approx(89.24, abs=0.1)


def test_parse_nws_observations_null_temperature():
    """Null temperature values become None in the DataFrame."""
    payload = {
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-96.85, 32.85]},
                "properties": {
                    "timestamp": "2026-06-18T17:45:00+00:00",
                    "stationId": "KDAL",
                    "temperature": {"value": None, "unitCode": "wmoUnit:degC"},
                    "dewpoint": {"value": None, "unitCode": "wmoUnit:degC"},
                    "windDirection": {"value": None, "unitCode": "wmoUnit:degree_(angle)"},
                    "windSpeed": {"value": None, "unitCode": "wmoUnit:km_h-1"},
                    "barometricPressure": {"value": None, "unitCode": "wmoUnit:Pa"},
                    "relativeHumidity": {"value": None, "unitCode": "wmoUnit:percent"},
                    "visibility": {"value": None, "unitCode": "wmoUnit:m"},
                },
            },
        ]
    }
    df = parse_nws_observations(payload)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["tmpf"])


def test_parse_nws_observations_empty():
    """Empty payload returns empty DataFrame with correct columns."""
    df = parse_nws_observations({"features": []})
    assert df.empty
    assert "station" in df.columns
    assert "tmpf" in df.columns


def test_fetch_nws_observations_mocked():
    """Fetch with a mocked HTTP response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _sample_nws_payload()
    mock_response.raise_for_status = MagicMock()

    with patch("dfw_temp_model.data.nws_api.requests.get", return_value=mock_response):
        df = fetch_nws_observations("KDAL", limit=2)
    assert len(df) == 2
    assert df.iloc[0]["station"] == "KDAL"
    assert df.iloc[0]["tmpf"] == pytest.approx(89.6, abs=0.1)