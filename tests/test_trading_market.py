import json
from unittest.mock import patch
from dfw_temp_model.trading.market import MarketMetadata, resolve_market


def fake_gamma_response():
    market = {
        "question": "Will Dallas-Fort Worth hit 95°F on June 18, 2026?",
        "conditionId": "0xabc123",
        "slug": "dallas-fort-worth-hit-95f-june-18-2026",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.35", "0.65"]),
        "clobTokenIds": json.dumps(["yes_token_123", "no_token_456"]),
        "volume": 120000.0,
        "closed": False,
        "endDate": "2026-06-18T23:59:59Z",
    }
    return {
        "events": [
            {
                "title": "Dallas Temperature June 18",
                "slug": "dallas-temperature-june-18",
                "volume": 120000.0,
                "markets": [market],
            }
        ],
        "pagination": {"totalResults": 1},
    }


@patch("dfw_temp_model.trading.market._gamma_get")
def test_resolve_market_parses_fields(mock_get):
    mock_get.return_value = fake_gamma_response()
    meta = resolve_market("Dallas high temperature")
    assert isinstance(meta, MarketMetadata)
    assert meta.condition_id == "0xabc123"
    assert meta.yes_token_id == "yes_token_123"
    assert meta.no_token_id == "no_token_456"
    assert meta.yes_price == 0.35
    assert meta.no_price == 0.65
    assert meta.active is True
    assert meta.volume_usdc == 120000.0


@patch("dfw_temp_model.trading.market._gamma_get")
def test_resolve_market_returns_none_when_no_events(mock_get):
    mock_get.return_value = {"events": [], "pagination": {"totalResults": 0}}
    assert resolve_market("nowhere market") is None


@patch("dfw_temp_model.trading.market._gamma_get")
def test_resolve_market_prefers_active(mock_get):
    active_market = {
        "question": "Active",
        "conditionId": "0x1",
        "slug": "active",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.5", "0.5"]),
        "clobTokenIds": json.dumps(["a", "b"]),
        "volume": 100.0,
        "closed": False,
    }
    closed_market = {
        "question": "Closed",
        "conditionId": "0x2",
        "slug": "closed",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.5", "0.5"]),
        "clobTokenIds": json.dumps(["c", "d"]),
        "volume": 999999.0,
        "closed": True,
    }
    mock_get.return_value = {
        "events": [
            {"title": "evt", "volume": 999999.0, "markets": [closed_market, active_market]}
        ],
        "pagination": {"totalResults": 1},
    }
    meta = resolve_market("Dallas high temperature")
    assert meta.slug == "active"
    assert meta.active is True
