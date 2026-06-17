
from unittest.mock import MagicMock, patch
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.client import PolymarketClient


@patch("dfw_temp_model.trading.client.ClobClient")
def test_client_in_dry_run_does_not_post(mock_clob):
    cfg = TradingConfig(
        private_key="0x" + "11" * 32,
        host="https://clob.polymarket.com",
        chain_id=137,
        dry_run=True,
        max_order_size_usdc=50.0,
        min_edge_bps=100.0,
        market_search_query="Dallas high temperature",
        target_icao="KDAL",
        settlement_icao="KDFW",
    )
    mock_client = MagicMock()
    mock_clob.return_value = mock_client
    mock_client.create_or_derive_api_key.return_value = {"apiKey": "k", "secret": "s", "passphrase": "p"}

    client = PolymarketClient(cfg)
    resp = client.post_limit_order("token_id", 0.55, 10.0, "BUY")
    assert resp["dry_run"] is True
    mock_client.create_and_post_limit_order.assert_not_called()
