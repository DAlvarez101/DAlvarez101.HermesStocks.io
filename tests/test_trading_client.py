
from pathlib import Path
from unittest.mock import MagicMock, patch
from py_clob_client_v2.clob_types import ApiCreds
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.client import PolymarketClient


def _cfg():
    return TradingConfig(
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


@patch("dfw_temp_model.trading.client.ClobClient")
def test_client_in_dry_run_does_not_post(mock_clob):
    cfg = _cfg()
    mock_client = MagicMock()
    mock_clob.return_value = mock_client
    mock_client.get_address.return_value = "0x" + "22" * 20
    mock_client.create_or_derive_api_key.return_value = ApiCreds(
        api_key="k", api_secret="s", api_passphrase="p"
    )

    client = PolymarketClient(cfg, credentials_path=Path("/tmp/test_clob_creds.json"))
    resp = client.post_limit_order("token_id", 0.55, 10.0, "BUY")
    assert resp["dry_run"] is True
    mock_client.create_and_post_limit_order.assert_not_called()


@patch("dfw_temp_model.trading.client.ClobClient")
def test_client_caches_credentials(mock_clob, tmp_path):
    cfg = _cfg()
    mock_client = MagicMock()
    mock_clob.return_value = mock_client
    mock_client.get_address.return_value = "0x" + "22" * 20
    mock_client.create_or_derive_api_key.return_value = ApiCreds(
        api_key="k", api_secret="s", api_passphrase="p"
    )

    cache = tmp_path / "creds.json"

    # First instantiation derives and caches.
    c1 = PolymarketClient(cfg, credentials_path=cache)
    assert c1.get_address() == "0x" + "22" * 20
    assert cache.exists()
    mock_client.create_or_derive_api_key.assert_called_once()

    # Reset mock to verify second instantiation loads from cache.
    mock_client.create_or_derive_api_key.reset_mock()
    c2 = PolymarketClient(cfg, credentials_path=cache)
    assert c2.get_address() == "0x" + "22" * 20
    mock_client.create_or_derive_api_key.assert_not_called()
