
from unittest.mock import MagicMock, patch
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.executor import execute_intent
from dfw_temp_model.trading.decision import OrderIntent


def test_execute_intent_dry_run_returns_record():
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
    client = MagicMock()
    intent = OrderIntent(side="BUY", token_id="Y", price=0.45, size=10.0, edge_bps=150.0, reason="edge")
    result = execute_intent(cfg, client, intent)
    assert result["dry_run"] is True
    client.post_limit_order.assert_not_called()
