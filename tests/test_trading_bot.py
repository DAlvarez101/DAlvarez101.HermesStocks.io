
from unittest.mock import MagicMock, patch
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.bot import run_bot


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


@patch("dfw_temp_model.trading.bot.resolve_market")
def test_run_bot_no_trade_when_market_closed(mock_resolve):
    cfg = _cfg()
    mock_resolve.return_value = MagicMock(
        question="Will Dallas hit 95F?",
        active=False,
        slug="closed-market",
        yes_price=0.5,
        no_price=0.5,
        yes_token_id="Y",
        no_token_id="N",
    )
    report = run_bot(cfg, "/tmp/nonexistent.db")
    assert report["status"] == "MARKET_CLOSED"


@patch("dfw_temp_model.trading.bot.resolve_market")
def test_run_bot_no_market(mock_resolve):
    cfg = _cfg()
    mock_resolve.return_value = None
    report = run_bot(cfg, "/tmp/nonexistent.db")
    assert report["status"] == "NO_MARKET"
