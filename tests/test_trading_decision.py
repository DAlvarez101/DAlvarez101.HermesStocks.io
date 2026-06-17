
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.decision import OrderIntent, decide_trade


def _cfg(min_edge_bps: float = 100.0, dry_run: bool = True) -> TradingConfig:
    return TradingConfig(
        private_key="0x" + "11" * 32,
        host="https://clob.polymarket.com",
        chain_id=137,
        dry_run=dry_run,
        max_order_size_usdc=50.0,
        min_edge_bps=min_edge_bps,
        market_search_query="Dallas high temperature",
        target_icao="KDAL",
        settlement_icao="KDFW",
    )


def test_decide_trade_buy_yes_when_model_beats_market():
    cfg = _cfg(min_edge_bps=100.0)
    signal = {"probability_yes": 0.60, "probability_no": 0.40}
    market = {"yes_price": 0.45, "no_price": 0.55, "yes_token_id": "Y", "no_token_id": "N"}
    intent = decide_trade(cfg, signal, market, balance_usdc=100.0)
    assert isinstance(intent, OrderIntent)
    assert intent.side == "BUY"
    assert intent.token_id == "Y"
    assert intent.edge_bps == 1500.0


def test_decide_trade_buy_no_when_no_has_edge():
    cfg = _cfg(min_edge_bps=100.0)
    signal = {"probability_yes": 0.30, "probability_no": 0.70}
    market = {"yes_price": 0.45, "no_price": 0.55, "yes_token_id": "Y", "no_token_id": "N"}
    intent = decide_trade(cfg, signal, market, balance_usdc=100.0)
    assert intent is not None
    assert intent.token_id == "N"


def test_decide_trade_no_trade_when_edge_too_small():
    cfg = _cfg(min_edge_bps=500.0)
    signal = {"probability_yes": 0.60, "probability_no": 0.40}
    market = {"yes_price": 0.59, "no_price": 0.41, "yes_token_id": "Y", "no_token_id": "N"}
    intent = decide_trade(cfg, signal, market, balance_usdc=100.0)
    assert intent is None


def test_decide_trade_caps_size_by_balance():
    cfg = _cfg(min_edge_bps=100.0, dry_run=True)
    signal = {"probability_yes": 0.80, "probability_no": 0.20}
    market = {"yes_price": 0.50, "no_price": 0.50, "yes_token_id": "Y", "no_token_id": "N"}
    intent = decide_trade(cfg, signal, market, balance_usdc=10.0)
    # 50% of 10 = 5 / 0.5 = 10 shares
    assert intent.size == 10.0
