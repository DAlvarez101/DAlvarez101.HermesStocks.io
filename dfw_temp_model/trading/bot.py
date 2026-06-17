"""Main bot orchestrator: fetch market, generate signal, decide, execute."""
from typing import Optional

from dfw_temp_model.trading.client import PolymarketClient
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.decision import decide_trade
from dfw_temp_model.trading.executor import execute_intent
from dfw_temp_model.trading.market import MarketMetadata, resolve_market
from dfw_temp_model.trading.signal import forecast_high_temp


def run_bot(cfg: TradingConfig, db_path: str) -> dict:
    """Run one trading iteration and return a report dict."""
    market = resolve_market(cfg.market_search_query)
    if market is None:
        return {"status": "NO_MARKET", "reason": f"No market found for: {cfg.market_search_query}"}

    if not market.active:
        return {"status": "MARKET_CLOSED", "market": market.slug}

    signal = forecast_high_temp(db_path, market.question, predicted_residual=None, model_std=2.0)

    # In dry run, we don't need a real balance; use max_order_size_usdc as balance.
    balance_usdc = cfg.max_order_size_usdc
    client = None
    if not cfg.dry_run:
        client = PolymarketClient(cfg)
        balance = client.get_balance()
        balance_usdc = float(balance.get("usable_balance", balance.get("balance", 0)))

    market_dict = {
        "yes_price": market.yes_price,
        "no_price": market.no_price,
        "yes_token_id": market.yes_token_id,
        "no_token_id": market.no_token_id,
    }
    intent = decide_trade(cfg, signal, market_dict, balance_usdc)

    if intent is None:
        return {
            "status": "NO_TRADE",
            "market": market.slug,
            "signal": signal,
        }

    result = execute_intent(cfg, client, intent)

    return {
        "status": "TRADE_EXECUTED" if not cfg.dry_run else "DRY_RUN_TRADE",
        "market": market.slug,
        "signal": signal,
        "intent": {
            "side": intent.side,
            "token_id": intent.token_id,
            "price": intent.price,
            "size": intent.size,
            "edge_bps": intent.edge_bps,
            "reason": intent.reason,
        },
        "result": result,
    }
