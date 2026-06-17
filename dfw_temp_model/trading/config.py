"""Trading bot configuration loaded from environment variables."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TradingConfig:
    private_key: str
    host: str
    chain_id: int
    dry_run: bool
    max_order_size_usdc: float
    min_edge_bps: float
    market_search_query: str
    target_icao: str
    settlement_icao: str


def load_config() -> TradingConfig:
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if not private_key:
        raise ValueError("POLYMARKET_PRIVATE_KEY environment variable is required")

    dry_run_str = os.environ.get("POLYMARKET_DRY_RUN", "true").lower()
    dry_run = dry_run_str in {"1", "true", "yes"}

    return TradingConfig(
        private_key=private_key,
        host=os.environ.get("POLYMARKET_HOST", "https://clob.polymarket.com"),
        chain_id=int(os.environ.get("POLYMARKET_CHAIN_ID", "137")),
        dry_run=dry_run,
        max_order_size_usdc=float(os.environ.get("POLYMARKET_MAX_ORDER_SIZE_USDC", "50.0")),
        min_edge_bps=float(os.environ.get("POLYMARKET_MIN_EDGE_BPS", "100.0")),
        market_search_query=os.environ.get("POLYMARKET_MARKET_QUERY", "Dallas high temperature"),
        target_icao=os.environ.get("POLYMARKET_TARGET_ICAO", "KDAL"),
        settlement_icao=os.environ.get("POLYMARKET_SETTLEMENT_ICAO", "KDFW"),
    )
