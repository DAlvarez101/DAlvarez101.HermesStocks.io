"""Execute or simulate order intents."""
from typing import Optional

from dfw_temp_model.trading.client import PolymarketClient
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.decision import OrderIntent


def execute_intent(cfg: TradingConfig, client: Optional[PolymarketClient], intent: OrderIntent) -> dict:
    """Post the order or, in dry-run mode, return a record without posting."""
    if cfg.dry_run:
        return {
            "dry_run": True,
            "action": "WOULD_POST",
            "side": intent.side,
            "token_id": intent.token_id,
            "price": intent.price,
            "size": intent.size,
            "edge_bps": intent.edge_bps,
            "reason": intent.reason,
        }

    if client is None:
        raise RuntimeError("PolymarketClient is required for live order execution")

    return client.post_limit_order(
        token_id=intent.token_id,
        price=intent.price,
        size=intent.size,
        side=intent.side,
    )
