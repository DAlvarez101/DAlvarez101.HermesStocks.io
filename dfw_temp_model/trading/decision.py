"""Convert model signal + market price into an order intent."""
from dataclasses import dataclass
from typing import Optional

from dfw_temp_model.trading.config import TradingConfig


@dataclass(frozen=True)
class OrderIntent:
    side: str  # BUY or SELL
    token_id: str
    price: float
    size: float
    edge_bps: float
    reason: str


def _edge_bps(model_prob: float, market_price: float) -> float:
    """Edge in basis points. Positive means model thinks outcome is more likely."""
    return (model_prob - market_price) * 10000


def decide_trade(
    cfg: TradingConfig,
    signal: dict,
    market: dict,
    balance_usdc: float,
) -> Optional[OrderIntent]:
    """Return an order intent if the model edge exceeds the configured threshold."""
    p_yes = signal["probability_yes"]
    p_no = signal["probability_no"]
    yes_price = market["yes_price"]
    no_price = market["no_price"]

    yes_edge = _edge_bps(p_yes, yes_price)
    no_edge = _edge_bps(p_no, no_price)

    best_side = None
    best_edge = 0.0
    best_token = None
    best_price = None

    if yes_edge > no_edge and yes_edge >= cfg.min_edge_bps:
        best_side = "BUY"
        best_edge = yes_edge
        best_token = market["yes_token_id"]
        best_price = yes_price
    elif no_edge >= cfg.min_edge_bps:
        best_side = "BUY"
        best_edge = no_edge
        best_token = market["no_token_id"]
        best_price = no_price

    if best_side is None:
        return None

    # Simple sizing: risk a fixed max amount per trade.
    assert best_price is not None
    size = min(cfg.max_order_size_usdc, balance_usdc * 0.5) / best_price
    size = round(size, 2)
    if size <= 0:
        return None

    # Expected value sanity check: ignore trades where the payoff is tiny.
    # Buying NO at 0.98 only wins 2¢ per share; require at least 5¢ expected edge.
    expected_value_per_share = best_edge / 10000  # in dollars
    if expected_value_per_share < 0.05:
        return None

    assert best_token is not None
    return OrderIntent(
        side=best_side,
        token_id=best_token,
        price=best_price,
        size=size,
        edge_bps=round(best_edge, 1),
        reason=f"model_prob={p_yes:.2f} vs market={best_price:.2f}, edge={best_edge:.0f}bps",
    )
