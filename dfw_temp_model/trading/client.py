"""Authenticated Polymarket CLOB client wrapper with dry-run support."""
from typing import Any

from py_clob_client_v2 import ClobClient
from py_clob_client_v2.order_builder.constants import BUY, SELL

from dfw_temp_model.trading.config import TradingConfig


class PolymarketClient:
    """Thin wrapper around py-clob-client-v2 that adds dry-run and safety helpers."""

    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg
        self._client = ClobClient(
            host=cfg.host,
            chain_id=cfg.chain_id,
            key=cfg.private_key,
        )
        self._creds = self._client.create_or_derive_api_key()
        # Re-init with credentials for L2 auth.
        self._client = ClobClient(
            host=cfg.host,
            chain_id=cfg.chain_id,
            key=cfg.private_key,
            creds=self._creds,
        )

    def get_balance(self) -> dict:
        """Return available USDC balance and POL balance from CLOB."""
        return self._client.get_balance()  # type: ignore[no-any-return]

    def get_order_book(self, token_id: str) -> dict:
        return self._client.get_order_book(token_id)  # type: ignore[no-any-return]

    def post_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
    ) -> dict[str, Any]:
        """Post a GTC limit order, or return a dry-run record if dry_run=True."""
        if self.cfg.dry_run:
            return {
                "dry_run": True,
                "token_id": token_id,
                "price": price,
                "size": size,
                "side": side,
                "note": "order not submitted (dry run)",
            }

        side_const = BUY if side.upper() == "BUY" else SELL
        return self._client.create_and_post_limit_order(  # type: ignore[no-any-return]
            token_id=token_id,
            price=price,
            size=size,
            side=side_const,
        )
