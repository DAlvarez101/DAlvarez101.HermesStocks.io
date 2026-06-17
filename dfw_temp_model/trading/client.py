"""Authenticated Polymarket CLOB client wrapper with dry-run support."""
import json
import os
from pathlib import Path
from typing import Any, Optional

from py_clob_client_v2 import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, AssetType, BalanceAllowanceParams, OrderArgsV2
from py_clob_client_v2.order_builder.constants import BUY, SELL

from dfw_temp_model.trading.config import TradingConfig


def _credentials_cache_path(host: str, address: str) -> Path:
    """Return a deterministic cache path for CLOB credentials."""
    safe_host = host.replace("://", "_").replace("/", "_")
    return Path.home() / ".cache" / "dfw-temp-model" / f"clob_creds_{safe_host}_{address}.json"


def _api_creds_to_dict(creds: ApiCreds) -> dict:
    return {
        "api_key": creds.api_key,
        "api_secret": creds.api_secret,
        "api_passphrase": creds.api_passphrase,
    }


def _dict_to_api_creds(data: dict) -> ApiCreds:
    return ApiCreds(
        api_key=data["api_key"],
        api_secret=data["api_secret"],
        api_passphrase=data["api_passphrase"],
    )


class PolymarketClient:
    """Thin wrapper around py-clob-client-v2 that adds dry-run and safety helpers."""

    def __init__(self, cfg: TradingConfig, credentials_path: Optional[Path] = None):
        self.cfg = cfg
        self._client = ClobClient(
            host=cfg.host,
            chain_id=cfg.chain_id,
            key=cfg.private_key,
        )
        # Derive wallet address for cache key (does not expose the private key).
        address = self._client.get_address()
        cache_path = credentials_path or _credentials_cache_path(cfg.host, address)

        creds = self._load_cached_credentials(cache_path)
        if creds is None:
            creds = self._client.create_or_derive_api_key()
            self._save_cached_credentials(cache_path, creds)

        # Re-init with credentials for L2 auth.
        self._client = ClobClient(
            host=cfg.host,
            chain_id=cfg.chain_id,
            key=cfg.private_key,
            creds=creds,
        )

    def _load_cached_credentials(self, path: Path) -> Optional[ApiCreds]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return _dict_to_api_creds(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _save_cached_credentials(self, path: Path, creds: ApiCreds) -> None:
        try:
            path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            path.write_text(json.dumps(_api_creds_to_dict(creds)))
            os.chmod(path, 0o600)
        except OSError:
            pass  # Non-fatal: next run will just re-derive.

    def get_address(self) -> str:
        return self._client.get_address()

    def get_balance(self) -> dict:
        """Return USDC and conditional token balances/allowances for this wallet."""
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)  # type: ignore[arg-type]
        return self._client.get_balance_allowance(params)  # type: ignore[no-any-return]

    def get_order_book(self, token_id: str, side: Optional[str] = None) -> dict:
        """Return order book for a token."""
        return self._client.get_order_book(token_id)  # type: ignore[no-any-return]

    def get_orders(self, **kwargs: Any) -> list:
        """Return open orders, optionally filtered."""
        return self._client.get_orders(**kwargs)  # type: ignore[no-any-return]

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
        order = self._client.create_order(
            order_args=OrderArgsV2(
                token_id=token_id,
                price=price,
                size=size,
                side=side_const,
            ),
        )
        return self._client.post_order(order)  # type: ignore[no-any-return]

    def cancel_all_orders(self) -> dict:
        """Cancel all open orders. Useful for risk-off."""
        return self._client.cancel_all_orders()  # type: ignore[no-any-return]
