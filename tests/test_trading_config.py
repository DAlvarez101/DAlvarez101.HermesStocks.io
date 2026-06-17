import os
import pytest
from dfw_temp_model.trading.config import TradingConfig, load_config


def test_load_config_requires_private_key(monkeypatch):
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
    monkeypatch.setenv("POLYMARKET_DRY_RUN", "true")
    with pytest.raises(ValueError, match="POLYMARKET_PRIVATE_KEY"):
        load_config()


def test_load_config_defaults_to_dry_run(monkeypatch):
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x" + "11" * 32)
    monkeypatch.setenv("POLYMARKET_DRY_RUN", "true")
    cfg = load_config()
    assert isinstance(cfg, TradingConfig)
    assert cfg.dry_run is True
    assert cfg.host == "https://clob.polymarket.com"
    assert cfg.chain_id == 137
    assert cfg.max_order_size_usdc == 50.0
    assert cfg.min_edge_bps == 100.0
    assert cfg.market_search_query == "Dallas high temperature"
    assert cfg.target_icao == "KDAL"
    assert cfg.settlement_icao == "KDFW"


def test_load_config_allows_live_mode(monkeypatch):
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0x" + "11" * 32)
    monkeypatch.setenv("POLYMARKET_DRY_RUN", "false")
    cfg = load_config()
    assert cfg.dry_run is False
