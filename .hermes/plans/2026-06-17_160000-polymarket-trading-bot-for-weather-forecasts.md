# Polymarket Weather-Forecast Trading Bot Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Build a self-contained Polymarket trading bot module that consumes the existing DFW temperature forecast pipeline, identifies mispriced weather markets, and places trades — all without modifying or disrupting the live dashboard ingestion/cron flow.

**Architecture:** Add a new `dfw_temp_model/trading/` package that sits beside (not inside) the existing data pipeline. It reuses the SQLite forecast/observation DB and the advection model output, adds a Polymarket CLOB trading client wrapper, a signal generator, a decision engine, and a separate cron entry. The existing `cron_update_dashboard.sh` continues to run unchanged.

**Tech Stack:** Python 3.13, existing project venv, `py-clob-client-v2`, `web3.py`/`eth-account` (fallback), `pytest`, project SQLite store.

---

## Current Context / Assumptions

- Existing pipeline: `scripts/ingest_live_metars.py --hrrr` → `scripts/generate_dashboard.py` → `cron_update_dashboard.sh` (hourly).
- Forecast DB: `/opt/data/stock-research/dfw_temp_model/data/cache/db/weather_observations.db`.
- Target station for model: `KDAL` (Dallas Love Field). Polymarket DFW high-temp markets typically settle against `KDFW` (Dallas/Fort Worth International); this plan explicitly handles that mapping and documents the mismatch.
- Read-only Polymarket helper exists in the Hermes skill at `/opt/hermes/skills/research/polymarket/scripts/polymarket.py`. We will not modify it; we will vendor the relevant Gamma/CLOB parsing logic into the trading package.
- Trading requires a Polygon wallet with USDC.e and POL for gas. The plan assumes credentials are supplied via env vars; no keys are hard-coded.
- **Geographic/legal note:** The implementer must confirm they are in a jurisdiction where Polymarket trading is permitted. The bot starts in **dry-run mode** by default.

---

## Proposed Approach

1. **Create the trading package skeleton** under `dfw_temp_model/trading/`.
2. **Add the Polymarket SDK dependency** and verify it imports.
3. **Build a read-only market resolver** that maps a forecast target (KDAL daily high) to the correct Polymarket market by slug/conditionId.
4. **Build an authenticated CLOB client wrapper** with dry-run support, balance checks, and allowance checks.
5. **Build a signal generator** that loads the latest corrected forecast and produces a probability distribution over the market's temperature threshold.
6. **Build a decision engine** that compares model probability to market price, accounts for spread/fees, and emits an order intent.
7. **Build an executor** that converts intents into signed/limit orders only when dry-run is disabled and risk checks pass.
8. **Add a standalone entry script** `scripts/run_polymarket_bot.py` and a separate cron wrapper `scripts/cron_polymarket_bot.sh`.
9. **Write tests** with mocked CLOB responses and mocked wallet interactions.
10. **Smoke-test end-to-end** in dry-run mode against live market data.

---

## Step-by-Step Plan

### Task 1: Create the trading package skeleton

**Objective:** Add `dfw_temp_model/trading/` with empty modules and update `pyproject.toml` package discovery.

**Files:**
- Create: `dfw_temp_model/trading/__init__.py`
- Create: `dfw_temp_model/trading/config.py`
- Create: `dfw_temp_model/trading/market.py`
- Create: `dfw_temp_model/trading/client.py`
- Create: `dfw_temp_model/trading/signal.py`
- Create: `dfw_temp_model/trading/decision.py`
- Create: `dfw_temp_model/trading/executor.py`
- Create: `dfw_temp_model/trading/bot.py`
- Modify: `pyproject.toml`

**Step 1: Add modules**

Create `dfw_temp_model/trading/__init__.py`:
```python
"""Polymarket weather-trading bot package."""
```

Create the other module files with docstring-only placeholders, e.g. `dfw_temp_model/trading/config.py`:
```python
"""Trading bot configuration."""
```

**Step 2: Update package discovery**

Modify `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
include = ["dfw_temp_model*"]
exclude = ["data*", "notebooks*", "scripts*", "tests*"]
```
→ no change needed; `dfw_temp_model*` already matches.

**Step 3: Verify import**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -c "from dfw_temp_model import trading; print(trading.__file__)"`
Expected: prints path to `dfw_temp_model/trading/__init__.py`

**Step 4: Commit**

```bash
git add dfw_temp_model/trading/ pyproject.toml
git commit -m "chore: create trading package skeleton"
```

---

### Task 2: Install and verify the Polymarket SDK

**Objective:** Add `py-clob-client-v2` to project dependencies and confirm it imports inside the project venv.

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (via `uv lock`)

**Step 1: Add dependency**

Modify `pyproject.toml` dependencies list to append:
```toml
"py-clob-client-v2>=1.0.1",
"web3>=7.0",
```

**Step 2: Sync venv**

Run: `cd /opt/data/stock-research/dfw_temp_model && uv sync`
Expected: installs new packages without errors.

**Step 3: Verify import**

Run: `.venv/bin/python -c "from py_clob_client_v2 import ClobClient, OrderArgs; print('py-clob-client-v2 OK')"`
Expected: `py-clob-client-v2 OK`

Run: `.venv/bin/python -c "from web3 import Web3; print('web3 OK')"`
Expected: `web3 OK`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add py-clob-client-v2 and web3"
```

---

### Task 3: Build config loader for trading credentials

**Objective:** Centralize env-var loading with validation; never hard-code secrets.

**Files:**
- Create: `dfw_temp_model/trading/config.py`
- Create: `tests/test_trading_config.py`

**Step 1: Write failing test**

`tests/test_trading_config.py`:
```python
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
    assert cfg.dry_run is True
    assert cfg.host == "https://clob.polymarket.com"
    assert cfg.chain_id == 137
```

**Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_trading_config.py -v`
Expected: FAIL — `module not found` or `load_config not defined`

**Step 3: Implement config loader**

`dfw_temp_model/trading/config.py`:
```python
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
```

**Step 4: Run test to verify pass**

Run: `.venv/bin/pytest tests/test_trading_config.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add dfw_temp_model/trading/config.py tests/test_trading_config.py
git commit -m "feat: add trading config loader with dry-run default"
```

---

### Task 4: Build read-only Polymarket market resolver

**Objective:** Discover the active DFW high-temperature market, parse token IDs, and expose market metadata without touching any wallet code.

**Files:**
- Create: `dfw_temp_model/trading/market.py`
- Create: `tests/test_trading_market.py`

**Step 1: Write failing test**

`tests/test_trading_market.py`:
```python
import json
from unittest.mock import patch, MagicMock
import pandas as pd
from dfw_temp_model.trading.market import MarketMetadata, resolve_market


def fake_gamma_response():
    market = {
        "question": "Will Dallas-Fort Worth hit 95°F on June 18, 2026?",
        "conditionId": "0xabc123",
        "slug": "dallas-fort-worth-hit-95f-june-18-2026",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.35", "0.65"]),
        "clobTokenIds": json.dumps(["yes_token_123", "no_token_456"]),
        "volume": 120000.0,
        "closed": False,
        "endDate": "2026-06-18T23:59:59Z",
    }
    return {
        "events": [
            {
                "title": "Dallas Temperature June 18",
                "slug": "dallas-temperature-june-18",
                "markets": [market],
            }
        ],
        "pagination": {"totalResults": 1},
    }


@patch("dfw_temp_model.trading.market._gamma_get")
def test_resolve_market_parses_fields(mock_get):
    mock_get.return_value = fake_gamma_response()
    meta = resolve_market("Dallas high temperature")
    assert isinstance(meta, MarketMetadata)
    assert meta.condition_id == "0xabc123"
    assert meta.yes_token_id == "yes_token_123"
    assert meta.no_token_id == "no_token_456"
    assert meta.yes_price == 0.35
    assert meta.active is True
    assert meta.volume_usdc == 120000.0
```

**Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_trading_market.py -v`
Expected: FAIL — `module not found` or similar

**Step 3: Implement market resolver**

`dfw_temp_model/trading/market.py`:
```python
"""Read-only Polymarket market discovery and metadata."""
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

GAMMA = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class MarketMetadata:
    question: str
    condition_id: str
    slug: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    volume_usdc: float
    active: bool
    end_date: Optional[str]


def _gamma_get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "dfw-temp-bot/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _parse_json_field(value) -> Optional[list]:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value if isinstance(value, list) else None


def resolve_market(query: str, active: bool = True, closed: bool = False) -> Optional[MarketMetadata]:
    """Search Gamma for the most relevant market matching *query*."""
    q = urllib.parse.quote(query)
    data = _gamma_get(f"{GAMMA}/public-search?q={q}")
    events = data.get("events", [])
    if not events:
        return None

    # Pick the event with the highest volume.
    events_sorted = sorted(events, key=lambda e: float(e.get("volume", 0) or 0), reverse=True)
    candidate_markets = []
    for evt in events_sorted:
        for m in evt.get("markets", []):
            candidate_markets.append(m)

    if not candidate_markets:
        return None

    # Prefer active, high-volume markets.
    candidate_markets = sorted(
        candidate_markets,
        key=lambda m: (not m.get("closed", False), float(m.get("volume", 0) or 0)),
        reverse=True,
    )

    m = candidate_markets[0]
    prices = _parse_json_field(m.get("outcomePrices"))
    tokens = _parse_json_field(m.get("clobTokenIds"))
    if not prices or not tokens or len(prices) < 2 or len(tokens) < 2:
        return None

    return MarketMetadata(
        question=m.get("question", ""),
        condition_id=m.get("conditionId", ""),
        slug=m.get("slug", ""),
        yes_token_id=tokens[0],
        no_token_id=tokens[1],
        yes_price=float(prices[0]),
        no_price=float(prices[1]),
        volume_usdc=float(m.get("volume", 0) or 0),
        active=not m.get("closed", False),
        end_date=m.get("endDate"),
    )
```

**Step 4: Run test to verify pass**

Run: `.venv/bin/pytest tests/test_trading_market.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add dfw_temp_model/trading/market.py tests/test_trading_market.py
git commit -m "feat: add read-only Polymarket market resolver"
```

---

### Task 5: Build authenticated CLOB client wrapper

**Objective:** Encapsulate wallet auth, balance/allowance checks, order creation, and heartbeat. Expose a `dry_run` flag that prints instead of posting.

**Files:**
- Create: `dfw_temp_model/trading/client.py`
- Create: `tests/test_trading_client.py`

**Step 1: Write failing test**

`tests/test_trading_client.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from dfw_temp_model.trading.client import PolymarketClient
from dfw_temp_model.trading.config import TradingConfig


@pytest.fixture
def cfg():
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


@patch("dfw_temp_model.trading.client.ClobClient")
def test_client_in_dry_run_does_not_post(mock_clob, cfg):
    mock_client = MagicMock()
    mock_clob.return_value = mock_client
    mock_client.create_or_derive_api_key.return_value = {"apiKey": "k", "secret": "s", "passphrase": "p"}

    client = PolymarketClient(cfg)
    resp = client.post_limit_order("token_id", 0.55, 10.0, "BUY")
    assert resp["dry_run"] is True
    mock_client.create_and_post_order.assert_not_called()
```

**Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_trading_client.py -v`
Expected: FAIL

**Step 3: Implement client wrapper**

`dfw_temp_model/trading/client.py`:
```python
"""Authenticated Polymarket CLOB client wrapper with dry-run support."""
from typing import Any, Optional

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
        return self._client.get_balance()

    def get_order_book(self, token_id: str) -> dict:
        return self._client.get_order_book(token_id)

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
        return self._client.create_and_post_limit_order(
            token_id=token_id,
            price=price,
            size=size,
            side=side_const,
        )
```

**Step 4: Run test to verify pass**

Run: `.venv/bin/pytest tests/test_trading_client.py -v`
Expected: 1 passed

**Step 5: Commit**

```bash
git add dfw_temp_model/trading/client.py tests/test_trading_client.py
git commit -m "feat: add authenticated Polymarket client wrapper"
```

---

### Task 6: Build signal generator from corrected forecast

**Objective:** Load latest HRRR/METAR data from the existing SQLite DB, apply the advection model (if wind data available) or IDW baseline, and produce a forecasted high temperature and probability that it exceeds the market threshold.

**Files:**
- Create: `dfw_temp_model/trading/signal.py`
- Create: `tests/test_trading_signal.py`
- Modify: `dfw_temp_model/models/advection.py` (add a convenience one-day forecast helper if not present)

**Step 1: Write failing test**

`tests/test_trading_signal.py`:
```python
import sqlite3
import pandas as pd
from dfw_temp_model.trading.signal import forecast_high_temp, probability_above_threshold


def test_probability_above_threshold_with_gaussian():
    # mean=90, std=2, threshold=95 -> ~0.0062
    p = probability_above_threshold(90.0, 2.0, 95.0)
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_forecast_high_temp_returns_value():
    # Smoke test: even without a real DB we can test the helper.
    result = forecast_high_temp(
        latest_observed=92.0,
        hrrr_raw_high=94.0,
        predicted_residual=1.5,
        model_std=2.0,
    )
    assert result["corrected_high"] == 95.5
    assert result["model_std"] == 2.0
```

**Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_trading_signal.py -v`
Expected: FAIL

**Step 3: Implement signal generator**

`dfw_temp_model/trading/signal.py`:
```python
"""Generate a corrected temperature-forecast signal from the live DB."""
import math
from typing import Optional

import pandas as pd
from scipy.stats import norm

from dfw_temp_model.config import TARGET_ICAO
from dfw_temp_model.storage.obs_db import get_db, hrrr_forecast_for_cycle, latest_complete_hrrr_cycle
from dfw_temp_model.storage.obs_db import latest_by_station


def probability_above_threshold(mean_temp: float, std: float, threshold: float) -> float:
    """Probability that the true high exceeds *threshold* under a Gaussian model."""
    if std <= 0:
        return 1.0 if mean_temp > threshold else 0.0
    return float(1.0 - norm.cdf(threshold, loc=mean_temp, scale=std))


def _extract_market_threshold(question: str) -> Optional[float]:
    """Naive threshold extractor: find the first temperature-like number in °F."""
    import re
    match = re.search(r"(\d+)\s*°?F", question, re.IGNORECASE)
    return float(match.group(1)) if match else None


def forecast_high_temp(
    db_path: str,
    market_question: str,
    predicted_residual: Optional[float] = None,
    model_std: float = 2.0,
) -> dict:
    """Return corrected forecast for the target station and market threshold."""
    conn = get_db(db_path)
    latest = latest_by_station(conn)
    target_row = latest[latest["station"] == TARGET_ICAO]
    latest_observed = float(target_row.iloc[0]["tmpf"]) if not target_row.empty else None

    init_dt = latest_complete_hrrr_cycle(conn, TARGET_ICAO, required_hours=18)
    hrrr_raw_high = None
    if init_dt:
        df = hrrr_forecast_for_cycle(conn, TARGET_ICAO, init_dt)
        if not df.empty:
            df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
            today = pd.Timestamp.utcnow().floor("d")
            today_rows = df[df["valid_dt"].dt.floor("d") == today]
            if not today_rows.empty:
                hrrr_raw_high = float(today_rows["tmpf"].max())

    conn.close()

    if hrrr_raw_high is None:
        raise ValueError("No HRRR forecast available for today")

    corrected_high = hrrr_raw_high + (predicted_residual or 0.0)
    threshold = _extract_market_threshold(market_question)
    if threshold is None:
        raise ValueError(f"Could not extract temperature threshold from: {market_question}")

    prob_yes = probability_above_threshold(corrected_high, model_std, threshold)
    return {
        "corrected_high": round(corrected_high, 2),
        "hrrr_raw_high": round(hrrr_raw_high, 2),
        "predicted_residual": predicted_residual,
        "model_std": model_std,
        "threshold": threshold,
        "probability_yes": round(prob_yes, 4),
        "probability_no": round(1.0 - prob_yes, 4),
        "latest_observed": latest_observed,
    }


def forecast_high_temp_simple(
    latest_observed: Optional[float],
    hrrr_raw_high: float,
    predicted_residual: Optional[float],
    model_std: float,
) -> dict:
    """Pure helper for tests and offline use."""
    corrected_high = hrrr_raw_high + (predicted_residual or 0.0)
    return {
        "corrected_high": corrected_high,
        "hrrr_raw_high": hrrr_raw_high,
        "predicted_residual": predicted_residual,
        "model_std": model_std,
    }
```

**Step 4: Add scipy dependency**

Modify `pyproject.toml` to add `"scipy>=1.11"` if not already present. Run `uv sync`.

**Step 5: Run test to verify pass**

Run: `.venv/bin/pytest tests/test_trading_signal.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add dfw_temp_model/trading/signal.py tests/test_trading_signal.py pyproject.toml uv.lock
git commit -m "feat: add forecast signal generator"
```

---

### Task 7: Build decision engine

**Objective:** Compare model probability to market price and emit an order intent only when edge > min_edge_bps and risk limits allow.

**Files:**
- Create: `dfw_temp_model/trading/decision.py`
- Create: `tests/test_trading_decision.py`

**Step 1: Write failing test**

`tests/test_trading_decision.py`:
```python
from dfw_temp_model.trading.decision import OrderIntent, decide_trade
from dfw_temp_model.trading.config import TradingConfig


def test_decide_trade_buy_when_model_beats_market():
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
    signal = {"probability_yes": 0.60, "probability_no": 0.40, "threshold": 95.0}
    market = {"yes_price": 0.45, "no_price": 0.55, "yes_token_id": "Y", "no_token_id": "N"}
    intent = decide_trade(cfg, signal, market, balance_usdc=100.0)
    assert isinstance(intent, OrderIntent)
    assert intent.side == "BUY"
    assert intent.token_id == "Y"


def test_decide_trade_no_trade_when_edge_too_small():
    cfg = TradingConfig(
        private_key="0x" + "11" * 32,
        host="https://clob.polymarket.com",
        chain_id=137,
        dry_run=True,
        max_order_size_usdc=50.0,
        min_edge_bps=500.0,
        market_search_query="Dallas high temperature",
        target_icao="KDAL",
        settlement_icao="KDFW",
    )
    signal = {"probability_yes": 0.60, "probability_no": 0.40, "threshold": 95.0}
    market = {"yes_price": 0.59, "no_price": 0.41, "yes_token_id": "Y", "no_token_id": "N"}
    intent = decide_trade(cfg, signal, market, balance_usdc=100.0)
    assert intent is None
```

**Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_trading_decision.py -v`
Expected: FAIL

**Step 3: Implement decision engine**

`dfw_temp_model/trading/decision.py`:
```python
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
    size = min(cfg.max_order_size_usdc, balance_usdc * 0.5) / best_price
    size = round(size, 2)
    if size <= 0:
        return None

    return OrderIntent(
        side=best_side,
        token_id=best_token,
        price=best_price,
        size=size,
        edge_bps=round(best_edge, 1),
        reason=f"model_prob={p_yes:.2f} vs market={best_price:.2f}, edge={best_edge:.0f}bps",
    )
```

**Step 4: Run test to verify pass**

Run: `.venv/bin/pytest tests/test_trading_decision.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add dfw_temp_model/trading/decision.py tests/test_trading_decision.py
git commit -m "feat: add trading decision engine"
```

---

### Task 8: Build executor with risk checks

**Objective:** Tie client, signal, decision, and market together. Log every action; only post orders when dry_run=False.

**Files:**
- Create: `dfw_temp_model/trading/executor.py`
- Create: `dfw_temp_model/trading/bot.py`
- Create: `tests/test_trading_executor.py`

**Step 1: Write failing test**

`tests/test_trading_executor.py`:
```python
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
```

**Step 2: Run test to verify failure**

Run: `.venv/bin/pytest tests/test_trading_executor.py -v`
Expected: FAIL

**Step 3: Implement executor**

`dfw_temp_model/trading/executor.py`:
```python
"""Execute or simulate order intents."""
from dfw_temp_model.trading.client import PolymarketClient
from dfw_temp_model.trading.config import TradingConfig
from dfw_temp_model.trading.decision import OrderIntent


def execute_intent(cfg: TradingConfig, client: PolymarketClient, intent: OrderIntent) -> dict:
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

    return client.post_limit_order(
        token_id=intent.token_id,
        price=intent.price,
        size=intent.size,
        side=intent.side,
    )
```

`dfw_temp_model/trading/bot.py`:
```python
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
    if not cfg.dry_run:
        client = PolymarketClient(cfg)
        balance = client.get_balance()
        balance_usdc = float(balance.get("usable_balance", balance.get("balance", 0)))
    else:
        client = None

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

    if cfg.dry_run:
        result = execute_intent(cfg, client, intent)
    else:
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
```

**Step 4: Run test to verify pass**

Run: `.venv/bin/pytest tests/test_trading_executor.py tests/test_trading_bot.py -v`
(Note: create `tests/test_trading_bot.py` only if you add bot tests.)

Expected: 1 passed for executor.

**Step 5: Commit**

```bash
git add dfw_temp_model/trading/executor.py dfw_temp_model/trading/bot.py tests/test_trading_executor.py
git commit -m "feat: add trading executor and bot orchestrator"
```

---

### Task 9: Add standalone entry script

**Objective:** Provide `scripts/run_polymarket_bot.py` that loads config and runs `run_bot()` once. Do not modify existing scripts.

**Files:**
- Create: `scripts/run_polymarket_bot.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Run the Polymarket weather-trading bot once."""
import argparse
import json
import sys
from pathlib import Path

from dfw_temp_model.config import CACHE_DIR
from dfw_temp_model.trading.config import load_config
from dfw_temp_model.trading.bot import run_bot


def main():
    parser = argparse.ArgumentParser(description="Polymarket weather-trading bot")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to the existing weather SQLite DB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Override POLYMARKET_DRY_RUN to true",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.dry_run:
        # Force dry run regardless of env.
        from dataclasses import replace
        cfg = replace(cfg, dry_run=True)

    if not cfg.dry_run:
        print("WARNING: running in LIVE trading mode", file=sys.stderr)
        print("Set POLYMARKET_DRY_RUN=true or use --dry-run to simulate.", file=sys.stderr)

    report = run_bot(cfg, args.db)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["status"].startswith(("DRY_RUN", "NO_TRADE", "NO_MARKET")) else 0)


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs in dry-run mode**

Run:
```bash
cd /opt/data/stock-research/dfw_temp_model
POLYMARKET_PRIVATE_KEY=0x$(openssl rand -hex 32) POLYMARKET_DRY_RUN=true .venv/bin/python scripts/run_polymarket_bot.py --dry-run
```
Expected: JSON report printed; status likely `NO_MARKET` if no Dallas market matches, or `DRY_RUN_TRADE`/`NO_TRADE`.

**Step 3: Commit**

```bash
git add scripts/run_polymarket_bot.py
git commit -m "feat: add standalone Polymarket bot entry script"
```

---

### Task 10: Add separate cron wrapper

**Objective:** Schedule the bot independently of the dashboard cron, so existing data ingestion is never blocked.

**Files:**
- Create: `scripts/cron_polymarket_bot.sh`

**Step 1: Write the cron wrapper**

```bash
#!/bin/bash
# Run the Polymarket weather-trading bot on its own schedule.
# This script is separate from cron_update_dashboard.sh so that trading failures
# never break the live dashboard ingestion pipeline.
set -euo pipefail

export HERMES_HOME=/opt/data
export HERMES_DOCKER_EXEC_AS_ROOT=1

PROJECT_DIR="/opt/data/stock-research/dfw_temp_model"
DB_PATH="${PROJECT_DIR}/data/cache/db/weather_observations.db"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
LOG_FILE="${LOG_DIR}/polymarket_bot_$(date -u +%Y%m%d_%H%M%S).log"

# Always run dry-run from cron by default; override with env if you want live.
DRY_RUN_FLAG="--dry-run"
if [ "${POLYMARKET_CRON_LIVE:-}" = "1" ]; then
    DRY_RUN_FLAG=""
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running Polymarket bot ..." | tee -a "$LOG_FILE"
"$PYTHON" "${PROJECT_DIR}/scripts/run_polymarket_bot.py" --db "$DB_PATH" $DRY_RUN_FLAG 2>&1 | tee -a "$LOG_FILE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done." | tee -a "$LOG_FILE"
```

**Step 2: Make executable**

Run: `chmod +x /opt/data/stock-research/dfw_temp_model/scripts/cron_polymarket_bot.sh`

**Step 3: Smoke test the wrapper**

Run:
```bash
cd /opt/data/stock-research/dfw_temp_model
POLYMARKET_PRIVATE_KEY=0x$(openssl rand -hex 32) ./scripts/cron_polymarket_bot.sh
```
Expected: script completes, log file created under `logs/`, JSON report visible in output.

**Step 4: Commit**

```bash
git add scripts/cron_polymarket_bot.sh
git commit -m "feat: add separate cron wrapper for Polymarket bot"
```

---

### Task 11: Run full test suite

**Objective:** Ensure the new trading code does not break existing tests.

**Step 1: Run all tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/pytest -q`
Expected: all existing tests pass, new trading tests pass.

**Step 2: Commit if any fixes were needed**

If fixes were needed, commit them with a clear message. If no fixes, commit a no-op or skip.

---

### Task 12: Document the bot and safety checklist

**Objective:** Add a README section explaining how to run the bot, how to enable live mode, and the risks.

**Files:**
- Modify: `/opt/data/stock-research/dfw_temp_model/README.md`

**Step 1: Append trading section**

Add to `README.md`:
```markdown
## Polymarket Trading Bot (experimental)

A separate optional module in `dfw_temp_model/trading/` uses the live forecast
data to evaluate Polymarket weather markets. It is **dry-run by default**.

### Run once (dry run)

```bash
POLYMARKET_PRIVATE_KEY=0x... POLYMARKET_DRY_RUN=true \
  .venv/bin/python scripts/run_polymarket_bot.py --dry-run
```

### Run on a schedule

Add `scripts/cron_polymarket_bot.sh` to cron independently of the dashboard
cron. It always defaults to dry-run; set `POLYMARKET_CRON_LIVE=1` only after
reviewing the safety checklist below.

### Safety checklist before live mode

- [ ] Wallet has only funds you can afford to lose.
- [ ] `POLYMARKET_MAX_ORDER_SIZE_USDC` is set to a small value.
- [ ] You are legally permitted to trade on Polymarket from your jurisdiction.
- [ ] You understand the settlement station may be KDFW while the model targets KDAL.
- [ ] You have run at least 7 days of dry-run trades and reviewed the logs.

### Architecture note

The trading bot reads from the same SQLite DB that the dashboard writes to, but
it does not modify the dashboard ingestion scripts. A failure in the bot leaves
the hourly dashboard update (`cron_update_dashboard.sh`) untouched.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Polymarket bot usage and safety notes"
```

---

## Tests / Validation

- `tests/test_trading_config.py` — env loading and dry-run default.
- `tests/test_trading_market.py` — Gamma search parsing.
- `tests/test_trading_client.py` — dry-run prevents posting.
- `tests/test_trading_signal.py` — probability math and forecast helper.
- `tests/test_trading_decision.py` — edge-based intent generation.
- `tests/test_trading_executor.py` — execution/safety layer.
- Full suite: `.venv/bin/pytest -q`
- End-to-end dry run: `scripts/run_polymarket_bot.py --dry-run`

## Risks, Tradeoffs, and Open Questions

1. **Settlement station mismatch.** The model targets KDAL (Dallas Love Field) but Polymarket DFW markets usually settle against KDFW. This plan documents it and makes it configurable, but a real-money bot should either model KDFW directly or quantify the KDAL/KDFW basis.
2. **Front-day model risk.** The advection model has a front-day fallback with higher uncertainty. The decision engine currently uses a fixed `model_std`; this could be widened on front days.
3. **Private key handling.** The plan uses env vars only. For production, consider a secrets manager or hardware wallet.
4. **Allowances and gas.** The client wrapper does not yet auto-approve USDC/CTF allowances. Before first live trade, allowances must be set manually or an additional task added.
5. **Liquidity and slippage.** The current decision engine uses midpoint/last price, not real order-book depth. A production bot should check `get_order_book()` and size against available liquidity.
6. **Redemption.** Winning positions are not auto-redeemed. A follow-up task should add a redemption sweep.
7. **Legal/jurisdictional.** Confirm Polymarket trading is permitted in your jurisdiction before enabling live mode.

## Files Likely to Change

- `pyproject.toml`
- `uv.lock`
- `dfw_temp_model/trading/config.py`
- `dfw_temp_model/trading/market.py`
- `dfw_temp_model/trading/client.py`
- `dfw_temp_model/trading/signal.py`
- `dfw_temp_model/trading/decision.py`
- `dfw_temp_model/trading/executor.py`
- `dfw_temp_model/trading/bot.py`
- `dfw_temp_model/trading/__init__.py`
- `scripts/run_polymarket_bot.py`
- `scripts/cron_polymarket_bot.sh`
- `tests/test_trading_*.py`
- `README.md`

## Explicit Non-Changes

The following files are intentionally left untouched to protect the working dashboard pipeline:

- `scripts/ingest_live_metars.py`
- `scripts/generate_dashboard.py`
- `scripts/cron_update_dashboard.sh`
- `dfw_temp_model/storage/obs_db.py`
- `dfw_temp_model/data/hrrr.py`
- `dfw_temp_model/data/aviationweather.py`
