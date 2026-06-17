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
