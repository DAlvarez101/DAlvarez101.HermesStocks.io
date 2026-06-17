# Polymarket Weather Prediction Markets — Automation Research Report

**Date:** 2026-06-14  
**Scope:** Typical methods, API endpoints, and workflows for discovering, analyzing, and automating Polymarket weather prediction markets.  
**Disclaimer:** This is a research summary, not a trading recommendation. Automated trading carries financial, technical, and regulatory risks.

---

## 1. Executive Summary

Polymarket operates the largest crypto-native prediction market by volume. Its **Weather** category turns meteorological outcomes (daily city temperatures, precipitation, snowfall, hurricanes, global climate benchmarks) into binary or bucketed tradable contracts. Prices are probabilities: a share trading at $0.72 implies a 72% market-implied probability of the stated outcome.

Automation opportunities center on three gaps:
1. **Information lag:** Forecast models update every 6 hours; market prices often lag by minutes to hours.
2. **Spatial mismatch:** Bucketed temperature markets settle on airport weather stations, but naive forecasters use city-center coordinates, introducing 3–8°F error on 1–2°F-wide buckets.
3. **Calibration error:** Most wallets lose money because of position sizing and market selection, not just directional wrongness.

A typical automated weather trading system ingests model forecasts, computes bucket probabilities, compares them to market prices, sizes positions with Kelly logic, and posts limit orders via Polymarket's CLOB API on Polygon.

---

## 2. Polymarket API Architecture

There are three public API layers. Read-only market discovery is unauthenticated; trading requires wallet-based signatures.

| API | Base URL | Auth | Purpose |
|---|---|---|---|
| **Gamma API** | `https://gamma-api.polymarket.com` | None | Market discovery, events, metadata, prices, volume, search |
| **CLOB API** | `https://clob.polymarket.com` | Wallet for writes | Order book, prices, order entry/cancel, price history |
| **Data API** | `https://data-api.polymarket.com` | None | Trades, activity, open interest, positions |

### 2.1 Key Identifiers

- **Event** — a group of related markets under one theme (e.g., "Chicago weather, April 14–16").
- **Market** — a single question, often binary or categorical buckets.
- **Condition ID** — a 0x-prefixed hex string representing the market's condition.
- **Token ID** — a numeric string from `clobTokenIds`; each market has one token per outcome (`[YES, NO]` or `[bucket1, bucket2, ...]`).

Fields `outcomePrices`, `outcomes`, and `clobTokenIds` are returned as **JSON strings inside JSON** (double-encoded). In Python, parse them with `json.loads(market["outcomePrices"])`.

---

## 3. Market Discovery for Weather Markets

### 3.1 Gamma API Search and Listing Endpoints

| Method | Endpoint | Use Case |
|---|---|---|
| Search | `GET /public-search?q=QUERY` | Full-text search for weather events |
| List markets | `GET /markets?limit=N&active=true&closed=false&order=volume&ascending=false` | Browse active weather markets |
| Filter by tag | `GET /markets?tag=weather&active=true` | Dedicated weather category scan |
| List events | `GET /events?limit=N&active=true&closed=false&order=volume&ascending=false` | Browse events with nested markets |
| Tags | `GET /tags` | Discover available category slugs |

### 3.2 Example: Scan Weather Markets

```bash
curl "https://gamma-api.polymarket.com/markets?tag=weather&active=true&limit=100"
```

Response fields to extract for automation:
- `question` — the human-readable market question
- `conditionId` — needed for price history and Data API
- `clobTokenIds` — needed for CLOB price/book/order endpoints
- `outcomePrices` — current implied probabilities
- `volume`, `liquidity`, `openInterest` — market size and participation
- `endDate` — market expiry/resolution date
- `description` — often contains the resolution source and weather station

### 3.3 Parsing Market Metadata

```python
import json, requests

url = "https://gamma-api.polymarket.com/markets?tag=weather&active=true&limit=10"
markets = requests.get(url).json()

for m in markets:
    prices = json.loads(m["outcomePrices"])
    outcomes = json.loads(m["outcomes"])
    tokens = json.loads(m["clobTokenIds"])
    print(m["question"])
    for o, p, t in zip(outcomes, prices, tokens):
        print(f"  {o}: {float(p)*100:.1f}% (token {t})")
```

---

## 4. Weather Market Types and Resolution

| Market Type | Example | Settlement Source |
|---|---|---|
| Daily high temperature bucket | "Will Chicago high on Apr 14 be 46–47°F?" | Official airport station daily max (NWS/NOAA) |
| Daily low temperature bucket | "Will NYC low on Apr 14 be 38–39°F?" | Official airport station daily min |
| Binary threshold | "Will London high exceed 70°F on Apr 14?" | Station reading vs. threshold |
| Precipitation binary | "Will Miami get ≥0.01" rain on Apr 14?" | Station daily precipitation |
| Snowfall threshold | "Will Boston get ≥2" snow this week?" | Station cumulative snowfall |
| Hurricane/tornado | Landfall or count within region/time | NOAA / National Hurricane Center |
| Global climate | "Will 2026 be hottest year on record?" | NOAA, NASA GISS, Copernicus |

Bucket markets are typically 1–2°F wide and resolve to the exact official reading at the specified station. The station matters more than the city name because urban heat-island effects can shift temperatures by several degrees.

---

## 5. Forecast Data Ingestion

Automation strategies require matching Polymarket questions to high-quality weather forecasts at the exact resolution station.

### 5.1 Common Forecast Sources

| Source | Base URL / API | Update | Strengths |
|---|---|---|---|
| **ECMWF IFS** | Open-Meteo / ECMWF APIs | ~6 h | Best global deterministic skill |
| **GEFS** | NOAA / Open-Meteo | ~6 h | 31-member ensemble, uncertainty quantification |
| **UKMO** | Met Office / Open-Meteo | ~6 h | Strong medium-range skill |
| **NWS hourly** | `https://api.weather.gov` | ~1 h | US official observations and short-term forecasts |
| **Open-Meteo** | `https://api.open-meteo.com` | Varies | Convenient ensemble access |

### 5.2 Best Practices for Forecast Matching

1. Parse the market description to extract the exact weather station identifier (airport code, WMO ID, or station name).
2. Pull forecasts at the station coordinates, not the city center.
3. Use the forecast horizon to set uncertainty (σ). Typical rule of thumb:
   - 6 hours: σ ≈ 0.8°F
   - 1–2 days: σ ≈ 1.5–2.5°F
   - 7–10 days: σ ≈ 4–5.5°F
4. Blend multiple models by historical Brier score or inverse-variance weighting.

---

## 6. Probability and Edge Models

### 6.1 Gaussian Bucket Probability

For a temperature bucket market with lower bound L and upper bound U, given a forecast mean μ and horizon-derived σ:

```
P(bucket) = CDF(U; μ, σ) − CDF(L; μ, σ)
```

Use the normal CDF or a t-distribution if the ensemble spread is heavy-tailed. GEFS member agreement can be used directly as an alternative probability estimate.

### 6.2 Ensemble Blending

```python
weights = {"ECMWF": 0.35, "GEFS": 0.25, "UKMO": 0.20, "NWS": 0.20}
blended_prob = sum(weights[m] * model_prob[m] for m in weights)
```

Outlier forecasts (>1.5σ from ensemble mean) are often down-weighted 50%. Weights can be recalibrated per city by tracking recent Brier scores.

### 6.3 Edge Detection

```
edge = model_probability − market_price
```

- Positive edge on YES → consider buying.
- Negative edge on YES / positive edge on NO → consider selling or buying NO.

Typical filters before trading:
- `|edge|` ≥ threshold (e.g., 8%)
- Z-score ≥ threshold (e.g., 1.5) to ensure the edge is statistically robust
- Time to expiry > minimum (e.g., 2 hours) to avoid illiquid last-minute pricing
- Volume / liquidity above a minimum to ensure exit capacity

---

## 7. Order Execution and CLOB API

### 7.1 Read-Only Price and Book

| Endpoint | Description |
|---|---|
| `GET /price?token_id=TOKEN&side=buy` | Best available buy price |
| `GET /midpoint?token_id=TOKEN` | Midpoint of best bid/ask |
| `GET /spread?token_id=TOKEN` | Bid-ask spread |
| `GET /book?token_id=TOKEN` | Full order book |
| `GET /prices-history?market=CONDITION&interval=...&fidelity=N` | Historical midpoint prices |

### 7.2 Trading Authentication

The CLOB API uses wallet-based authentication, not a simple API key:

1. Create or derive API credentials from a Polygon wallet using `py-clob-client`.
2. Sign requests with `POLY_HMAC_SHA256` headers derived from the API secret.
3. Two levels:
   - **L1** — main wallet, full trading permissions.
   - **L2** — derived key, read-only.

Example setup with the official Python SDK:

```python
from py_clob_client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key="YOUR_PRIVATE_KEY",
    chain_id=137
)

api_creds = client.create_or_derive_api_creds()
client.set_api_creds(api_creds)
```

### 7.3 Placing Orders

```python
order = client.create_and_post_order(
    token_id="YES_TOKEN_ID",
    price=0.55,
    size=100,
    side="BUY"
)
```

- Markets settle in **USDC** on **Polygon chain ID 137**.
- Orders are GTC (good-till-cancelled) by default unless a TTL is set.
- WebSocket feeds can monitor fills and price changes.

### 7.4 WebSocket Feeds

Base URL: `wss://ws-subscriptions-clob.polymarket.com/ws/`

| Feed | Subscription |
|---|---|
| Price updates | `{ "type": "market", "assets_id": "TOKEN_ID" }` |
| Order book diffs | Same channel with book-depth parameter |
| Trades | Global or market-specific trade stream |
| User events | Authenticated personal order status |

---

## 8. Risk Management and Position Sizing

Most automated strategies use some variant of fractional Kelly to avoid ruin.

### 8.1 Fractional Kelly Example

```
size = full_kelly(win_prob, odds) × fraction × bankroll
       | fraction = 0.15
       | max bankroll = 5%
       | hard cap = $100 per trade
```

- Larger edges produce larger stakes.
- Marginal edges produce near-zero stakes automatically.
- Precipitation/snowfall markets often get smaller allocation because confidence bands are wider.

### 8.2 Other Risk Controls

- **Market selection filters:** skip markets with <2 h to expiry, low volume, or wide spreads.
- **Spread tolerance:** only trade if the bid-ask spread does not swallow the edge.
- **Max exposure per city / event:** avoid correlated weather risk.
- **Daily/weekly loss limits:** stop trading if drawdown exceeds threshold.
- **Model validation:** backtest against historical bucket-hit rates and recalibrate weights.

---

## 9. Typical Automation Workflow

1. **Discovery:** Poll `gamma-api.polymarket.com/markets?tag=weather&active=true` every 2–5 minutes.
2. **Parsing:** Extract condition IDs, token IDs, thresholds, resolution stations, and expiry times.
3. **Forecasting:** Query ECMWF, GEFS, UKMO, and NWS for each active market.
4. **Modeling:** Compute blended bucket probabilities and edge vs. market price.
5. **Filtering:** Apply edge, z-score, liquidity, and expiry filters.
6. **Sizing:** Run fractional Kelly to determine order size.
7. **Execution:** Post limit orders via CLOB API on Polygon.
8. **Monitoring:** Use WebSocket or REST polling for fills; cancel and re-evaluate stale orders.
9. **Settlement:** Track market resolution and reconcile P&L via Data API.

---

## 10. Data and Analytics Endpoints

| Endpoint | Use |
|---|---|
| `GET https://data-api.polymarket.com/trades?limit=N` | Recent global trades |
| `GET https://data-api.polymarket.com/trades?market=CONDITION_ID&limit=N` | Trades for a specific market |
| `GET https://data-api.polymarket.com/oi?market=CONDITION_ID` | Open interest |
| `GET https://clob.polymarket.com/prices-history?market=CONDITION_ID&interval=all&fidelity=50` | Historical prices |

These are useful for:
- Backtesting edge detection signals.
- Analyzing whale wallet activity.
- Building calibration plots (market price vs. outcome frequency).
- Monitoring open interest for liquidity and crowding.

---

## 11. Tools, SDKs, and References

| Resource | URL | Purpose |
|---|---|---|
| Official docs | `https://docs.polymarket.com` | API reference |
| Python SDK | `https://github.com/Polymarket/py-clob-client` | Trading client |
| Polymarket GitHub | `https://github.com/Polymarket` | Open-source repos |
| Conditional tokens contracts | `https://github.com/gnosis/conditional-tokens-contracts` | Smart contract reference |
| pm.wiki guide | `https://pm.wiki/learn/polymarket-api` | Third-party API guide |
| Polymarket Weather example | `https://polymarketweather.com/` | Automated weather trading pipeline reference |

---

## 12. Risks, Limitations, and Compliance Notes

1. **Trading risk:** Prediction markets are zero-sum minus fees. Historical data shows the majority of wallets lose money.
2. **Model risk:** Weather forecasts can be wrong; ensemble disagreement can be large for extreme or rapidly evolving events.
3. **Liquidity risk:** Thin markets may not allow entry/exit at quoted prices, especially near expiry.
4. **Operational risk:** API downtime, wallet key management, and smart-contract bugs can cause losses.
5. **Geographic restrictions:** Polymarket trading is restricted in several jurisdictions (including the United States for many users). Read-only data access is generally available globally, but users must verify their local regulations.
6. **Fees:** Polymarket generally charges 0% trading fees on most markets, but settlement and blockchain transaction costs (Polygon gas) still apply.
7. **Settlement delays:** Official weather data can be delayed or revised; understand the market's resolution source before automating.

---

## 13. Does It Actually Work? — A Reality Check from the Field

This section synthesizes public data, independent research, and real-world trader reports (especially Reddit / forum discussions) to answer the question: *Can a retail trader or bot consistently profit on Polymarket weather markets?*

### 13.1 The headline numbers are brutal

| Study / Source | Scope | Key Finding |
|---|---|---|
| Sergeenkov / The Defiant, Apr 2026 | 2.5M wallets, on-chain Polygon data | **84.1% of Polymarket traders are unprofitable**; only **2%** ever earned >$1,000; only **0.033%** earned >$100,000. Odds of earning $5,000/month in any single month: **<1%** (The Defiant, 2026). |
| Digitech Bytes / Thorsten Meyer, May 2026 | 95M transactions, Apr 2024–Dec 2025 | Only **0.51%** of wallets earned >$1,000. **99.49%** broke even or lost. Average arbitrage window collapsed from **12.3s in 2024 to 2.7s in 2026**; **73%** of arb profits captured by sub-100ms bots (Meyer, 2026). |
| Decrypt / LayerHub, Nov 2024 | Post-election wallet data | **86–88%** of wallets were in the red; only ~**1%** realized ≥$1,000 in profit (Decrypt, 2024). |
| Reddit consensus (r/investing, r/Polymarket, r/predictionmarkets) | Anecdotal + P&L screenshots | “Boring markets” (85–90% likely outcomes) and “wait for overreaction, then fade” are the most commonly cited working strategies. Most users report losses or tiny gains after fees and locked capital (Polifly, 2025; Prediction News, 2026). |

> **Plain-language takeaway:** The probability that an average retail wallet finishes ahead is roughly 12–16%. The probability it finishes meaningfully ahead (>$1,000 lifetime) is roughly 0.5–2%. Automation improves execution discipline but does not fix the structural math.

### 13.2 What successful traders and bots actually do

Public sources converge on a small set of durable edges:

| Approach | Why it can work | What it requires | Source |
|---|---|---|---|
| **Domain specialization** | Genuine expertise in a narrow category (e.g., weather, NFL, Fed policy) lets you out-model the crowd. | Meteorological data access, per-city bias calibration, careful position sizing. | Digitech Bytes (Meyer, 2026); Polymarket Weather website; Laika AI (2026). |
| **Cross-platform arbitrage** | Mispricings between Polymarket and sharper venues (sportsbooks, Kalshi) persist for minutes, not seconds. | Accounts on both venues, fast settlement rails, $5K–$50K working capital. | Reddit user @b00k13 reported ~$8,300 over 3 months; Prediction News (2026). |
| **Fade the overreaction** | News spikes often overshoot; buying the opposite side 30–60 min later harvests mean reversion. | Patience, liquidity screening, stop-loss discipline. | Reddit strategy summary via Polifly (2025). |
| **“Boring markets” / high-probability bond strategies** | Buying near-certain outcomes at 85–95¢ earns small, repeatable premiums. | Diversification across many independent events; strict sizing to survive rare tails. | Reddit / Polifly (2025); Digitech Bytes strategy #5. |
| **Smart-wallet copy/monitoring** | A handful of wallets show persistent edge; tracking their early entries can surface opportunities. | On-chain tooling, skepticism (past P&L ≠ future), fast execution. | Polyburg (2025); polymarketanalytics.com leaderboard. |

**Cited successful examples:**
- **gopfan2** — ~$2M profit, mostly weather markets: buy YES < $0.15, buy NO > $0.45, ~$1 per position, thousands of trades (Laika AI, 2026).
- **meropi** — ~$30K profit with fully automated $1–$3 micro-bets; bought long-shots at $0.01 for 500× payoffs when they hit (Laika AI, 2026).
- **1pixel** — ~$18.5K from $2.3K deposits trading only NYC/London weather; turned $6→$590 and $15→$547 on mispriced ranges (Laika AI, 2026).
- **Reddit @b00k13** — ~$8,300 in 3 months from cross-venue sportsbook/Polymarket arbitrage, with verified P&L data (Prediction News, 2026).

> **Important caveat:** Survivorship bias is massive. The winners above are visible precisely because they are rare. The median wallet loses money.

### 13.3 Pitfalls that the existing notes under-weighted

Community and researcher feedback surfaced several failure modes that the original automation blueprint did not emphasize enough:

1. **Execution quality matters as much as signal quality.** A backtest assumes instant, perfect fills; live trading has market-data latency, strategy latency, execution latency, queue competition, partial fills, and slippage. Mateosoul (DEV Community, 2025) reports that adding realistic latency/fill simulation wiped out most of his apparent alpha until he rebuilt the system around microstructure.

2. **Arbitrage windows are microseconds, not minutes, for retail bots.** The average Polymarket arb window is now ~2.7 seconds; 73% of profits go to sub-100ms infrastructure. Off-the-shelf “arbitrage finder” tools are usually too slow (Meyer, 2026).

3. **Capital lock-up / opportunity cost is real.** Long-dated markets tie up USDC for weeks or months. Reddit traders repeatedly note that the 2–5% “risk-free” premium on 95¢ NO shares is often eaten by the opportunity cost of locked capital and the tail risk of a black-swan outcome.

4. **Oracle / resolution risk is non-zero.** Polymarket uses the UMA Optimistic Oracle. A $7M Ukraine mineral-deal market in March 2025 resolved “Yes” despite no official confirmation, after a single UMA whale reportedly cast ~5M tokens (~25% of votes). Polymarket declined refunds, calling it an “unprecedented situation” (CCN, 2025; The Defiant, 2025). This means even a “correct” forecast can lose to governance or ambiguous interpretation.

5. **Yes-bias and adverse selection.** Markets systematically overestimate positive/attention-grabbing outcomes. The Binance Square bot-battle experiment found the “reverse strategy” (fading consensus >80%) won $1,740 from $500, while the “smartest” arbitrage bot ended with $82 (Binance Square, 2026). Retail bots often get picked off by better-informed flow.

6. **Influencer / YouTube hype overstates ease.** Reddit threads such as “Is Polymarket a money glitch?” are dominated by corrections: it is not free money, large bets move prices, 2% tail events happen, and fees/spreads erode the premium. YouTube “$1K/week” claims conflict directly with the on-chain wallet data.

7. **Regulatory and jurisdictional risk is rising.** The CFTC classified prediction markets as derivatives in March 2026, issued an insider-trading advisory under Rule 180.1 in February 2026, and warned about AI trading algorithm fraud. State-level taxes (e.g., Kentucky’s proposed 14.25%) and country bans (e.g., France investigation) add operational uncertainty (Digitech Bytes, 2026; Decrypt, 2024).

8. **Weather-specific pitfalls:**
   - **Station mismatch:** City-center vs. airport coordinates can shift temperatures 3–8°F, enough to flip a 1–2°F bucket (Polymarket Weather website; Laika AI, 2026).
   - **Model-to-market lag is shrinking:** As more weather bots enter, the 6-hour model-update edge decays faster. Polymarket Weather reports 85–90% bucket-hit rate but caps sizing tightly because confidence intervals widen quickly beyond 1–2 days.
   - **Precipitation/snowfall is harder than temperature:** Wider confidence bands, lower predictability, and thinner liquidity mean smaller or no allocations.
   - **CLI/NWS settlement timing:** US weather contracts settle on the NWS Daily Climate Report at 8:00 AM ET the next day (delayed to 11:00 AM ET if CLI conflicts with METAR). Last-fair-price settlement after one week adds uncertainty (Polymarket US docs, 2026).

### 13.4 Verdict: reasonable vs. bad bet

**Automation is a reasonable bet only if:**
- You have genuine domain expertise and the bot merely automates an independently sound thesis.
- You treat it as research with loss-bounded capital, not a yield strategy.
- You cross-platform arb with real accounts on both venues and can tolerate settlement delays.
- You log every trade and honestly evaluate 6-month P&L before scaling.
- You size so that a single bad oracle call or black-swan event cannot ruin the bankroll.

**Automation is a bad bet if:**
- You expect off-the-shelf templates to capture 2.7-second arb windows.
- You are chasing YouTube/influencer “easy money” claims.
- You lock most of your capital in long-dated markets for a few percent.
- You ignore execution latency, fill quality, and adverse selection.
- You assume the weather model is always right; models disagree, especially beyond 48 hours.

### 13.5 Bottom line for the weather-automation project

The original notes described a technically sound pipeline. The field evidence suggests the pipeline is **necessary but not sufficient**: the hard part is not wiring Gamma + CLOB + Open-Meteo together; it is achieving **superior forecasts, superior execution, and superior risk control** in a market where ~85% of participants lose. A small, experimental bot with tight Kelly caps, paper-trading first, and a 6-month review horizon is defensible. Scaling it as a reliable profit engine is not supported by the public data.

### 13.6 Sources referenced in this reality check

- Andrey Sergeenkov / The Defiant. “84% of Polymarket Traders Are Losing Money, New Research Finds.” April 6, 2026. https://thedefiant.io/news/research-and-opinion/polymarket-profitability-report-april-2026
- Thorsten Meyer / Digitech Bytes. “Are Polymarket Trading Bots Actually Profitable? The Math Behind 2026’s Prediction-Market Arbitrage Industry.” May 2026. https://digitechbytes.com/emerging-consumer-tech-explained/are-polymarket-trading-bots-actually-profitable-the-math-behind-2026-s-predictio
- Ryan S. Gladwin / Decrypt. “Whales Won Millions Betting on Trump, But 86% of Polymarket Traders Have Lost Money.” November 7, 2024. https://decrypt.co/290625/most-polymarket-traders-have-lost-money
- Prediction News. “Reddit and Hacker News Users Claim AI and Bot Arbitrage Profits on Polymarket.” June 9, 2026. https://predictionnews.com/story/reddit-user-says-ai-coded-arbitrage-bot-made-5k-on-polymarket-93b950a4
- Mateosoul / DEV Community. “Building Polymarket Trading Bot: Why Most Trading Bots Fail Once You Model Reality Properly.” May 27, 2026. https://dev.to/mateosoul/building-polymarket-trading-bot-why-most-trading-bots-fail-once-you-model-reality-properly--2bkn
- Polifly. “Polymarket Profit Strategies from Reddit: What Actually Works.” January 6, 2025. https://polifly.io/blog/polymarket-profit-strategies-reddit
- Polyburg Team. “How Smart Wallets Dominate Polymarket: A Data-Driven Analysis.” January 15, 2025. https://polyburg.com/blog/smart-wallets-polymarket-guide
- Laika AI. “Polymarket Weather Markets: Trading Strategies Guide 2026.” April 13, 2026. https://laikalabs.ai/prediction-markets/trade-polymarket-weather-markets
- Polymarket Weather. “WeatherBot for Polymarket Markets.” https://polymarketweather.com/
- 万联welinkBTC / Binance Square. “In-Depth Analysis of the Polymarket Bot Battle Experiment.” June 2026. https://www.binance.com/en/square/post/296107213210482
- Prashant Jha / CCN. “Polymarket Hit by Whale Manipulation as 5 Million Votes Rig Settlement.” March 26, 2025. https://www.ccn.com/news/crypto/polymarket-whale-manipulation-5-million-votes-rig-settlement/
- r/investing. “Is Polymarket a money glitch?” Archived thread. https://www.reddit.com/r/investing/comments/1hsmtwo/is_polymarket_a_money_glitch/
- r/CryptoCurrency. “Polymarket Betters Are Overwhelmingly Losing Money: Here’s Why.” https://www.reddit.com/r/CryptoCurrency/comments/1hsmtwo/polymarket_betters_are_overwhelmingly_losing_money_heres_why/

---

## 14. Summary Table: Endpoints at a Glance

| Layer | Endpoint | Auth | Weather Use |
|---|---|---|---|
| Gamma | `GET /markets?tag=weather&active=true` | No | Discovery |
| Gamma | `GET /events?tag=weather&active=true` | No | Event grouping |
| Gamma | `GET /public-search?q=weather+CITY` | No | Search |
| CLOB | `GET /price?token_id=...&side=buy` | No | Current price |
| CLOB | `GET /midpoint?token_id=...` | No | Fair midpoint |
| CLOB | `GET /book?token_id=...` | No | Liquidity |
| CLOB | `GET /prices-history?market=...` | No | Backtesting |
| CLOB | `POST /order` | Wallet | Trade execution |
| Data | `GET /trades?market=...` | No | Activity analysis |
| Data | `GET /oi?market=...` | No | Position sizing cue |
| WebSocket | `wss://ws-subscriptions-clob.polymarket.com/ws/` | No/User | Real-time ticks |

---

*Report compiled from the Hermes `polymarket` skill, official Polymarket documentation, and public third-party guides. Not financial advice.*
