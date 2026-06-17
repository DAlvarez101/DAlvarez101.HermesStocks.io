# DAlvarez101.HermesStocks.io

A GitHub Pages dashboard-of-dashboards repository for stock research, weather forecast visualization, and experimental Polymarket trading tools.

## DFW Temperature Advection Model

A research Python package under `dfw_temp_model/` that builds a historical observation + forecast dataset for the Dallas–Fort Worth area and experiments with advection-weighted temperature models for high-temperature prediction markets.

The project fetches IEM ASOS observations and Open-Meteo historical forecasts, aligns them into a Parquet dataset, and tests baseline and advection-weighted prediction models on a 2024 holdout.

### Install

```bash
cd dfw_temp_model
uv sync
# or
pip install -e .
```

Then run the tests with:

```bash
pytest
```

### Polymarket trading bot (experimental, dry-run by default)

A separate trading subpackage under `dfw_temp_model/trading/` wires the live METAR/HRRR forecast pipeline into Polymarket temperature markets. It is **dry-run by default** and will not post real orders unless you explicitly opt in.

Required environment variables (all loaded from the environment, never hard-coded):

| Variable | Purpose |
|----------|---------|
| `POLYMARKET_PRIVATE_KEY` | 64-hex Polygon private key used to sign CLOB orders |
| `POLYMARKET_HOST` | CLOB host (default `https://clob.polymarket.com`) |
| `POLYMARKET_CHAIN_ID` | Polygon chain ID (default `137`) |
| `POLYMARKET_DRY_RUN` | `true` to simulate orders, `false` for live trading |
| `POLYMARKET_MAX_ORDER_SIZE_USDC` | Max notional to risk per trade |
| `POLYMARKET_MIN_EDGE_BPS` | Minimum model edge in basis points before trading |

Run once in dry-run mode:

```bash
cd dfw_temp_model
export POLYMARKET_PRIVATE_KEY=0x...
export POLYMARKET_DRY_RUN=true
./scripts/cron_polymarket_bot.sh
```

To enable live trading from cron, set `POLYMARKET_CRON_LIVE=1` in addition to `POLYMARKET_DRY_RUN=false`.

This is research code for Polymarket weather markets and is not intended for production use.
