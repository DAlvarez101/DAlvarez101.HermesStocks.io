# DFW Temperature Advection Model

A research Python package that builds a historical observation + forecast dataset for the Dallas–Fort Worth area and experiments with advection-weighted temperature models for high-temperature prediction markets.

The project fetches IEM ASOS observations and Open-Meteo historical forecasts, aligns them into a Parquet dataset, and tests baseline and advection-weighted prediction models on a 2024 holdout.

## Install

```bash
uv sync
# or
pip install -e .
```

Then run the tests with:

```bash
pytest
```

This is research code for Polymarket weather markets and is not intended for production use.
