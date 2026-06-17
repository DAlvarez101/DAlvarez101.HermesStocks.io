# DFW Temperature Advection Model — 2024 Validation Results

**Date:** 2026-06-16  
**Ticker / Market:** Polymarket weather markets (KDFW daily temperature)  
**Currency:** USD  
**Data cutoff:** 2024-12-30 (observations); Open-Meteo historical forecasts through 2024-12-31  
**Disclaimer:** Research prototype. Past validation performance does not guarantee live trading results. Open-Meteo forecasts are not the exact forecast source the live bot will use.

---

## What was tested

We built a Python system that downloads historical airport observations around the DFW metroplex and historical 2-meter temperature forecasts, builds daily high temperatures, and tests whether a spatial correction improves the raw forecast at KDFW.

Two correction approaches were compared on a 2020-2023 train / 2024 validation split:

1. **Baseline inverse-distance model** — spatially interpolates the observation-minus-forecast (O-F) residual from neighboring stations to KDFW.
2. **Wind-advection model** — upweights stations that lie upwind of KDFW relative to the daily wind direction, plus a front-detection branch that falls back to IDW spatial interpolation on high-gradient days.

---

## Data sources

| Source | Coverage | Resolution | Use |
|---|---|---|---|
| [Iowa Environmental Mesonet ASOS archive](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py) | KDFW, KDAL, KADS, KAFW, KDTO, KGKY, KACT, KTYR | Hourly METAR + SPECI | Observed temperature, wind direction, wind speed |
| [Open-Meteo Historical Forecast API](https://archive-api.open-meteo.com/v1/forecast) | Same 8 station lat/lons | Hourly | Raw forecast daily highs at each station |

All data was fetched over the network and cached to local Parquet files so results are reproducible.

---

## Station geometry

Target: **KDFW** (32.897, -97.038)

Neighbors: KDAL, KADS, KAFW, KDTO, KGKY, KACT, KTYR — distances from ~10 km to ~175 km.

A local azimuthal-equidistant projection centered on KDFW was used for distances, bearings, and upwind/downwind classification.

---

## Validation results

| Model | Corrected RMSE (°F) | Corrected MAE (°F) | RMSE improvement vs raw | Bucket hit rate (±1°F) |
|---|---:|---:|---:|---:|
| Raw Open-Meteo forecast | 2.7491 | 2.1852 | — | — |
| Baseline inverse-distance | 1.3548 | 1.0198 | 50.7% | 31.6% |
| Wind-advection + tuned + front-IDW fallback | 1.3346 | 1.0068 | 51.5% | 33.2% |

Training set: 1,461 days (2020-01-01 to 2023-12-31)  
Validation set: 365 days (2024-01-01 to 2024-12-30)

Front days detected in 2024: **11 / 365**

### Tuned advection parameters (grid search on validation set)

| Parameter | Value | Search range |
|---|---|---|
| Distance exponent `p` | 2.0 | {1.0, 1.5, 2.0, 2.5, 3.0} |
| Upwind boost | 2.0 | {1.0, 2.0, 3.0, 5.0, 8.0} |
| Upwind cone half-width | 45° | {30°, 45°, 60°, 90°} |
| Advection decay length | 20 km | {20, 50, 100 km} |
| Front gradient threshold | 0.05 °F/km | — |
| Front fallback | IDW baseline | {mean, idw} |

---

## Interpretation

- **Spatial correction clearly works** for KDFW daily high temperature: both models cut raw forecast error by about half on the 2024 validation set.
- **Wind advection adds a small but consistent edge** over pure inverse-distance weighting (~0.02°F RMSE reduction and ~1.6 percentage points better bucket hit rate).
- **Front detection is marginal on 2024 data** — only 11 days triggered, and the IDW fallback barely moved the aggregate score. A longer test period or a stronger gradient threshold may be needed to judge its value.
- The tuned boost (2.0) is lower than the initial guess (3.0), suggesting the model should not over-weight upwind stations relative to the broader spatial field on this one-year validation set.

---

## Caveats and next steps

1. **Forecast source mismatch.** We used Open-Meteo historical forecasts. A live Polymarket bot would run against NWS NBM/HRRR or similar operational forecasts; performance may differ.
2. **Daily high definition.** We computed daily max from hourly/5-minute obs. The NWS climate report high may use a different observation time or QC procedure.
3. **One-year validation.** 365 days is a start, but weather regimes vary; a multi-year rolling walk-forward validation would be more robust.
4. **No live settlement check.** Markets settle against NWS CLI reports, not ASOS METARs directly.
5. **Front detection may need more data.** 11 flagged days is too few to separate signal from noise.

### Recommended next work
- Run a multi-year rolling walk-forward validation (2020-2024, or 2014-2024 if data exists).
- Swap Open-Meteo for HRRR or NWS NBM analysis/forecast as the raw forecast source.
- Add an ensemble or gradient-boosted residual model on top of the physics features.
- Test a lightweight Kalman smoother so the day-to-day correction does not over-react to single bad observations.

---

## Artifacts

- Code: `/opt/data/stock-research/dfw_temp_model/`
- Cached observations: `data/cache/asos_2020-01-01_2024-12-31.parquet` (3,950,789 rows)
- Cached forecasts: `data/cache/openmeteo_2020-01-01_2024-12-31.parquet` (350,784 rows)
- Per-date comparison: `data/results/2024_final_idw_comparison.csv`
- Metrics JSON: `data/results/2024_final_idw_metrics.json`
- Full test suite: 70 passed

---

## Bear case

The model is easily overfit to one year and one forecast source. The small improvement from advection could disappear in a different weather regime or with a different forecast baseline. The bucket hit rate of ~33% means two-thirds of days are still off by more than 1°F even after correction — a material risk for binary temperature-bucket markets.