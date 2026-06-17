# DFW Temperature Advection Model: Instructions

**Date:** 2026-06-15  
**Ticker / Market:** Polymarket weather markets (KDFW daily temperature)  
**Currency:** USD  
**Data cutoff:** 2026-06-15  
**Disclaimer:** Research design only. No live trading is implemented here. Past model performance on historical data does not guarantee future prediction accuracy or trading profit.

---

## 1. Goal

Build an internal, wind-direction-aware temperature estimation model for the DFW metroplex. The model must know where surrounding stations are relative to KDFW, how far away they are, and whether they sit upwind or downwind. It will use that geometry to weight live observations so that upwind stations influence the target temperature more than downwind stations — the mechanism that lets the model see fronts before they reach KDFW.

---

## 2. Core idea: advected observation-minus-forecast residuals

Treat temperature as a spatial scalar field moving with the wind. Instead of forecasting temperature from scratch, estimate the forecast error at surrounding stations and advect that error toward KDFW.

```
residual_i(t) = observed_temp_i(t) - forecast_temp_i(t)
corrected_KDFW(t) = forecast_KDFW(t) + interpolated_residual_at_KDFW(t)
```

An upwind station’s residual is a preview of the air mass that will reach KDFW after the advection delay.

---

## 3. Station geometry

### 3.1 Local Cartesian grid

Project all lat/lon coordinates to a local tangent plane centered on KDFW so distances and bearings are Euclidean.

```python
import pyproj
proj = pyproj.Proj(proj='aeqd', lat_0=32.897, lon_0=-97.038, units='m')
x, y = proj(lon, lat)  # meters relative to KDFW
```

### 3.2 DFW station network

| Station | ICAO | Lat | Lon | Elev (ft) | Role |
|---|---|---|---|---|---|
| KDFW | target | 32.897 | -97.038 | 607 | Source / resolution station |
| KDAL | neighbor | 32.848 | -96.851 | 487 | Urban Dallas east |
| KADS | neighbor | 33.075 | -96.837 | 645 | North suburban |
| KAFW | neighbor | 32.990 | -97.319 | 679 | NW Fort Worth / exurban |
| KDTO | neighbor | 33.200 | -97.198 | 642 | Denton / rural north |
| KGKY | neighbor | 32.664 | -97.094 | 628 | Arlington / mid-cities |
| KACT | neighbor | 31.611 | -97.230 | 686 | Waco / southern fetch |
| KTYR | neighbor | 32.354 | -95.402 | 550 | Tyler / eastern fetch |

Precompute for every ordered pair:
- `dx`, `dy`, `dist_km`
- `bearing_i_to_j`
- elevation difference
- upwind flag for any given wind direction

---

## 4. Weighting model

For each neighbor station `i` relative to target KDFW:

```
weight_i = distance_weight * upwind_weight * recency_weight * quality_weight
```

| Factor | Formula | Purpose |
|---|---|---|
| **Distance weight** | `1 / dist_i^p` | Closer stations are more representative |
| **Upwind weight** | `1 + upwind_boost * cos(angle_diff)` up to `upwind_boost` | Upwind stations preview arriving air |
| **Recency weight** | `exp(-age / tau)` | Older observations matter less |
| **Quality weight** | 1.0 for ASOS, lower for suspect mesonet | Down-weight noisy sensors |

Parameters to learn:

| Parameter | Symbol | Expected range | How to tune |
|---|---|---|---|
| Distance decay exponent | `p` | 1.0 – 4.0 | Grid search / Optuna |
| Upwind angular half-width | `theta_half` | 30° – 90° | Grid search |
| Upwind boost multiplier | `upwind_boost` | 1.0 – 10.0 | Grid search / log scale |
| Advection decay length | `L_adv` | 20 – 100 km | Grid search |
| Front gradient threshold | `G_front` | 0.05 – 0.20 °F/km | Grid search |
| Front uncertainty multiplier | `sigma_mult` | 1.5 – 3.0 | Grid search |
| Kalman process noise | `Q` | 0.01 – 0.5 | Manual / EM |

---

## 5. Advection time and front handling

### 5.1 Advection delay

If station `i` is `d` km upwind and wind speed is `V` m/s:

```
advection_time_seconds = (d * 1000) / max(V, 0.1)
```

A residual measured upwind now is expected to influence the target after that delay. Use the observation nearest to `target_valid_time - advection_time`.

Apply a decay factor:

```
spatial_decay = exp(-d / L_adv)
```

### 5.2 Residual gradient and front detection

Fit a local plane to neighbor residuals:

```
residual(x, y) ≈ a*x + b*y + c
```

Use weighted least squares with the weights above. The gradient magnitude is:

```
front_strength = sqrt(a^2 + b^2)
```

Rules:
- `front_strength < 1°F per ~40 km`: use smooth interpolation.
- `front_strength >= 3°F per ~40 km`: switch to front-aware mode.

### 5.3 Front-aware mode

1. Cluster residuals into two air masses with k-means (k=2).
2. Identify which cluster is upwind of KDFW using wind direction.
3. Estimate target residual from the upwind cluster mean.
4. Multiply `corrected_sigma` by `sigma_mult` because frontal timing is uncertain.

---

## 6. Kalman smoothing

Track the target residual with a simple Kalman filter to avoid overreacting to single-station noise:

```
state = [residual, trend]
prediction: state_t|t-1 = F * state_t-1 + process_noise
update:   K = P H' / (H P H' + R)
          state_t = state_t|t-1 + K * (measurement - predicted)
```

Set measurement noise `R` larger when `front_detected` is true.

---

## 7. Historical data sources

| Source | URL / API | Resolution | Coverage | Best for | Cost | Caveats |
|---|---|---|---|---|---|---|
| IEM ASOS hourly | `mesonet.agron.iastate.edu/cgi-bin/request/asos.py` | Hourly METARs | Global ASOS/AWOS | Bulk historical airport obs | Free | Hourly for calm periods; SPECI events compressed |
| IEM ASOS 1-minute | `mesonet.agron.iastate.edu/request/asos/1min.phtml` | 1 minute | US ASOS | True sub-hourly swings | Free | Delayed ~18–36 hours; sparse variables |
| NCEI ISD | `ncei.noaa.gov/access/search/data-search/global-hourly` | Hourly | Global | Long standardized history | Free | Hourly only |
| Synoptic Data API | `api.synopticdata.com/v2/stations/timeseries` | 5–15 min typical | US mesonets + airports | Dense non-airport coverage | Free tier + paid | Requires token |
| NOAA HRRR analysis (AWS) | `s3://noaa-hrrr-bdp-pds/` | Hourly, 3 km | CONUS | High-res analysis/forecast fields | Free | Large files; use Herbie |
| Open-Meteo historical | `archive-api.open-meteo.com/v1/archive` | Hourly | Global | Long-term model reanalysis | Free | Lower resolution than HRRR |

**Recommended starting dataset:** IEM ASOS hourly for KDFW + 7 neighbors, 2020–present. Use Open-Meteo or HRRR as the forecast source for residuals.

---

## 8. Data schema

### 8.1 Per-observation table (Parquet or SQLite)

| Column | Meaning |
|---|---|
| `valid_time` | UTC timestamp |
| `valid_date` | UTC date |
| `lead_time_hours` | Forecast lead time |
| `station_id` | ICAO |
| `obs_temp_f` | Observed 2-m temperature (°F) |
| `fcst_temp_f` | Forecast 2-m temperature at station lat/lon |
| `residual_f` | `obs - fcst` |
| `wind_dir_deg` | Wind direction at source |
| `wind_speed_kts` | Wind speed at source |
| `cloud_cover_pct` | Sky cover proxy |
| `solar_elevation_deg` | Solar elevation |
| `hour_local` | Local hour (0–23) |
| `month` | Month |
| `x_m`, `y_m` | Local Cartesian coords |
| `dist_to_target_km` | Distance to KDFW |
| `bearing_to_target_deg` | Bearing to KDFW |
| `elevation_diff_m` | Elevation minus KDFW |

### 8.2 Target prediction table

| Column | Meaning |
|---|---|
| `valid_date` | UTC date of predicted daily high |
| `lead_time_hours` | Forecast lead time |
| `kdfw_obs_high_f` | Observed daily high at KDFW |
| `kdfw_fcst_high_f` | Raw model high at KDFW |
| `residual_target_f` | `obs - fcst` at KDFW |
| neighbor residual columns | Residuals from each neighbor |
| `predicted_residual_f` | Model output residual |
| `corrected_temp_f` | `fcst + predicted_residual` |
| `corrected_sigma_f` | Corrected uncertainty |
| `front_detected` | Boolean |

---

## 9. Pipeline steps

1. **Fetch observations** from IEM ASOS for all 8 stations.
2. **Compute daily highs** from hourly or 1-minute data.
3. **Fetch forecasts** at each station lat/lon from Open-Meteo or HRRR for matching valid dates and lead times.
4. **Align** observations and forecasts on `(station_id, valid_date, lead_time_hours)`.
5. **Compute residuals** and add geometry features.
6. **Build target dataset** by collecting neighbor residuals around each KDFW prediction event.
7. **Train / validate / test** using time-based walk-forward splits.
8. **Tune parameters** with grid search or Optuna.
9. **Evaluate** RMSE, MAE, bucket hit rate, Brier score, front-day RMSE.

---

## 10. Train / validation / test split

**Do not use random splits** — weather is autocorrelated and random splits leak information.

| Split | Dates | Use |
|---|---|---|
| Train | 2020-01-01 to 2023-12-31 | Learn parameters |
| Validation | 2024-01-01 to 2024-12-31 | Tune hyperparameters, detect overfit |
| Test / holdout | 2025-01-01 to present | Final unbiased performance estimate |

Recommended: rolling walk-forward validation, refitting every 6 months.

---

## 11. Evaluation metrics

| Metric | Purpose |
|---|---|
| RMSE of corrected high temp | Overall accuracy |
| MAE of corrected high temp | Robust accuracy |
| Bucket hit rate | Fraction of days predicted bucket matches observed |
| Brier score | Probability calibration |
| Front-day RMSE | Accuracy on high-gradient days |
| UHI-hour RMSE | Accuracy during peak UHI periods |
| Bias by wind direction | Detect weaknesses in specific flow regimes |

---

## 12. Code architecture

```
dfw_temp_model/
├── data/
│   ├── fetch_iem_asos.py        # bulk download from IEM
│   ├── fetch_openmeteo.py       # forecast / reanalysis download
│   ├── build_dataset.py         # merge obs + fcst, compute residuals + geometry
│   └── cache/                   # raw and processed Parquet files
├── features/
│   ├── geometry.py              # lat/lon -> local x/y, bearings, distances
│   ├── advection.py             # upwind detection, advection time
│   └── residuals.py             # plane-fit gradients, front detection
├── models/
│   ├── physics_nowcaster.py     # weighted residual interpolation + front logic
│   └── ml_residual.py           # LightGBM / XGBoost regressor on features
├── training/
│   ├── walkforward_split.py     # time-based splits
│   ├── tune_physics.py          # grid search / Optuna
│   └── evaluate.py              # metrics
└── config/
    └── dfw_stations.yaml        # station metadata + learned biases
```

---

## 13. First concrete experiment

1. Download IEM ASOS hourly data for KDFW + 7 neighbors for 2020–2024.
2. Download Open-Meteo historical hourly forecasts for those stations and dates.
3. Compute daily observed highs and forecast highs.
4. Build the merged dataset.
5. Run baseline: `corrected = KDFW_forecast + inverse-distance-weighted neighbor residual`.
6. Run advection model: add upwind boost + advection time + front detection.
7. Compare RMSE and bucket hit rate on a 2024 holdout.

That experiment directly measures how influential wind advection and temperature advection are versus simple spatial interpolation.

---

## 14. Known limitations

- **Temporal resolution:** Most ASOS report routine METARs hourly. SPECI triggers add events but are not uniformly 5-minute. Use IEM 1-minute or Synoptic mesonet if higher cadence is needed.
- **Representativeness:** Airport ASOS sensors sit on tarmac and may not match a city-center climate signal.
- **Front timing:** A cold front between two stations can make interpolation fail if the model misidentifies which air mass is upwind.
- **Nighttime UHI:** UHI is strongest at night; daily low markets need stronger bias correction than daily highs.
- **Data outages:** Missing stations should drop to zero weight, not break the pipeline.

---

## 15. Why this beats simple interpolation

| Scenario | Simple interpolation | Advection model |
|---|---|---|
| Cold front passed KAFW but not KDFW | Blends warm + cold into a mushy average | Uses upwind KAFW residual to lower KDFW forecast |
| Front stalled between stations | Silent failure | Detects high gradient, widens uncertainty |
| Uniform residual field | Works fine | Works fine, no harm |
| Light/no wind | Works fine | Falls back to spatial interpolation |
| Strong unusual wind direction | Uses wrong nearest neighbors | Reorients upwind sector automatically |

---

## 16. References

- Iowa Environmental Mesonet ASOS download: https://mesonet.agron.iastate.edu/request/download.phtml
- IEM ASOS 1-minute archive: https://mesonet.agron.iastate.edu/request/asos/1min.phtml
- IEM Python examples: https://github.com/akrherz/iem/tree/main/scripts/asos
- NCEI Integrated Surface Dataset: https://www.ncei.noaa.gov/access/search/data-search/global-hourly
- Synoptic Data API docs: https://docs.synoptic.com/s/rest/api-reference
- NOAA HRRR AWS Open Data: s3://noaa-hrrr-bdp-pds/
- Open-Meteo historical API: https://archive-api.open-meteo.com/v1/archive
- pyproj local projections: https://pyproj4.github.io/pyproj/stable/
- Optuna hyperparameter tuning: https://optuna.org/

---

*End of instructions.*
