# Live METAR Data

The project can pull **current airport observations** directly from the free AviationWeather.gov JSON API. No API key is required.

## Quick command

```bash
cd /opt/data/stock-research/dfw_temp_model
.venv/bin/python scripts/fetch_live_metars.py
```

Example output:

```text
Station               Valid (UTC)  Temp (F)  Dewpt (F)  Wind Dir  Wind (kt) Sky
   KACT 2026-06-16 16:51:00+00:00     82.94      71.96       0.0        0.0 BKN
   KADS 2026-06-16 16:47:00+00:00     82.40      69.80     190.0        3.0 SCT
   KAFW 2026-06-16 17:29:00+00:00     84.02      69.98     140.0        7.0 SCT
   KDAL 2026-06-16 16:53:00+00:00     84.02      69.98     120.0        4.0 SCT
   KDFW 2026-06-16 16:53:00+00:00     84.02      69.08       0.0        3.0 SCT
   KDTO 2026-06-16 17:07:00+00:00     82.94      71.96     110.0        4.0 BKN
   KGKY 2026-06-16 16:53:00+00:00     82.04      69.98       0.0        0.0 BKN
   KTYR 2026-06-16 16:53:00+00:00     82.04      71.06       0.0        3.0 FEW

Cached to: data/cache/live_metars.parquet
```

The result is cached to `data/cache/live_metars.parquet`.

## Use in the experiment runner

You can also ask the main experiment script to fetch live METARs:

```bash
.venv/bin/python scripts/run_first_experiment.py \
    --obs-source aviationweather \
    --start-date 2026-06-16 \
    --end-date 2026-06-16 \
    --aviationweather-hours 2
```

It will print the live summary and exit, because AviationWeather.gov only provides recent reports and cannot yet be used for historical daily-high targets.

## Source

- Endpoint: `https://aviationweather.gov/api/data/metar`
- Documentation: `https://aviationweather.gov/data/api`
- Returns: real METAR/SPECI reports from ASOS/AWOS stations
- Update frequency: routine hourly plus SPECI events

## Limitations

- This is **not** the 5-minute Synoptic/MADIS stream. It is the best free, no-key, live airport-observation source available.
- It cannot be used for historical daily-high training because it only keeps a few hours of recent reports.
- The daily high target for backtesting still comes from the IEM ASOS daily summary (`request/daily.py`), which aligns with the NWS climate report.
