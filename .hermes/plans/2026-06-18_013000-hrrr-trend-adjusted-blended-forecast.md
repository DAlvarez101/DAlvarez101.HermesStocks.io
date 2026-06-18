# HRRR Trend-Adjusted Blended Forecast Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add a model-trend correction to the existing bias-corrected forecast so the final blended forecast incorporates both (a) the real-world METAR bias and (b) a slight pull toward the direction recent HRRR cycles are trending at each future valid hour.

**Architecture:** A new `compute_trend_correction()` function in `blending/bias.py` computes, for each valid hour in the target cycle, the weighted linear slope of forecast temperature vs. cycle age across recent cycles. The orchestrator in `blend.py` adds this trend correction (scaled by a small weight) to the existing METAR bias correction. The dashboard chart adds a new trace for the trend-adjusted forecast. No new dependencies.

**Tech Stack:** Python 3.13, pandas, numpy (already installed), plotly (already installed), SQLite (stdlib). No new dependencies.

---

## Current Context

### What exists now

The blending subpackage at `dfw_temp_model/blending/` has:
- `providers.py` — `ForecastProvider` protocol + `HRRRProvider`
- `bias.py` — `compute_rolling_bias()` (EWMA of obs - fcst) + `apply_bias_correction()` (adds bias to raw forecast)
- `blend.py` — `blended_forecast()` orchestrator: loads forecast, loads METAR, computes rolling bias, applies correction
- `scripts/generate_dashboard.py` — `blended_forecast_chart()` function with dropdown, METAR overlay, uncertainty band

### The data

The DB has 10 complete 18-hour HRRR cycles for KDAL. For the latest cycle (init 00Z), each forecast hour has 1-10 previous cycles that also forecast that same valid_dt. For example, f01 (valid 01Z) has 10 cycles forecasting it, with temps ranging from 85.6 to 86.2°F. This is the data the trend correction will use.

### The technique

"Model trend extrapolation" — for each valid hour, look at how the forecast temperature has changed across successive model runs. If newer runs are consistently cooler, apply a slight downward adjustment. This captures information the METAR bias can't: the model's own evolving understanding.

### Design constraints

1. **Mimic existing code patterns** — the bias.py and blend.py functions are the template. Same style, same docstring format, same return types.
2. **Slight bias** — the user said "very slight." Default `trend_weight=0.15` (15% of the trend slope applied as correction).
3. **Graceful degradation** — if fewer than 2 previous cycles cover a valid hour, return 0 trend for that hour.
4. **No disruption** — additive only. Existing `blended_forecast()` signature gains optional params with defaults that preserve current behavior.
5. **Simple and robust** — straightforward pandas operations, no clever tricks.

### How the trend is computed

For each valid_dt in the target cycle:
1. Query all cycles that have a forecast at that valid_dt
2. Sort by init_dt (oldest to newest)
3. Assign weights: newer cycles weigh more (exponential decay, same halflife concept as the bias)
4. Compute the weighted linear slope of tmpf vs. cycle age (in hours)
5. The trend correction = slope * trend_weight

Example: valid 06Z has forecasts from 5 cycles. Temps (oldest to newest): 78.4, 78.4, 78.2, 78.7, 78.7. The slope is slightly positive (warming) in recent cycles. With trend_weight=0.15, the correction would be small and positive.

---

## Task 1: Add compute_trend_correction to bias.py

**Objective:** Create the function that computes a per-valid-hour trend correction from multiple forecast cycles.

**Files:**
- Modify: `dfw_temp_model/blending/bias.py` (add one function ~50 lines)
- Test: `tests/test_blending_bias.py` (add 3 test functions)

**Step 1: Write failing tests**

Add these tests to `tests/test_blending_bias.py`:

```python
def test_compute_trend_correction_basic():
    """Trend correction is positive when newer cycles are warmer."""
    # 3 cycles, each 2 hours apart, warming trend at valid hour 10Z
    cycles_df = pd.DataFrame({
        "valid_dt": ["2026-06-17T10:00:00Z"] * 3,
        "init_dt": [
            "2026-06-17T04:00:00Z",
            "2026-06-17T06:00:00Z",
            "2026-06-17T08:00:00Z",
        ],
        "tmpf": [78.0, 79.0, 80.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert "valid_dt" in result.columns
    assert "trend_correction" in result.columns
    # Slope = +1.0 deg F per 2-hour cycle step. Correction = 1.0 * 0.15 = 0.15
    assert result["trend_correction"].iloc[0] == pytest.approx(0.15, abs=0.02)


def test_compute_trend_correction_cooling():
    """Trend correction is negative when newer cycles are cooler."""
    cycles_df = pd.DataFrame({
        "valid_dt": ["2026-06-17T10:00:00Z"] * 3,
        "init_dt": [
            "2026-06-17T04:00:00Z",
            "2026-06-17T06:00:00Z",
            "2026-06-17T08:00:00Z",
        ],
        "tmpf": [82.0, 81.0, 80.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert result["trend_correction"].iloc[0] == pytest.approx(-0.15, abs=0.02)


def test_compute_trend_correction_single_cycle():
    """Only one cycle available -> zero trend correction."""
    cycles_df = pd.DataFrame({
        "valid_dt": ["2026-06-17T10:00:00Z"],
        "init_dt": ["2026-06-17T08:00:00Z"],
        "tmpf": [80.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert result["trend_correction"].iloc[0] == pytest.approx(0.0, abs=0.01)


def test_compute_trend_correction_multiple_valid_hours():
    """Trend correction computed independently for each valid hour."""
    cycles_df = pd.DataFrame({
        "valid_dt": [
            "2026-06-17T10:00:00Z", "2026-06-17T10:00:00Z", "2026-06-17T10:00:00Z",
            "2026-06-17T11:00:00Z", "2026-06-17T11:00:00Z", "2026-06-17T11:00:00Z",
        ],
        "init_dt": [
            "2026-06-17T04:00:00Z", "2026-06-17T06:00:00Z", "2026-06-17T08:00:00Z",
            "2026-06-17T04:00:00Z", "2026-06-17T06:00:00Z", "2026-06-17T08:00:00Z",
        ],
        "tmpf": [78.0, 79.0, 80.0, 76.0, 75.0, 74.0],
    })
    target_init = "2026-06-17T08:00:00Z"
    result = compute_trend_correction(cycles_df, target_init, trend_weight=0.15)
    assert len(result) == 2
    # Hour 10Z: warming trend -> positive correction
    h10 = result[result["valid_dt"] == "2026-06-17T10:00:00Z"]
    assert h10["trend_correction"].iloc[0] > 0
    # Hour 11Z: cooling trend -> negative correction
    h11 = result[result["valid_dt"] == "2026-06-17T11:00:00Z"]
    assert h11["trend_correction"].iloc[0] < 0
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_bias.py -v -k "trend"`
Expected: FAIL — `ImportError: cannot import name 'compute_trend_correction'`

**Step 3: Write implementation**

Add to `dfw_temp_model/blending/bias.py` (after `apply_bias_correction`):

```python
def compute_trend_correction(
    cycles_df: pd.DataFrame,
    target_init: str,
    trend_weight: float = 0.15,
    halflife_hours: float = 6.0,
) -> pd.DataFrame:
    """Compute a per-valid-hour trend correction from multiple forecast cycles.

    For each valid hour, fits a weighted linear slope of forecast temperature
    vs. cycle age (hours before the target cycle). Newer cycles weigh more
    (exponential decay). The trend correction is ``slope * trend_weight``.

    Parameters
    ----------
    cycles_df : pd.DataFrame
        Must have columns ``valid_dt`` (str, ISO datetime), ``init_dt``
        (str, ISO datetime), and ``tmpf`` (float). Contains forecasts from
        multiple cycles for one or more valid hours.
    target_init : str
        The init_dt of the target cycle (the one being corrected).
    trend_weight : float
        Fraction of the raw slope to apply as correction. 0.15 = 15%.
    halflife_hours : float
        Half-life for exponential weighting of cycles by age. Newer cycles
        weigh more.

    Returns
    -------
    pd.DataFrame
        Columns: ``valid_dt`` (str), ``trend_correction`` (float, degrees F),
        ``n_cycles`` (int, number of cycles used).
    """
    if cycles_df.empty:
        return pd.DataFrame(columns=["valid_dt", "trend_correction", "n_cycles"])

    df = cycles_df.copy()
    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
    df["init_dt"] = pd.to_datetime(df["init_dt"], utc=True)
    target_ts = pd.to_datetime(target_init, utc=True)

    # Age of each cycle in hours (how old relative to target)
    df["cycle_age_h"] = (target_ts - df["init_dt"]).dt.total_seconds() / 3600.0

    results = []
    for vdt, group in df.groupby("valid_dt"):
        group = group.sort_values("cycle_age_h")  # newest first (age=0 is target)
        if len(group) < 2:
            results.append({
                "valid_dt": vdt.isoformat(),
                "trend_correction": 0.0,
                "n_cycles": len(group),
            })
            continue

        # Exponential weights: newer cycles (smaller age) weigh more.
        ages = group["cycle_age_h"].values
        # Weight = 2^(-age / halflife), so a cycle one halflife old has half weight
        weights = np.power(2.0, -ages / halflife_hours)
        weights = weights / weights.sum()

        # Weighted linear regression: tmpf = a * age + b
        # Slope tells us how much the forecast changes per hour of cycle age.
        x = ages
        y = group["tmpf"].values
        w_mean_x = np.average(x, weights=weights)
        w_mean_y = np.average(y, weights=weights)
        cov_xy = np.average((x - w_mean_x) * (y - w_mean_y), weights=weights)
        var_x = np.average((x - w_mean_x) ** 2, weights=weights)
        slope = cov_xy / var_x if var_x > 0 else 0.0

        # Positive slope = older cycles were warmer, newer are cooler -> cooling trend
        # We want correction in the direction the model is trending.
        # If slope > 0 (older=warm, newer=cool), the model is trending cooler,
        # so we should adjust downward: correction = -slope * weight.
        # If slope < 0 (older=cool, newer=warm), the model is trending warmer,
        # so we should adjust upward: correction = -slope * weight.
        correction = -slope * trend_weight

        results.append({
            "valid_dt": vdt.isoformat(),
            "trend_correction": round(correction, 4),
            "n_cycles": len(group),
        })

    return pd.DataFrame(results)
```

**Step 4: Run tests to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_bias.py -v -k "trend"`
Expected: 4 passed

**Step 5: Run full bias test suite**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_bias.py -v`
Expected: 8 passed (4 existing + 4 new)

**Step 6: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/blending/bias.py tests/test_blending_bias.py
git commit -m "feat: add compute_trend_correction for multi-cycle model trend extrapolation"
```

---

## Task 2: Wire trend correction into blended_forecast

**Objective:** Modify the orchestrator to compute and apply the trend correction alongside the existing METAR bias.

**Files:**
- Modify: `dfw_temp_model/blending/blend.py` (add trend computation and application, ~20 lines)
- Test: `tests/test_blending_blend.py` (add 2 tests)

**Step 1: Write failing tests**

Add these tests to `tests/test_blending_blend.py`:

```python
def test_blended_forecast_has_trend_correction():
    """blended_forecast with trend_weight > 0 returns a trend_adjusted column."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(
        conn, "KDAL", provider,
        init_dt="2026-06-17T18:00:00+00:00",
        trend_weight=0.15,
    )
    assert "trend_correction" in result.columns
    assert "tmpf_trend_adjusted" in result.columns
    conn.close()


def test_blended_forecast_trend_weight_zero():
    """When trend_weight=0, trend_adjusted = bias_corrected (no trend change)."""
    conn = _make_db()
    provider = HRRRProvider()
    result = blended_forecast(
        conn, "KDAL", provider,
        init_dt="2026-06-17T18:00:00+00:00",
        trend_weight=0.0,
    )
    # With zero trend weight, trend_adjusted should equal corrected
    assert (result["tmpf_trend_adjusted"] == result["tmpf_corrected"]).all()
    conn.close()
```

**Step 2: Run test to verify failure**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_blend.py -v -k "trend"`
Expected: FAIL — `KeyError: 'trend_correction'` or `KeyError: 'tmpf_trend_adjusted'`

**Step 3: Modify blend.py**

In `dfw_temp_model/blending/blend.py`:

a) Add import:
```python
from dfw_temp_model.blending.bias import (
    apply_bias_correction,
    compute_rolling_bias,
    compute_trend_correction,
)
```

b) Add a new helper function after `_load_forecast_for_matching`:

```python
def _load_all_cycles_for_trend(
    conn: sqlite3.Connection,
    provider: ForecastProvider,
    station: str,
    cycles: list[str],
) -> pd.DataFrame:
    """Load forecast rows from multiple cycles for trend computation.

    Returns a DataFrame with valid_dt, init_dt, and tmpf columns.
    Unlike _load_forecast_for_matching, this keeps init_dt so we can
    compute per-cycle trends at each valid hour.
    """
    frames = []
    for init_dt in cycles:
        df = provider.fetch_forecast(conn, station, init_dt)
        if df.empty:
            continue
        frames.append(df[["valid_dt", "init_dt", "tmpf"]])
    if not frames:
        return pd.DataFrame(columns=["valid_dt", "init_dt", "tmpf"])
    return pd.concat(frames, ignore_index=True)
```

c) Modify `blended_forecast()` signature and body. The new signature:

```python
def blended_forecast(
    conn: sqlite3.Connection,
    station: str,
    provider: ForecastProvider,
    init_dt: str | None = None,
    halflife_hours: float = 6.0,
    uncertainty_multiplier: float = 1.0,
    trend_weight: float = 0.0,
) -> pd.DataFrame:
```

d) Add trend computation and application after the `apply_bias_correction` call (at the end of `blended_forecast`):

```python
    # Apply bias correction
    result = apply_bias_correction(forecast, bias_df, uncertainty_multiplier=uncertainty_multiplier)

    # Compute and apply trend correction if requested
    if trend_weight > 0.0 and len(all_cycles) > 1:
        trend_cycles = _load_all_cycles_for_trend(conn, provider, station, all_cycles)
        trend_df = compute_trend_correction(trend_cycles, init_dt, trend_weight=trend_weight)

        if not trend_df.empty:
            # Merge trend correction onto result by valid_dt
            trend_df["valid_dt"] = pd.to_datetime(trend_df["valid_dt"], utc=True)
            result["valid_dt"] = pd.to_datetime(result["valid_dt"], utc=True)
            result = result.merge(
                trend_df[["valid_dt", "trend_correction", "n_cycles"]],
                on="valid_dt",
                how="left",
            )
            result["trend_correction"] = result["trend_correction"].fillna(0.0)
            result["tmpf_trend_adjusted"] = result["tmpf_corrected"] + result["trend_correction"]
        else:
            result["trend_correction"] = 0.0
            result["tmpf_trend_adjusted"] = result["tmpf_corrected"]
    else:
        result["trend_correction"] = 0.0
        result["tmpf_trend_adjusted"] = result["tmpf_corrected"]

    return result
```

**Step 4: Run tests to verify pass**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_blend.py -v`
Expected: 6 passed (4 existing + 2 new)

**Step 5: Run all blending tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_blending_*.py -v`
Expected: 15 passed (3 provider + 8 bias + 4 blend... wait, 6 blend now = 17 total)

**Step 6: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add dfw_temp_model/blending/blend.py tests/test_blending_blend.py
git commit -m "feat: wire trend correction into blended_forecast orchestrator"
```

---

## Task 3: Add trend-adjusted trace to dashboard chart

**Objective:** Add a fourth trace to the blended forecast chart showing the trend-adjusted forecast alongside the existing raw, bias-corrected, and METAR traces.

**Files:**
- Modify: `scripts/generate_dashboard.py` — `blended_forecast_chart()` function

**Step 1: Modify the chart function**

In `scripts/generate_dashboard.py`, inside `blended_forecast_chart()`:

a) Change the `blended_forecast()` call to pass `trend_weight=0.15`:

Find:
```python
        blended = blended_forecast(conn, TARGET_ICAO, provider, init_dt=cycle_dt)
```
Replace with:
```python
        blended = blended_forecast(conn, TARGET_ICAO, provider, init_dt=cycle_dt, trend_weight=0.15)
```

b) Add a new trace for trend-adjusted after the bias-corrected trace (after line ~402, before the visibility list):

```python
        # Trend-adjusted (bias correction + model trend)
        if "tmpf_trend_adjusted" in blended.columns:
            fig.add_trace(go.Scatter(
                x=blended["valid_dt"],
                y=blended["tmpf_trend_adjusted"],
                mode="lines+markers",
                name=f"Trend-adjusted (cycle {i+1})",
                line={"color": "#a78bfa", "width": 2},
                marker={"size": 5, "color": "#a78bfa"},
                hovertemplate=(
                    f"<b>Trend-adjusted</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>"
                    f"%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>"
                    f"Bias: {bias_val:+.1f}°F + trend<br>"
                    f"Cycle: {init_label} · {init_ct}<extra></extra>"
                ),
                customdata=ct_labels,
                visible=(i == 0),
            ))
```

c) Update the visibility list to account for the 4th trace per cycle. Change:
```python
        n_traces_per_cycle = 3
        visibility = [True] * n_metar  # METAR always on
        for j in range(len(cycles)):
            if j == i:
                visibility.extend([True, True, True])
            else:
                visibility.extend([False, False, False])
```
To:
```python
        visibility = [True] * n_metar  # METAR always on
        for j in range(len(cycles)):
            if j == i:
                visibility.extend([True, True, True, True])
            else:
                visibility.extend([False, False, False, False])
```

d) Update the chart title and subtitle to mention the trend-adjusted line:

Find:
```python
        title=f"Bias-Corrected Forecast — {TARGET_ICAO}<br><sup>HRRR raw (orange) vs corrected (green) vs METAR (blue)</sup>",
```
Replace with:
```python
        title=f"Blended Forecast — {TARGET_ICAO}<br><sup>raw (orange) · bias-corrected (green) · trend-adjusted (purple) · METAR (blue)</sup>",
```

**Step 2: Run the dashboard generator**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python scripts/generate_dashboard.py --db data/cache/db/weather_observations.db --output-dir /tmp/dfw-trend-test 2>&1`

Expected: `Dashboard written to: /tmp/dfw-trend-test/index.html`

**Step 3: Verify the output**

Run: `grep -c 'Trend-adjusted' /tmp/dfw-trend-test/index.html && grep -c 'trend_adjusted' /tmp/dfw-trend-test/index.html && grep -c '#a78bfa' /tmp/dfw-trend-test/index.html`

Expected: All > 0

**Step 4: Run existing dashboard tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/test_generate_dashboard.py -v`
Expected: All passed

**Step 5: Commit**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add scripts/generate_dashboard.py
git commit -m "feat: add trend-adjusted forecast trace (purple) to blended forecast chart"
```

---

## Task 4: Run full pipeline and push to GitHub Pages

**Step 1: Run all non-network tests**

Run: `cd /opt/data/stock-research/dfw_temp_model && .venv/bin/python -m pytest tests/ -q -m "not network and not slow" --tb=short`
Expected: 0 failed

**Step 2: Run the cron script to generate and push**

Run: `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh 2>&1 | tail -15`
Expected: Dashboard generated, DB viewer generated, pushed to GitHub Pages.

**Step 3: Verify the live page**

Wait 30 seconds, then:
Run: `curl -sL https://dalvarez101.github.io/DAlvarez101.HermesStocks.io/dfw-live-dashboard/ | grep -c 'Trend-adjusted'`
Expected: > 0

**Step 4: Commit and push all remaining changes**

```bash
cd /opt/data/stock-research/dfw_temp_model
git add -A
git commit -m "feat: HRRR trend-adjusted blended forecast with model trend extrapolation"
git push origin main
```

---

## Files Changed Summary

| File | Action | Description |
|------|--------|-------------|
| `dfw_temp_model/blending/bias.py` | Modify | Add `compute_trend_correction()` function |
| `dfw_temp_model/blending/blend.py` | Modify | Add `_load_all_cycles_for_trend()`, modify `blended_forecast()` signature and body |
| `tests/test_blending_bias.py` | Modify | Add 4 trend correction tests |
| `tests/test_blending_blend.py` | Modify | Add 2 trend integration tests |
| `scripts/generate_dashboard.py` | Modify | Add trend-adjusted trace to chart, update title |

## What is NOT touched

- `dfw_temp_model/blending/providers.py` — untouched
- `scripts/ingest_live_metars.py` — untouched
- `dfw_temp_model/data/hrrr.py` — untouched
- `dfw_temp_model/storage/obs_db.py` — untouched
- `/opt/data/.hermes/scripts/dfw_live_metar_hourly.sh` — untouched
- All trading code — untouched

## How the trend correction works (for the implementer)

The existing METAR bias correction answers: "How much warmer/cooler is reality vs. what HRRR predicted?" and applies that as a constant offset.

The trend correction answers a different question: "Is the HRRR model itself getting warmer or cooler across its recent runs at this future hour?" and applies a slight pull in that direction.

For example, at valid 06Z:
- Cycle 18Z (10h old) forecast 78.4°F
- Cycle 20Z (8h old) forecast 78.2°F
- Cycle 22Z (6h old) forecast 78.7°F
- Cycle 00Z (0h old, target) forecast 78.7°F

The weighted slope is slightly positive (warming in recent cycles). With trend_weight=0.15, the correction is about +0.1°F. So the trend-adjusted forecast = bias_corrected + 0.1°F.

This is intentionally a small effect. The user said "very slight." The trend_weight parameter controls how much of the raw slope is applied.

## Risks and Mitigations

1. **Overcorrection** — The trend_weight is small (0.15) by default and can be set to 0.0 to disable entirely. The trend correction is additive on top of the METAR bias, not multiplicative, so it can't amplify.

2. **Few cycles at later forecast hours** — f17-f18 may only have 1 cycle. The function returns 0.0 trend when fewer than 2 cycles are available, so those hours get no trend correction. This is correct behavior.

3. **Existing behavior preserved** — `trend_weight` defaults to 0.0 in `blended_forecast()`, so all existing callers that don't pass it get the same result as before. Only the dashboard chart passes `trend_weight=0.15`.

## Open Questions

- Should the trend_weight be configurable via environment variable for the dashboard? Not yet — 0.15 is a reasonable default and can be changed in the chart function if needed.
- Should the trading bot's signal.py use the trend-adjusted forecast? Eventually yes, but that's a separate task. The trading bot currently calls `forecast_high_temp()` from `signal.py`, which doesn't use the blending pipeline yet.