# Correction Stats Dashboard Tiles Implementation Plan

> **For Hermes:** Use the `high-reliability-implementation-workflows` skill to implement this plan task-by-task. That workflow combines TDD subagent delegation, parallel verification, red-team review, and smoke testing.

**Goal:** Add per-cycle correction stat tiles and a bias decomposition table to the blended forecast chart so the user can fact-check the bias/trend math and explore historical HRRR runs with hindsight.

**Architecture:** Pre-compute all 5 cycles' stats and bias traces at render time in Python. Replace the Plotly built-in dropdown with a native HTML `<select>` that toggles (a) stat tile visibility via CSS, (b) a collapsible bias decomposition table via CSS, and (c) Plotly trace visibility via `Plotly.restyle()`. No database changes. No new JS frameworks — just a small inline `<script>`.

**Tech Stack:** Python (pandas, plotly), HTML/CSS, vanilla JavaScript, existing SQLite DB

---

## Current Context

### What exists

- `scripts/generate_dashboard.py` — `blended_forecast_chart(conn)` builds a Plotly chart with 5 cycle trace-sets (raw HRRR, uncertainty band, corrected, trend-adjusted) toggled by a Plotly `updatemenus` dropdown. All traces are pre-rendered; the dropdown only toggles `visible`.
- `dfw_temp_model/blending/blend.py` — `blended_forecast()` computes bias correction + optional trend correction. Returns a DataFrame with `tmpf`, `tmpf_corrected`, `bias_applied`, `trend_correction`, `tmpf_trend_adjusted`, `uncertainty_low/high`, `forecast_hour`, `valid_dt`. Currently does NOT return the bias trace (the per-hour EWMA evolution).
- `dfw_temp_model/blending/bias.py` — `compute_rolling_bias()` returns `valid_hour, bias, bias_std, n_matches`. Internally computes `error_mean`, `error_std`, `obs` and `fcst` per hour but does NOT return `error_mean`, `obs_mean`, or `fcst_mean`.
- `halflife_hours` default changed from 6.0 to 2.0 (previous session).
- `trend_weight=0.15` hardcoded in the dashboard call (line 360).
- The existing test `test_generate_dashboard_creates_html` runs the full dashboard as a subprocess against a fixture DB with 1 HRRR cycle (18 hours) and 3 METAR obs.

### Key design decisions

1. **Replace Plotly dropdown with HTML dropdown** — Plotly's `updatemenus` cannot update DOM elements outside the chart. An HTML `<select>` with `onchange` can both call `Plotly.restyle()` and update tiles/tables.
2. **Pre-render all cycles' stats as hidden HTML divs** — No JSON marshaling or dynamic JS text updates. Each cycle gets a `<div class="cycle-stats" id="stats-N">` block. CSS `display:none/block` toggles them. Simple and debuggable.
3. **Bias decomposition table** — Per-hour table showing `METAR obs`, `HRRR fcst`, `error`, `EWMA bias` for each matched hour. Wrapped in a `<details>` element (collapsible) to keep the dashboard minimal per user preference.
4. **Hindsight tiles** — When viewing an old HRRR cycle, show "Corrected at latest obs hour" vs "Latest observed" so the user can see how close the correction was using current observations. This already works in the math (bias uses all recent obs); the tiles just make it visible.
5. **Plotly div ID** — Use `pyo.plot(..., div_id="blended-chart")` so the JS can call `Plotly.restyle("blended-chart", ...)`.

### Files to change

| File | Change |
|---|---|
| `dfw_temp_model/blending/bias.py` | Add `error_mean`, `obs_mean`, `fcst_mean` to `compute_rolling_bias` output |
| `dfw_temp_model/blending/blend.py` | Add `return_bias_trace=False` param to `blended_forecast` |
| `scripts/generate_dashboard.py` | Refactor `blended_forecast_chart`: HTML dropdown, stat tiles, bias table, JS sync |
| `tests/test_blending_bias.py` | Add assertions for new columns |
| `tests/test_blending_blend.py` | Add test for `return_bias_trace` |
| `tests/test_generate_dashboard.py` | Add assertions for stat tiles, dropdown, bias table |

### Files NOT changed

- Database schema (no new tables, columns, or rows)
- `dfw_temp_model/blending/providers.py`
- `dfw_temp_model/data/` (HRRR fetch, NWS API, etc.)
- Any cron scripts

---

## Task 1: Add detail columns to compute_rolling_bias

**Objective:** Return per-hour `error_mean`, `obs_mean`, and `fcst_mean` alongside the existing `bias`, `bias_std`, `n_matches` so the dashboard can render a bias decomposition table.

**Files:**
- Modify: `dfw_temp_model/blending/bias.py:44-45` (empty return columns), `bias.py:59-63` (agg call), `bias.py:79` (return statement), `bias.py:38-42` (docstring)
- Test: `tests/test_blending_bias.py`

### Step 1: Write failing test

Add to `tests/test_blending_bias.py`:

```python
def test_compute_rolling_bias_returns_detail_columns():
    """compute_rolling_bias returns error_mean, obs_mean, fcst_mean for the bias table."""
    obs = pd.DataFrame({
        "valid_hour": pd.to_datetime([
            "2026-06-17T18:00:00Z",
            "2026-06-17T19:00:00Z",
        ], utc=True),
        "tmpf_obs": [89.0, 90.0],
    })
    fcst = pd.DataFrame({
        "valid_hour": pd.to_datetime([
            "2026-06-17T18:00:00Z",
            "2026-06-17T19:00:00Z",
        ], utc=True),
        "tmpf_fcst": [88.0, 89.0],
    })
    result = compute_rolling_bias(obs, fcst, halflife_hours=6.0)
    assert "error_mean" in result.columns
    assert "obs_mean" in result.columns
    assert "fcst_mean" in result.columns
    assert result["error_mean"].iloc[0] == pytest.approx(1.0, abs=0.01)
    assert result["obs_mean"].iloc[0] == pytest.approx(89.0, abs=0.01)
    assert result["fcst_mean"].iloc[0] == pytest.approx(88.0, abs=0.01)
```

### Step 2: Run test to verify failure

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_blending_bias.py::test_compute_rolling_bias_returns_detail_columns -v`
Expected: FAIL — `KeyError: 'error_mean'` or `assert "error_mean" in result.columns` fails

### Step 3: Implement

In `dfw_temp_model/blending/bias.py`:

**Line 44-45** — Add new columns to empty return:

```python
# OLD:
return pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches"])
# NEW:
return pd.DataFrame(columns=["valid_hour", "bias", "bias_std", "n_matches",
                               "error_mean", "obs_mean", "fcst_mean"])
```

**Line 59-63** — Add `obs_mean` and `fcst_mean` to the `agg()` call:

```python
# OLD:
hourly = merged.groupby("valid_hour").agg(
    error_mean=("error", "mean"),
    error_std=("error", "std"),
    n=("error", "count"),
).reset_index()

# NEW:
hourly = merged.groupby("valid_hour").agg(
    error_mean=("error", "mean"),
    error_std=("error", "std"),
    n=("error", "count"),
    obs_mean=("tmpf_obs", "mean"),
    fcst_mean=("tmpf_fcst", "mean"),
).reset_index()
```

**Line 79** — Add new columns to return:

```python
# OLD:
return hourly[["valid_hour", "bias", "bias_std", "n_matches"]]
# NEW:
return hourly[["valid_hour", "bias", "bias_std", "n_matches",
               "error_mean", "obs_mean", "fcst_mean"]]
```

**Docstring (lines 38-42)** — Update the Returns section:

Add after `n_matches` line:
```
        ``error_mean`` (float, raw mean of obs-fcst at that hour before EWMA),
        ``obs_mean`` (float, mean observed temp at that hour),
        ``fcst_mean`` (float, mean forecast temp at that hour).
```

### Step 4: Run tests to verify pass

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_blending_bias.py -v`
Expected: All tests PASS (existing tests unaffected since new columns are additive)

### Step 5: Commit

```bash
git add dfw_temp_model/blending/bias.py tests/test_blending_bias.py
git commit -m "feat: add error_mean, obs_mean, fcst_mean to compute_rolling_bias output"
```

---

## Task 2: Add return_bias_trace to blended_forecast

**Objective:** Allow the dashboard to get the per-hour bias trace (EWMA evolution) alongside the corrected forecast, without duplicating the matching logic.

**Files:**
- Modify: `dfw_temp_model/blending/blend.py:94-184` (function signature + return logic)
- Test: `tests/test_blending_blend.py`

### Step 1: Write failing test

Add to `tests/test_blending_blend.py`:

```python
def test_blended_forecast_return_bias_trace():
    """blended_forecast with return_bias_trace=True returns (result, bias_df)."""
    conn = _make_db()
    provider = HRRRProvider()
    result, bias_df = blended_forecast(
        conn, "KDAL", provider,
        init_dt="2026-06-17T18:00:00+00:00",
        return_bias_trace=True,
    )
    assert "tmpf_corrected" in result.columns
    assert "bias" in bias_df.columns
    assert "error_mean" in bias_df.columns
    assert "obs_mean" in bias_df.columns
    assert "fcst_mean" in bias_df.columns
    assert len(bias_df) > 0
    conn.close()
```

### Step 2: Run test to verify failure

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_blending_blend.py::test_blended_forecast_return_bias_trace -v`
Expected: FAIL — `TypeError: blended_forecast() got an unexpected keyword argument 'return_bias_trace'`

### Step 3: Implement

In `dfw_temp_model/blending/blend.py`:

**Line 94-102** — Add parameter to signature:

```python
def blended_forecast(
    conn: sqlite3.Connection,
    station: str,
    provider: ForecastProvider,
    init_dt: str | None = None,
    halflife_hours: float = 2.0,
    uncertainty_multiplier: float = 1.0,
    trend_weight: float = 0.0,
    return_bias_trace: bool = False,
) -> pd.DataFrame:
```

**Line 182-184** — Change return logic:

```python
# OLD:
    return result

# NEW:
    if return_bias_trace:
        return result, bias_df
    return result
```

Update the docstring Returns section to note the optional tuple return:
```
    If ``return_bias_trace`` is True, returns a tuple ``(result, bias_df)``
    where ``bias_df`` is the per-hour bias trace from ``compute_rolling_bias``
    (includes ``valid_hour``, ``bias``, ``bias_std``, ``n_matches``,
    ``error_mean``, ``obs_mean``, ``fcst_mean``).
```

### Step 4: Run tests to verify pass

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_blending_blend.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add dfw_temp_model/blending/blend.py tests/test_blending_blend.py
git commit -m "feat: add return_bias_trace option to blended_forecast"
```

---

## Task 3: Add _compute_cycle_stats helper

**Objective:** Extract per-cycle stat tile values from the blended forecast result and bias trace. Pure function, easily testable.

**Files:**
- Modify: `scripts/generate_dashboard.py` (add new function after `blended_forecast_chart` or before it)
- Test: `tests/test_generate_dashboard.py`

### Step 1: Write failing test

Add to `tests/test_generate_dashboard.py`:

```python
def test_compute_cycle_stats():
    """_compute_cycle_stats extracts tile values from blended result + bias trace."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import pandas as pd

    blended = pd.DataFrame({
        "valid_dt": pd.to_datetime(["2026-06-19T13:00:00Z", "2026-06-19T14:00:00Z"], utc=True),
        "forecast_hour": [1, 2],
        "tmpf": [80.2, 78.8],
        "tmpf_corrected": [74.7, 73.2],
        "bias_applied": [-5.5, -5.5],
        "trend_correction": [0.04, -0.02],
        "tmpf_trend_adjusted": [74.7, 73.2],
        "uncertainty_low": [71.0, 69.5],
        "uncertainty_high": [78.4, 76.9],
    })
    bias_df = pd.DataFrame({
        "valid_hour": pd.to_datetime(["2026-06-19T12:00:00Z", "2026-06-19T13:00:00Z"], utc=True),
        "bias": [-0.06, -1.71],
        "bias_std": [2.86, 1.03],
        "n_matches": [196, 296],
        "error_mean": [-8.13, -10.80],
        "obs_mean": [73.1, 70.7],
        "fcst_mean": [81.3, 81.5],
    })
    latest_obs = pd.Series({"valid": pd.Timestamp("2026-06-19T14:20:00Z", tz="UTC"), "tmpf": 71.6})

    stats = mod._compute_cycle_stats(blended, bias_df, latest_obs, halflife_hours=2.0, trend_weight=0.15)

    assert stats["bias_applied"] == pytest.approx(-5.5, abs=0.01)
    assert stats["trend_min"] == pytest.approx(-0.02, abs=0.01)
    assert stats["trend_max"] == pytest.approx(0.04, abs=0.01)
    assert stats["n_matched_pairs"] == 296
    assert stats["n_matched_hours"] == 2
    assert stats["latest_obs_tmpf"] == pytest.approx(71.6, abs=0.01)
    assert "latest_obs_time" in stats
    assert "corrected_at_obs_hour" in stats
    assert "hindsight_error" in stats
    assert stats["halflife_hours"] == 2.0
    assert stats["trend_weight"] == 0.15
```

### Step 2: Run test to verify failure

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_generate_dashboard.py::test_compute_cycle_stats -v`
Expected: FAIL — `AttributeError: module 'generate_dashboard' has no attribute '_compute_cycle_stats'`

### Step 3: Implement

Add this function to `scripts/generate_dashboard.py`, before `blended_forecast_chart`:

```python
def _compute_cycle_stats(
    blended: pd.DataFrame,
    bias_df: pd.DataFrame,
    latest_obs: pd.Series | None,
    halflife_hours: float,
    trend_weight: float,
) -> dict:
    """Extract per-cycle stat tile values from blended result and bias trace.

    Parameters
    ----------
    blended : pd.DataFrame
        Output of blended_forecast() — must have tmpf, tmpf_corrected,
        bias_applied, trend_correction, uncertainty_low, uncertainty_high,
        valid_dt, forecast_hour.
    bias_df : pd.DataFrame
        Output of compute_rolling_bias() — must have valid_hour, bias,
        bias_std, n_matches, error_mean, obs_mean, fcst_mean.
    latest_obs : pd.Series or None
        Most recent observation row with 'valid' (datetime) and 'tmpf' (float).
        May be None if no observations exist.
    halflife_hours : float
        The EWMA half-life used for bias computation (for display).
    trend_weight : float
        The trend weight used (for display).

    Returns
    -------
    dict with keys:
        bias_applied, trend_min, trend_max, n_matched_pairs, n_matched_hours,
        uncertainty_plus, max_correction, latest_obs_tmpf, latest_obs_time,
        corrected_at_obs_hour, raw_at_obs_hour, hindsight_error, hindsight_raw_error,
        halflife_hours, trend_weight
    """
    bias_applied = float(blended["bias_applied"].iloc[0]) if len(blended) > 0 else 0.0
    trend_min = float(blended["trend_correction"].min()) if "trend_correction" in blended.columns else 0.0
    trend_max = float(blended["trend_correction"].max()) if "trend_correction" in blended.columns else 0.0

    n_matched_pairs = int(bias_df["n_matches"].iloc[-1]) if not bias_df.empty else 0
    n_matched_hours = len(bias_df) if not bias_df.empty else 0

    # Uncertainty: use the first row (constant for all hours since it's the latest bias_std)
    if len(blended) > 0 and "uncertainty_high" in blended.columns:
        unc_width = float(blended["uncertainty_high"].iloc[0] - blended["uncertainty_low"].iloc[0])
        uncertainty_plus = unc_width / 2.0
    else:
        uncertainty_plus = 0.0

    # Max total correction (bias + trend) across all hours
    if "tmpf_trend_adjusted" in blended.columns and len(blended) > 0:
        corrections = blended["tmpf_trend_adjusted"] - blended["tmpf"]
        max_correction = float(corrections.min())  # most negative = largest downward correction
    else:
        max_correction = bias_applied

    # Hindsight: find the forecast hour closest to the latest observation
    latest_obs_tmpf = None
    latest_obs_time = None
    corrected_at_obs_hour = None
    raw_at_obs_hour = None
    hindsight_error = None
    hindsight_raw_error = None

    if latest_obs is not None and len(blended) > 0:
        latest_obs_tmpf = float(latest_obs["tmpf"])
        latest_obs_time = latest_obs["valid"]
        obs_hour = pd.to_datetime(latest_obs["valid"], utc=True).floor("h")
        blended["valid_hour"] = pd.to_datetime(blended["valid_dt"], utc=True).dt.floor("h")
        match = blended[blended["valid_hour"] == obs_hour]
        if not match.empty:
            corrected_at_obs_hour = float(match["tmpf_corrected"].iloc[0])
            raw_at_obs_hour = float(match["tmpf"].iloc[0])
            hindsight_error = corrected_at_obs_hour - latest_obs_tmpf
            hindsight_raw_error = raw_at_obs_hour - latest_obs_tmpf

    return {
        "bias_applied": bias_applied,
        "trend_min": trend_min,
        "trend_max": trend_max,
        "n_matched_pairs": n_matched_pairs,
        "n_matched_hours": n_matched_hours,
        "uncertainty_plus": uncertainty_plus,
        "max_correction": max_correction,
        "latest_obs_tmpf": latest_obs_tmpf,
        "latest_obs_time": latest_obs_time,
        "corrected_at_obs_hour": corrected_at_obs_hour,
        "raw_at_obs_hour": raw_at_obs_hour,
        "hindsight_error": hindsight_error,
        "hindsight_raw_error": hindsight_raw_error,
        "halflife_hours": halflife_hours,
        "trend_weight": trend_weight,
    }
```

### Step 4: Run test to verify pass

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_generate_dashboard.py::test_compute_cycle_stats -v`
Expected: PASS

### Step 5: Commit

```bash
git add scripts/generate_dashboard.py tests/test_generate_dashboard.py
git commit -m "feat: add _compute_cycle_stats helper for dashboard stat tiles"
```

---

## Task 4: Add _build_stat_tiles_html and _build_bias_table_html helpers

**Objective:** Generate the HTML for stat tiles and bias decomposition table per cycle. Pure functions returning HTML strings.

**Files:**
- Modify: `scripts/generate_dashboard.py` (add two new functions)
- Test: `tests/test_generate_dashboard.py`

### Step 1: Write failing test

Add to `tests/test_generate_dashboard.py`:

```python
def test_build_stat_tiles_html():
    """_build_stat_tiles_html returns a div with stat tile cards."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stats = {
        "bias_applied": -5.5,
        "trend_min": -0.19,
        "trend_max": 0.13,
        "n_matched_pairs": 321,
        "n_matched_hours": 6,
        "uncertainty_plus": 3.6,
        "max_correction": -5.7,
        "latest_obs_tmpf": 71.6,
        "latest_obs_time": pd.Timestamp("2026-06-19T14:20:00Z", tz="UTC"),
        "corrected_at_obs_hour": 73.2,
        "raw_at_obs_hour": 78.8,
        "hindsight_error": 1.6,
        "hindsight_raw_error": 7.2,
        "halflife_hours": 2.0,
        "trend_weight": 0.15,
    }
    html = mod._build_stat_tiles_html(stats, cycle_idx=0, visible=True)
    assert 'id="stats-0"' in html
    assert "-5.5" in html
    assert "71.6" in html
    assert "73.2" in html
    assert "1.6" in html
    assert "2.0h" in html
    assert "15%" in html


def test_build_bias_table_html():
    """_build_bias_table_html returns a details element with per-hour rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_dashboard", "scripts/generate_dashboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    bias_df = pd.DataFrame({
        "valid_hour": pd.to_datetime(["2026-06-19T12:00:00Z", "2026-06-19T13:00:00Z"], utc=True),
        "bias": [-0.06, -1.71],
        "bias_std": [2.86, 1.03],
        "n_matches": [196, 296],
        "error_mean": [-8.13, -10.80],
        "obs_mean": [73.1, 70.7],
        "fcst_mean": [81.3, 81.5],
    })
    html = mod._build_bias_table_html(bias_df, cycle_idx=0, visible=True)
    assert 'id="bias-table-0"' in html
    assert "<table" in html
    assert "73.1" in html
    assert "81.3" in html
    assert "-8.1" in html
    assert "<details" in html
```

Add `import pandas as pd` and `import pytest` at the top of the test file if not already there (they are already imported).

### Step 2: Run test to verify failure

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_generate_dashboard.py::test_build_stat_tiles_html tests/test_generate_dashboard.py::test_build_bias_table_html -v`
Expected: FAIL — `AttributeError: module 'generate_dashboard' has no attribute '_build_stat_tiles_html'`

### Step 3: Implement

Add these functions to `scripts/generate_dashboard.py`:

```python
def _format_obs_time(obs_time) -> str:
    """Format an observation timestamp for tile display."""
    if obs_time is None:
        return "—"
    dt = pd.to_datetime(obs_time, utc=True)
    return dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")


def _build_stat_tiles_html(stats: dict, cycle_idx: int, visible: bool) -> str:
    """Build the stat tile HTML block for one cycle.

    Returns a <div> with id="stats-{cycle_idx}" containing tile cards.
    """
    display = "block" if visible else "none"

    bias_val = stats["bias_applied"]
    bias_color = "#f87171" if bias_val < 0 else "#4ade80" if bias_val > 0 else "#94a3b8"

    hindsight_err = stats.get("hindsight_error")
    if hindsight_err is not None:
        err_text = f"{hindsight_err:+.1f}°F"
        err_color = "#4ade80" if abs(hindsight_err) < 2 else "#fbbf24" if abs(hindsight_err) < 4 else "#f87171"
    else:
        err_text = "—"
        err_color = "#94a3b8"

    raw_err = stats.get("hindsight_raw_error")
    raw_err_text = f"{raw_err:+.1f}°F" if raw_err is not None else "—"

    corrected_val = stats.get("corrected_at_obs_hour")
    corrected_text = f"{corrected_val:.1f}°F" if corrected_val is not None else "—"

    raw_val = stats.get("raw_at_obs_hour")
    raw_text = f"{raw_val:.1f}°F" if raw_val is not None else "—"

    latest_tmpf = stats.get("latest_obs_tmpf")
    latest_text = f"{latest_tmpf:.1f}°F" if latest_tmpf is not None else "—"
    latest_time = _format_obs_time(stats.get("latest_obs_time"))

    return f"""<div class="cycle-stats" id="stats-{cycle_idx}" style="display:{display}">
<div class="stats-tiles">
  <div class="stat-tile"><h3>Bias Applied</h3><p style="color:{bias_color}">{bias_val:+.1f}°F</p><small>EWMA constant offset</small></div>
  <div class="stat-tile"><h3>Trend Correction</h3><p>{stats['trend_min']:+.2f} to {stats['trend_max']:+.2f}°F</p><small>Per-hour range</small></div>
  <div class="stat-tile"><h3>Matched Data</h3><p>{stats['n_matched_pairs']}</p><small>pairs across {stats['n_matched_hours']} hours</small></div>
  <div class="stat-tile"><h3>Uncertainty ±</h3><p>±{stats['uncertainty_plus']:.1f}°F</p><small>1-sigma band</small></div>
  <div class="stat-tile"><h3>Max Correction</h3><p style="color:{bias_color}">{stats['max_correction']:+.1f}°F</p><small>Largest total adjustment</small></div>
  <div class="stat-tile"><h3>Config</h3><p>{stats['halflife_hours']:.0f}h / {int(stats['trend_weight']*100)}%</p><small>EWMA half-life / trend weight</small></div>
  <div class="stat-tile"><h3>Latest Observed</h3><p>{latest_text}</p><small>{latest_time}</small></div>
  <div class="stat-tile"><h3>Corrected at Obs Hour</h3><p>{corrected_text}</p><small>Raw: {raw_text}</small></div>
  <div class="stat-tile"><h3>Hindsight Error</h3><p style="color:{err_color}">{err_text}</p><small>Corrected vs actual (raw: {raw_err_text})</small></div>
</div>
</div>"""


def _build_bias_table_html(bias_df: pd.DataFrame, cycle_idx: int, visible: bool) -> str:
    """Build a collapsible bias decomposition table for one cycle.

    Shows per-hour: METAR obs, HRRR fcst, raw error, EWMA bias.
    """
    display = "" if visible else "none"
    if bias_df.empty:
        return f'<details class="cycle-bias-table" id="bias-table-{cycle_idx}" style="display:{display}"><summary>Bias decomposition (no matched data)</summary></details>'

    rows = []
    for _, row in bias_df.iterrows():
        ct_time = row["valid_hour"].tz_convert(_CT).strftime("%m/%d %H:%M CT")
        rows.append(
            f"<tr><td>{ct_time}</td><td>{row['obs_mean']:.1f}</td><td>{row['fcst_mean']:.1f}</td>"
            f"<td>{row['error_mean']:+.1f}</td><td>{row['bias']:+.2f}</td>"
            f"<td>{int(row['n_matches'])}</td></tr>"
        )
    rows_html = "\n".join(rows)

    return f"""<details class="cycle-bias-table" id="bias-table-{cycle_idx}" style="display:{display}">
<summary>Bias decomposition — per-hour matched errors and EWMA evolution</summary>
<table class="bias-table">
<thead><tr><th>Valid Hour (CT)</th><th>METAR Obs °F</th><th>HRRR Fcst °F</th><th>Error °F</th><th>EWMA Bias °F</th><th>Pairs</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</details>"""
```

### Step 4: Run test to verify pass

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_generate_dashboard.py::test_build_stat_tiles_html tests/test_generate_dashboard.py::test_build_bias_table_html -v`
Expected: PASS

### Step 5: Commit

```bash
git add scripts/generate_dashboard.py tests/test_generate_dashboard.py
git commit -m "feat: add stat tile and bias table HTML builders"
```

---

## Task 5: Refactor blended_forecast_chart to use HTML dropdown + tiles + JS

**Objective:** Replace the Plotly built-in dropdown with an HTML `<select>` that controls stat tiles, bias table visibility, and Plotly trace visibility. Pre-compute all cycles' stats at render time.

**Files:**
- Modify: `scripts/generate_dashboard.py:279-486` (rewrite `blended_forecast_chart`)
- Modify: `scripts/generate_dashboard.py:47-127` (add CSS to HTML_TEMPLATE for stat tiles and bias table)
- Test: `tests/test_generate_dashboard.py`

### Step 1: Write failing test

Update `test_generate_dashboard_creates_html` in `tests/test_generate_dashboard.py` to add assertions for new elements:

```python
def test_generate_dashboard_creates_html(populated_db, tmp_path):
    output_dir = tmp_path / "dash"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dashboard.py",
            "--db",
            populated_db,
            "--output-dir",
            str(output_dir),
        ],
        cwd=".",
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, result.stderr
    index_path = output_dir / "index.html"
    assert index_path.exists()
    html = index_path.read_text(encoding="utf-8")
    assert "DFW Live Weather Dashboard" in html
    assert "KDAL" in html
    assert "METAR vs HRRR" in html
    assert "85.5°F" in html
    assert "+1.5°F" in html
    # Two matplotlib base64 chart images remain; HRRR is an interactive Plotly chart.
    assert html.count("data:image/png;base64,") >= 2
    assert "plotly" in html.lower()
    # New: stat tiles and dropdown
    assert 'id="cycle-selector"' in html
    assert "cycle-stats" in html
    assert "stat-tile" in html
    assert "Bias Applied" in html
    assert "Hindsight Error" in html
    assert "bias-table" in html
    assert "switchBlendedCycle" in html
```

### Step 2: Run test to verify failure

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_generate_dashboard.py::test_generate_dashboard_creates_html -v`
Expected: FAIL — `assert 'id="cycle-selector"' in html` fails

### Step 3: Add CSS to HTML_TEMPLATE

In `scripts/generate_dashboard.py`, add these CSS rules to the `HTML_TEMPLATE` `<style>` block (after the existing `.footer` rule, before `a {{ color: ... }}`):

```css
        .cycle-selector-wrap {{ margin: 0.5rem 0 1rem 0; }}
        select#cycle-selector {{ background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 0.5rem 0.75rem; border-radius: 0.5rem; font-size: 0.95rem; }}
        .stats-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin: 0.5rem 0 1rem 0; max-width: 900px; }}
        .stat-tile {{ background: #1e293b; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #334155; }}
        .stat-tile h3 {{ margin: 0 0 0.35rem 0; font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.03em; }}
        .stat-tile p {{ margin: 0; font-size: 1.2rem; font-weight: 600; color: #38bdf8; }}
        .stat-tile small {{ display: block; color: #64748b; margin-top: 0.2rem; font-size: 0.7rem; }}
        .cycle-bias-table {{ margin: 0.5rem 0 1rem 0; max-width: 900px; }}
        .cycle-bias-table summary {{ color: #7dd3fc; cursor: pointer; font-size: 0.85rem; margin: 0.5rem 0; }}
        .bias-table {{ border-collapse: collapse; margin: 0.5rem 0; width: 100%; font-size: 0.82rem; }}
        .bias-table th, .bias-table td {{ border: 1px solid #334155; padding: 0.35rem 0.5rem; text-align: left; }}
        .bias-table th {{ background: #1e293b; color: #94a3b8; }}
        .bias-table tr:nth-child(even) {{ background: #162032; }}
```

### Step 4: Rewrite blended_forecast_chart

Replace the entire `blended_forecast_chart` function (lines 279-486) with the new version. Key changes:

1. Call `blended_forecast(..., return_bias_trace=True)` for each cycle
2. Compute stats per cycle using `_compute_cycle_stats`
3. Get latest obs for hindsight tiles
4. Build HTML `<select>` dropdown with cycle labels
5. Build stat tiles HTML for all cycles (first visible, rest hidden)
6. Build bias table HTML for all cycles
7. Build Plotly chart WITHOUT `updatemenus` (dropdown is now HTML)
8. Use `div_id="blended-chart"` in `pyo.plot()`
9. Pre-compute visibility arrays for JS
10. Return concatenated HTML: dropdown + tiles + tables + chart div + JS script

```python
def blended_forecast_chart(conn) -> str:
    """Interactive chart with per-cycle correction stat tiles and bias decomposition.

    Replaces the Plotly built-in dropdown with an HTML <select> that syncs
    stat tiles, bias table, and Plotly trace visibility.
    """
    from dfw_temp_model.blending.blend import blended_forecast, list_recent_cycles
    from dfw_temp_model.blending.providers import HRRRProvider

    provider = HRRRProvider()
    cycles = list_recent_cycles(conn, TARGET_ICAO, provider, min_hours=18)
    if not cycles:
        return "<p>No complete HRRR forecast cycles available for blending</p>"
    cycles = cycles[:5]

    # Load ALL observations for overlay + latest obs for hindsight tiles
    obs_df = pd.read_sql_query(
        "SELECT valid, tmpf, source FROM metar_observations WHERE station = ? AND tmpf IS NOT NULL ORDER BY valid",
        conn,
        params=[TARGET_ICAO],
    )
    if not obs_df.empty:
        obs_df["valid"] = pd.to_datetime(obs_df["valid"], utc=True)
        obs_df["ct_label"] = obs_df["valid"].apply(
            lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")
        )
        obs_5min = obs_df[obs_df["source"] == "nws-api"].copy()
        obs_hourly = obs_df[obs_df["source"] == "aviationweather"].copy()
        latest_obs = obs_df.iloc[-1]
    else:
        obs_5min = pd.DataFrame()
        obs_hourly = pd.DataFrame()
        latest_obs = None

    n_obs_traces = int(not obs_5min.empty) + int(not obs_hourly.empty)

    fig = go.Figure()

    # Observation traces (always visible)
    if not obs_5min.empty:
        fig.add_trace(go.Scatter(
            x=obs_5min["valid"], y=obs_5min["tmpf"], mode="markers",
            name="5-min obs (NWS API)",
            marker={"size": 4, "color": "#818cf8", "symbol": "circle", "opacity": 0.6},
            hovertemplate="<b>5-min obs</b><br>%{x|%Y-%m-%d %H:%M UTC}<br>%{customdata}<br>Temp: %{y:.1f}°F<extra></extra>",
            customdata=obs_5min.get("ct_label", ""), visible=True,
        ))
    if not obs_hourly.empty:
        fig.add_trace(go.Scatter(
            x=obs_hourly["valid"], y=obs_hourly["tmpf"], mode="markers",
            name="METAR observed",
            marker={"size": 8, "color": "#38bdf8", "symbol": "circle"},
            hovertemplate="<b>METAR</b><br>%{x|%Y-%m-%d %H:%M UTC}<br>%{customdata}<br>Temp: %{y:.1f}°F<extra></extra>",
            customdata=obs_hourly.get("ct_label", ""), visible=True,
        ))

    halflife = 2.0  # matches blended_forecast default
    trend_w = 0.15

    cycle_labels = []
    all_tiles_html = []
    all_table_html = []
    visibility_arrays = []

    for i, cycle_dt in enumerate(cycles):
        blended, bias_df = blended_forecast(
            conn, TARGET_ICAO, provider, init_dt=cycle_dt,
            trend_weight=trend_w, return_bias_trace=True,
        )
        if blended.empty:
            visibility_arrays.append(None)
            all_tiles_html.append("")
            all_table_html.append("")
            cycle_labels.append("")
            continue

        blended["valid_dt"] = pd.to_datetime(blended["valid_dt"], utc=True)
        blended = blended.sort_values("forecast_hour")
        init_ts = pd.to_datetime(cycle_dt, utc=True)
        init_label = init_ts.strftime("%Y-%m-%d %H:%M UTC")
        init_ct = init_ts.tz_convert(_CT).strftime("%I:%M %p CT")

        ct_labels = blended["valid_dt"].apply(
            lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")
        )
        bias_val = float(blended["bias_applied"].iloc[0])

        # Raw HRRR trace
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"], y=blended["tmpf"], mode="lines+markers",
            name=f"HRRR raw (cycle {i+1})",
            line={"color": "#f59e0b", "width": 2, "dash": "dot"},
            marker={"size": 5, "color": "#f59e0b"},
            hovertemplate=f"<b>HRRR raw</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>Cycle: {init_label}<extra></extra>",
            customdata=ct_labels, visible=(i == 0),
        ))
        # Uncertainty band
        fig.add_trace(go.Scatter(
            x=list(blended["valid_dt"]) + list(blended["valid_dt"])[::-1],
            y=list(blended["uncertainty_high"]) + list(blended["uncertainty_low"])[::-1],
            fill="toself", fillcolor="rgba(34, 197, 94, 0.12)",
            line={"color": "rgba(34, 197, 94, 0)", "width": 0},
            name=f"Uncertainty (cycle {i+1})", hoverinfo="skip",
            visible=(i == 0), showlegend=False,
        ))
        # Bias-corrected
        fig.add_trace(go.Scatter(
            x=blended["valid_dt"], y=blended["tmpf_corrected"], mode="lines+markers",
            name=f"Corrected (cycle {i+1})",
            line={"color": "#22c55e", "width": 2.5},
            marker={"size": 6, "color": "#22c55e"},
            hovertemplate=f"<b>Corrected</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>Bias: {bias_val:+.1f}°F<br>Cycle: {init_label} · {init_ct}<extra></extra>",
            customdata=ct_labels, visible=(i == 0),
        ))
        # Trend-adjusted
        has_trend = "tmpf_trend_adjusted" in blended.columns
        if has_trend:
            fig.add_trace(go.Scatter(
                x=blended["valid_dt"], y=blended["tmpf_trend_adjusted"], mode="lines+markers",
                name=f"Trend-adjusted (cycle {i+1})",
                line={"color": "#a78bfa", "width": 2},
                marker={"size": 5, "color": "#a78bfa"},
                hovertemplate=f"<b>Trend-adjusted</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>%{{customdata}}<br>Temp: %{{y:.1f}}°F<br>Bias: {bias_val:+.1f}°F + trend<br>Cycle: {init_label} · {init_ct}<extra></extra>",
                customdata=ct_labels, visible=(i == 0),
            ))

        n_traces_per_cycle = 4 if has_trend else 3

        # Build visibility array for this cycle
        vis = [True] * n_obs_traces
        for j in range(len(cycles)):
            if j == i:
                vis.extend([True] * n_traces_per_cycle)
            else:
                vis.extend([False] * n_traces_per_cycle)
        visibility_arrays.append(vis)

        # Compute stats and build tiles + table
        stats = _compute_cycle_stats(blended, bias_df, latest_obs, halflife, trend_w)
        all_tiles_html.append(_build_stat_tiles_html(stats, cycle_idx=i, visible=(i == 0)))
        all_table_html.append(_build_bias_table_html(bias_df, cycle_idx=i, visible=(i == 0)))
        cycle_labels.append(init_ts.strftime("%m/%d %H:00Z"))

    # Build dropdown options HTML
    options_html = "\n".join(
        f'<option value="{i}">{label}</option>'
        for i, label in enumerate(cycle_labels) if label
    )

    fig.update_layout(
        title=f"Blended Forecast — {TARGET_ICAO}<br><sup>raw (orange) · corrected (green) · trend (purple) · 5-min obs (indigo) · METAR (blue)</sup>",
        xaxis_title="Valid time (UTC)", yaxis_title="Temperature (°F)",
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"}, margin={"l": 60, "r": 30, "t": 60, "b": 60},
        hovermode="x unified", showlegend=True,
        legend={"x": 0.01, "xanchor": "left", "y": 0.99, "yanchor": "top",
                "bgcolor": "rgba(15,23,42,0.8)", "font": {"size": 10}},
    )

    plotly_div = pyo.plot(fig, output_type="div", include_plotlyjs=False,
                          config={"displayModeBar": False}, div_id="blended-chart")

    # Build JS with embedded visibility arrays
    vis_json = str(visibility_arrays).replace("None", "null").replace("'", '"')
    js = f"""<script>
var blendedVisibilityArrays = {vis_json};
function switchBlendedCycle(idx) {{
    idx = parseInt(idx);
    document.querySelectorAll('.cycle-stats').forEach(el => el.style.display = 'none');
    var statsEl = document.getElementById('stats-' + idx);
    if (statsEl) statsEl.style.display = 'block';
    document.querySelectorAll('.cycle-bias-table').forEach(el => el.style.display = 'none');
    var tableEl = document.getElementById('bias-table-' + idx);
    if (tableEl) tableEl.style.display = '';
    var vis = blendedVisibilityArrays[idx];
    if (vis) Plotly.restyle('blended-chart', {{visible: vis}});
}}
</script>"""

    tiles_combined = "\n".join(t for t in all_tiles_html if t)
    tables_combined = "\n".join(t for t in all_table_html if t)

    return f"""<div class="cycle-selector-wrap"><label for="cycle-selector">HRRR cycle: </label><select id="cycle-selector" onchange="switchBlendedCycle(this.value)">{options_html}</select></div>
{tiles_combined}
{tables_combined}
{plotly_div}
{js}"""
```

### Step 5: Run test to verify pass

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/test_generate_dashboard.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add scripts/generate_dashboard.py tests/test_generate_dashboard.py
git commit -m "feat: replace Plotly dropdown with HTML dropdown + stat tiles + bias table"
```

---

## Task 6: Run full test suite + manual verification against live DB

**Objective:** Verify all tests pass and the live dashboard generates correctly with the storm data.

### Step 1: Run full test suite

Run: `/tmp/dfw_venv/bin/python3 -m pytest tests/ -v --timeout=60`
Expected: All tests PASS

### Step 2: Generate dashboard against live DB

Run:
```bash
/tmp/dfw_venv/bin/python3 scripts/generate_dashboard.py \
  --db /opt/data/stock-research/dfw_temp_model/data/cache/db/weather_observations.db \
  --output-dir /tmp/dashboard-test
```

### Step 3: Verify output HTML has stat tiles

Check that `/tmp/dashboard-test/index.html` contains:
- `id="cycle-selector"` — the HTML dropdown
- `id="stats-0"` through `stats-4"` — stat tile blocks for each cycle
- `id="bias-table-0"` through `bias-table-4"` — bias decomposition tables
- `Bias Applied` tile text
- `Hindsight Error` tile text
- `switchBlendedCycle` JS function
- `blended-chart` Plotly div ID

### Step 4: Verify the storm data shows correct hindsight error

Run a quick Python check:
```python
# Verify cycle 0 (latest, 12Z) has hindsight error ~1.6°F
# (corrected 73.2°F vs observed 71.6°F)
```
Expected: The stat tiles for the 12Z cycle should show:
- Bias Applied: ~−5.5°F
- Hindsight Error: ~+1.6°F (corrected vs actual)
- Hindsight raw error: ~+7.2°F (raw HRRR vs actual)

### Step 5: Verify an older cycle shows larger hindsight error

Select cycle 4 (08Z, oldest in dropdown). The HRRR raw at 14Z was 82.8°F, so with the 08Z cycle:
- Raw hindsight error: ~+11.2°F (82.8 vs 71.6)
- Corrected hindsight error should be smaller (bias correction pulls it down)

### Step 6: Commit

```bash
git add -A
git commit -m "test: verify correction stats dashboard against live storm data"
```

---

## Risks and Tradeoffs

### Risks

1. **Plotly `div_id` parameter may not work as expected** — If `pyo.plot()` doesn't support `div_id` with `output_type="div"`, the JS won't find the chart. Mitigation: verify in Task 6 Step 2. If it fails, post-process the returned HTML to inject a known ID, or use `plotly.io.to_html()` instead.

2. **Test fixture has limited data** — The fixture DB has only 1 HRRR cycle and 3 obs (1 overlapping hour). The bias table will have 1 row. This is enough for structural testing but won't test multi-cycle dropdown behavior. The live DB verification (Task 6) covers this.

3. **Visibility array mismatch** — If the number of traces per cycle varies (some cycles have trend, some don't), the visibility arrays must match. The code handles this with `has_trend` check and `n_traces_per_cycle`.

4. **JS `Plotly.restyle` availability** — The blended chart uses `include_plotlyjs=False` (relies on Plotly being loaded by the HRRR chart above it). If the HRRR chart section is removed or Plotly fails to load, `Plotly.restyle` will throw. This is the same dependency as the current code.

### Tradeoffs

- **HTML dropdown vs Plotly dropdown**: The HTML dropdown is more flexible (can update external DOM) but requires custom JS. The Plotly dropdown was simpler but couldn't update tiles/tables. This is the right tradeoff for the user's requirement.
- **Pre-rendered hidden divs vs JSON**: Pre-rendering all cycles' HTML is slightly more bytes in the HTML file but eliminates JSON marshaling complexity. For 5 cycles this is negligible.
- **Bias table inline vs separate page**: The user prefers detailed data on separate pages. The `<details>` collapsible element keeps it inline but collapsed by default, which is a reasonable compromise. If the user wants it on a separate page later, it can be extracted.

### Open questions

1. Should the stat tiles show the raw HRRR hindsight error alongside the corrected one? (Currently planned: yes, as a `<small>` subtitle on the Hindsight Error tile.)
2. Should the bias table be open or collapsed by default? (Currently planned: collapsed via `<details>` without `open` attribute, except for cycle 0.)
3. Should we show the per-hour `bias_std` in the table? (Currently not planned — would add a column. Can be added later if needed.)

---

## Summary

This plan adds 8 stat tiles and a collapsible bias decomposition table per HRRR cycle to the blended forecast chart section. The user can switch between the 5 most recent HRRR cycles via an HTML dropdown, and the tiles + table + chart all update to show that cycle's correction math. This enables hindsight analysis: selecting an older HRRR cycle shows how the bias correction (computed with current observations) would have adjusted that cycle's forecast, and the Hindsight Error tile shows the gap between the corrected forecast and the actual observed temperature.

No database changes. Three source files modified (bias.py, blend.py, generate_dashboard.py). Six tasks, each TDD with failing test first.