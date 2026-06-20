"""Generate a static HTML dashboard from the observation database."""
import argparse
import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo

from dfw_temp_model.config import CACHE_DIR, TARGET_ICAO
from dfw_temp_model.storage.obs_db import (
    get_db,
    hrrr_forecast_for_cycle,
    hrrr_forecast_range,
    latest_by_station,
    latest_complete_hrrr_cycle,
)

# Central Time zone (automatically handles CST/CDT).
_CT = ZoneInfo("America/Chicago")


def _utc_to_ct(utc_str: str, fmt: str = "%Y-%m-%d %H:%M UTC") -> str:
    """Convert a UTC timestamp string to a short Central Time string.

    Returns an empty string if parsing fails.
    """
    try:
        dt = pd.to_datetime(utc_str, utc=True)
    except Exception:
        return ""
    ct = dt.tz_convert(_CT)
    return ct.strftime(fmt)


def _dt_to_ct(dt, fmt: str = "%I:%M %p %Z") -> str:
    """Convert a timezone-aware datetime to a Central Time string."""
    ct = dt.tz_convert(_CT)
    return ct.strftime(fmt)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DFW Live Weather Dashboard</title>
    <style>
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #38bdf8; margin-bottom: 0.25rem; }}
        h2 {{ color: #7dd3fc; margin-top: 2rem; }}
        table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; max-width: 900px; }}
        th, td {{ border: 1px solid #334155; padding: 0.5rem 0.75rem; text-align: left; }}
        th {{ background: #1e293b; }}
        tr:nth-child(even) {{ background: #162032; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0; max-width: 900px; }}
        .card {{ background: #1e293b; padding: 1rem; border-radius: 0.5rem; }}
        .card h3 {{ margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.03em; }}
        .card p {{ margin: 0; font-size: 1.6rem; font-weight: 600; color: #38bdf8; }}
        .comparison {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; max-width: 900px; }}
        .comparison .card p {{ font-size: 1.4rem; }}
        .comparison .card small {{ display: block; color: #94a3b8; margin-top: 0.25rem; font-size: 0.75rem; }}
        img {{ max-width: 100%; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #334155; }}
        .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.85rem; max-width: 900px; }}
        a {{ color: #38bdf8; }}
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
    </style>
</head>
<body>
    <h1>DFW Live Weather Dashboard</h1>
    <p>Updated at {updated_at} UTC · {updated_at_ct} CT<br>Sources: AviationWeather.gov METAR JSON, NOAA HRRR AWS Open Data</p>

    <div class="stats">
        <div class="card"><h3>Total METAR obs</h3><p>{total_rows}</p></div>
        <div class="card"><h3>Stations</h3><p>{station_count}</p></div>
        <div class="card"><h3>First observation</h3><p>{first_obs}</p><small>{first_obs_ct}</small></div>
        <div class="card"><h3>Latest observation</h3><p>{last_obs}</p><small>{last_obs_ct}</small></div>
    </div>

    <h2>METAR vs HRRR at {TARGET_ICAO} — current hour</h2>
    <div class="comparison">
        <div class="card">
            <h3>METAR observed</h3>
            <p>{metar_tmpf}°F</p>
            <small>{metar_valid}<br>{metar_valid_ct}</small>
        </div>
        <div class="card">
            <h3>HRRR forecast</h3>
            <p>{hrrr_tmpf}°F</p>
            <small>Cycle {hrrr_init}<br>f{hrrr_fxx:02d} · valid {hrrr_valid}<br>{hrrr_valid_ct}</small>
        </div>
        <div class="card">
            <h3>Difference</h3>
            <p>{delta_text}</p>
            <small>HRRR minus METAR</small>
        </div>
    </div>

    <h2>HRRR 18-hour forecast ({TARGET_ICAO})</h2>
    {hrrr_chart}

    <h2>Bias-Corrected Forecast ({TARGET_ICAO})</h2>
    {blended_chart}

    <h2>Latest readings per station</h2>
    {latest_table}

    <h2>Temperature trend ({TARGET_ICAO})</h2>
    <img src="data:image/png;base64,{kdfw_chart}" alt="{TARGET_ICAO} temperature trend">

    <h2>Hourly observations ingested</h2>
    <img src="data:image/png;base64,{hourly_chart}" alt="Hourly ingestion volume">

    <div class="footer">
        Database: {db_path}<br>
        Dashboard files live in the GitHub Pages repo and update hourly.
        <br>
        <a href="https://github.com/DAlvarez101/DAlvarez101.HermesStocks.io" target="_blank">View repo →</a>
    </div>
</body>
</html>
"""


def summary_stats(conn) -> dict:
    df = pd.read_sql_query("SELECT * FROM metar_observations", conn)
    if df.empty:
        return {
            "total_rows": 0,
            "station_count": 0,
            "first_obs": "—",
            "last_obs": "—",
            "first_obs_ct": "",
            "last_obs_ct": "",
        }
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    first_dt = df["valid"].min()
    last_dt = df["valid"].max()
    return {
        "total_rows": len(df),
        "station_count": df["station"].nunique(),
        "first_obs": first_dt.strftime("%Y-%m-%d %H:%M UTC"),
        "last_obs": last_dt.strftime("%Y-%m-%d %H:%M UTC"),
        "first_obs_ct": _dt_to_ct(first_dt, "%m/%d %I:%M %p CT"),
        "last_obs_ct": _dt_to_ct(last_dt, "%m/%d %I:%M %p CT"),
    }


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def kdfw_temperature_chart(conn) -> str:
    df = pd.read_sql_query(
        """
        SELECT valid, tmpf FROM metar_observations
        WHERE station = ? ORDER BY valid
        """,
        conn,
        params=[TARGET_ICAO],
    )
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    if df.empty:
        ax.text(0.5, 0.5, "No KDFW data yet", ha="center", va="center", color="#e2e8f0")
        return _to_base64(fig)
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    ax.plot(df["valid"], df["tmpf"], color="#38bdf8", linewidth=1.5)
    ax.set_title(f"{TARGET_ICAO} Temperature", color="#e2e8f0")
    ax.set_ylabel("Temperature (°F)", color="#e2e8f0")
    ax.set_xlabel("UTC", color="#e2e8f0")
    ax.tick_params(colors="#e2e8f0")
    ax.grid(True, alpha=0.3, color="#334155")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return _to_base64(fig)


def hourly_count_chart(conn) -> str:
    df = pd.read_sql_query(
        "SELECT substr(valid, 1, 13) AS hour FROM metar_observations",
        conn,
    )
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    if df.empty:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color="#e2e8f0")
        return _to_base64(fig)
    counts = df.groupby("hour").size().reset_index(name="rows")
    ax.bar(counts["hour"], counts["rows"], color="#38bdf8")
    ax.set_title("Observations ingested per hour", color="#e2e8f0")
    ax.set_ylabel("Row count", color="#e2e8f0")
    ax.set_xlabel("Hour (UTC)", color="#e2e8f0")
    ax.tick_params(axis="x", rotation=45, colors="#e2e8f0")
    ax.tick_params(axis="y", colors="#e2e8f0")
    ax.grid(True, alpha=0.3, color="#334155", axis="y")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    return _to_base64(fig)


def hrrr_forecast_chart(conn) -> str:
    """Interactive Plotly HRRR 2 m temperature forecast chart for the target station.

    Uses the latest complete cycle in the DB (same cycle that backs the METAR
    comparison) so the chart and the comparison card never drift apart.
    """
    init_dt_str = latest_complete_hrrr_cycle(conn, TARGET_ICAO, required_hours=18, source="hrrr-aws")
    if init_dt_str is None:
        return "<p>No HRRR forecast data yet</p>"

    df = hrrr_forecast_for_cycle(conn, TARGET_ICAO, init_dt_str, source="hrrr-aws")
    if df.empty or len(df) < 18:
        return "<p>Complete HRRR forecast cycle not available</p>"

    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
    df["init_dt"] = pd.to_datetime(df["init_dt"], utc=True)
    df = df.sort_values("forecast_hour")
    init_label = df["init_dt"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
    init_ct_label = df["init_dt"].iloc[0].tz_convert(_CT).strftime("%I:%M %p CT")

    df["valid_ct"] = df["valid_dt"].apply(lambda dt: dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT"))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["valid_dt"],
                y=df["tmpf"],
                mode="lines+markers",
                name="HRRR 2m temp",
                line={"color": "#f59e0b", "width": 2},
                marker={"size": 6, "color": "#f59e0b"},
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.15)",
                hovertemplate=(
                    "<b>%{x|%Y-%m-%d %H:%M UTC}</b><br>"
                    "%{customdata}<br>"
                    "Temp: %{y:.1f}°F<br>"
                    f"Cycle: {init_label}<br>"
                    "f%{text}<extra></extra>"
                ),
                text=df["forecast_hour"].astype(int),
                customdata=df["valid_ct"],
            )
        ]
    )

    # Dynamic Y-axis with a little padding.
    ymin, ymax = df["tmpf"].min(), df["tmpf"].max()
    pad = max(1.0, (ymax - ymin) * 0.15)
    y_min = ymin - pad
    y_max = ymax + pad

    fig.update_layout(
        title=f"HRRR 18-hour forecast — {TARGET_ICAO} 2 m temp<br><sup>Cycle {init_label} · {init_ct_label}</sup>",
        xaxis_title="Valid time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        yaxis={"range": [y_min, y_max], "gridcolor": "#334155"},
        xaxis={"gridcolor": "#334155"},
        showlegend=False,
        hovermode="x unified",
    )

    return pyo.plot(fig, output_type="div", include_plotlyjs="cdn", config={"displayModeBar": False})


def _format_obs_time(obs_time) -> str:
    """Format an observation timestamp for tile display."""
    if obs_time is None:
        return "—"
    dt = pd.to_datetime(obs_time, utc=True)
    return dt.tz_convert(_CT).strftime("%m/%d %I:%M %p CT")


def _compute_cycle_stats(
    blended: pd.DataFrame,
    bias_df: pd.DataFrame,
    latest_obs: pd.Series | None,
    halflife_hours: float,
    trend_weight: float,
) -> dict:
    """Extract per-cycle stat tile values from blended result and bias trace.

    Returns a dict with bias_applied, trend_min/max, n_matched_pairs/hours,
    uncertainty_plus, max_correction, latest_obs_tmpf/time,
    corrected_at_obs_hour, raw_at_obs_hour, hindsight_error/raw_error,
    halflife_hours, trend_weight.
    """
    bias_applied = float(blended["bias_applied"].iloc[0]) if len(blended) > 0 else 0.0
    trend_min = float(blended["trend_correction"].min()) if "trend_correction" in blended.columns else 0.0
    trend_max = float(blended["trend_correction"].max()) if "trend_correction" in blended.columns else 0.0

    n_matched_pairs = int(bias_df["n_matches"].iloc[-1]) if not bias_df.empty else 0
    n_matched_hours = len(bias_df) if not bias_df.empty else 0

    if len(blended) > 0 and "uncertainty_high" in blended.columns:
        unc_width = float(blended["uncertainty_high"].iloc[0] - blended["uncertainty_low"].iloc[0])
        uncertainty_plus = unc_width / 2.0
    else:
        uncertainty_plus = 0.0

    if "tmpf_trend_adjusted" in blended.columns and len(blended) > 0:
        corrections = blended["tmpf_trend_adjusted"] - blended["tmpf"]
        max_correction = float(corrections.min())
    else:
        max_correction = bias_applied

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
        blended_copy = blended.copy()
        blended_copy["valid_hour"] = pd.to_datetime(blended_copy["valid_dt"], utc=True).dt.floor("h")
        match = blended_copy[blended_copy["valid_hour"] == obs_hour]
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


def _build_stat_tiles_html(stats: dict, cycle_idx: int, visible: bool) -> str:
    """Build the stat tile HTML block for one cycle."""
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
  <div class="stat-tile" title="The constant offset added to every HRRR forecast hour. Computed as an exponentially-weighted moving average (EWMA) of (observed minus forecast) errors over recent matched hours. A negative value means HRRR has been running too warm, so we subtract that many degrees from the forecast."><h3>Bias Applied</h3><p style="color:{bias_color}">{bias_val:+.1f}°F</p><small>EWMA constant offset</small></div>
  <div class="stat-tile" title="Extra per-hour adjustment based on how newer HRRR cycles are trending compared to older ones. If the model is cooling its forecasts cycle-over-cycle, we nudge the corrected forecast slightly cooler too. The range shows the min and max trend nudge across all 18 forecast hours."><h3>Trend Correction</h3><p>{stats['trend_min']:+.2f} to {stats['trend_max']:+.2f}°F</p><small>Per-hour range</small></div>
  <div class="stat-tile" title="How many (observation, forecast) pairs were matched and averaged to compute the bias. More pairs across more hours means a more stable bias estimate. The pair count is cumulative across all matched hours."><h3>Matched Data</h3><p>{stats['n_matched_pairs']}</p><small>pairs across {stats['n_matched_hours']} hours</small></div>
  <div class="stat-tile" title="The uncertainty band around the corrected forecast, shown as plus/minus degrees F. Derived from the spread of recent bias errors. The green shaded area on the chart spans corrected ± this value."><h3>Uncertainty ±</h3><p>±{stats['uncertainty_plus']:.1f}°F</p><small>1-sigma band</small></div>
  <div class="stat-tile" title="The largest total adjustment applied to any single forecast hour (bias + trend combined). This is the biggest amount the corrected line differs from the raw HRRR line across all 18 hours."><h3>Max Correction</h3><p style="color:{bias_color}">{stats['max_correction']:+.1f}°F</p><small>Largest total adjustment</small></div>
  <div class="stat-tile" title="The two parameters controlling correction strength. EWMA half-life is how fast old bias errors decay from memory (2h means an error 2 hours old has half the weight of the current one). Trend weight is what fraction of the model's cycle-over-cycle trend is applied (15% is a gentle nudge)."><h3>Config</h3><p>{stats['halflife_hours']:.0f}h / {int(stats['trend_weight']*100)}%</p><small>EWMA half-life / trend weight</small></div>
  <div class="stat-tile" title="The most recent actual temperature observed at the station, from 5-minute NWS API or hourly METAR data. This is the ground truth the corrected forecast is trying to match."><h3>Latest Observed</h3><p>{latest_text}</p><small>{latest_time}</small></div>
  <div class="stat-tile" title="What the bias-corrected forecast predicted for the hour of the latest observation, compared to what the raw HRRR forecast said for that same hour. Shows how much the correction moved the forecast at the time we can verify against."><h3>Corrected at Obs Hour</h3><p>{corrected_text}</p><small>Raw: {raw_text}</small></div>
  <div class="stat-tile" title="How far off the corrected forecast was from the actual observed temperature at the latest observation hour. A small green number means the correction worked well. The raw error in parentheses shows how far off the uncorrected HRRR would have been. Compare the two to see if the correction helped."><h3>Hindsight Error</h3><p style="color:{err_color}">{err_text}</p><small>Corrected vs actual (raw: {raw_err_text})</small></div>
</div>
</div>"""


def _build_bias_table_html(bias_df: pd.DataFrame, cycle_idx: int, visible: bool) -> str:
    """Build a collapsible bias decomposition table for one cycle."""
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
<summary title="Click to expand a table showing how the bias was computed hour-by-hour. Each row is one matched hour where we have both a METAR observation and an HRRR forecast. The EWMA Bias column shows how the rolling bias estimate evolved over time — the final value is what gets applied as the constant correction.">Bias decomposition — per-hour matched errors and EWMA evolution</summary>
<table class="bias-table">
<thead><tr><th title="The hour (in Central Time) that the observation and forecast both apply to.">Valid Hour (CT)</th><th title="Average observed temperature at this hour, from 5-minute NWS API and hourly METAR data combined.">METAR Obs °F</th><th title="Average HRRR forecast temperature for this hour, across all recent cycles that cover it.">HRRR Fcst °F</th><th title="Observed minus forecast. Negative means HRRR was too warm; positive means too cool. This is the raw error before smoothing.">Error °F</th><th title="The exponentially-weighted moving average of errors up to this hour. This is the smoothed bias estimate — the final row's value is what gets applied as the constant correction to all forecast hours.">EWMA Bias °F</th><th title="Cumulative count of (observation, forecast) pairs matched so far, including all cycles that cover this hour.">Pairs</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</details>"""


def blended_forecast_chart(conn) -> str:
    """Interactive chart with per-cycle correction stat tiles and bias decomposition.

    Uses an HTML <select> dropdown to switch between recent HRRR cycles.
    The dropdown syncs stat tiles, bias table, and Plotly trace visibility.
    All cycles' stats are pre-rendered as hidden HTML divs.
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

    plotly_div = pio.to_html(fig, include_plotlyjs=False,
                             config={"displayModeBar": False}, div_id="blended-chart",
                             full_html=False)

    # Build JS with embedded visibility arrays
    import json
    vis_json = json.dumps(visibility_arrays)
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


def _metar_vs_hrrr(conn) -> dict:
    """Return current-hour METAR and the matching HRRR forecast for that hour."""
    metar = pd.read_sql_query(
        """
        SELECT valid, tmpf
        FROM metar_observations
        WHERE station = ?
        ORDER BY valid DESC
        LIMIT 1
        """,
        conn,
        params=[TARGET_ICAO],
    )

    if metar.empty:
        return {
            "metar_tmpf": "—",
            "metar_valid": "No METAR data",
            "metar_valid_ct": "",
            "hrrr_tmpf": "—",
            "hrrr_init": "—",
            "hrrr_fxx": 0,
            "hrrr_valid": "No HRRR data",
            "hrrr_valid_ct": "",
            "delta_text": "—",
        }

    metar_tmpf = round(float(metar.iloc[0]["tmpf"]), 1)
    metar_valid = metar.iloc[0]["valid"]
    metar_dt = pd.to_datetime(metar_valid, utc=True)
    metar_hour = metar_dt.floor("h")

    # Use the same latest complete HRRR cycle as the chart.
    init_dt_str = latest_complete_hrrr_cycle(conn, TARGET_ICAO, required_hours=18, source="hrrr-aws")
    if init_dt_str is None:
        return {
            "metar_tmpf": metar_tmpf,
            "metar_valid": metar_valid,
            "metar_valid_ct": _utc_to_ct(metar_valid, "%m/%d %I:%M %p CT"),
            "hrrr_tmpf": "—",
            "hrrr_init": "—",
            "hrrr_fxx": 0,
            "hrrr_valid": "No HRRR data",
            "hrrr_valid_ct": "",
            "delta_text": "—",
        }

    df = hrrr_forecast_for_cycle(conn, TARGET_ICAO, init_dt_str, source="hrrr-aws")
    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
    df["init_dt"] = pd.to_datetime(df["init_dt"], utc=True)

    # Find the forecast hour whose valid time is closest to the METAR hour.
    df["hour_diff"] = (df["valid_dt"] - metar_hour).abs()
    hrrr = df.sort_values("hour_diff").iloc[0]

    hrrr_tmpf = round(float(hrrr["tmpf"]), 1)
    hrrr_init = hrrr["init_dt"].strftime("%Y-%m-%d %H:%M UTC")
    hrrr_fxx = int(hrrr["forecast_hour"])
    hrrr_valid = hrrr["valid_dt"].strftime("%Y-%m-%d %H:%M UTC")
    delta = round(hrrr_tmpf - metar_tmpf, 1)

    return {
        "metar_tmpf": metar_tmpf,
        "metar_valid": metar_valid,
        "metar_valid_ct": _dt_to_ct(metar_dt, "%m/%d %I:%M %p CT"),
        "hrrr_tmpf": hrrr_tmpf,
        "hrrr_init": hrrr_init,
        "hrrr_fxx": hrrr_fxx,
        "hrrr_valid": hrrr_valid,
        "hrrr_valid_ct": _dt_to_ct(hrrr["valid_dt"], "%m/%d %I:%M %p CT"),
        "delta_text": f"{delta:+0.1f}°F",
    }


def generate_dashboard(db_path: str, output_dir: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_db(db_path)
    stats = summary_stats(conn)
    latest = latest_by_station(conn)
    comparison = _metar_vs_hrrr(conn)
    conn.close()

    # Format latest table: drop internal id/fetched_at, round floats.
    display_latest = latest.copy()
    if "id" in display_latest.columns:
        display_latest = display_latest.drop(columns=["id"])
    if "fetched_at" in display_latest.columns:
        display_latest = display_latest.drop(columns=["fetched_at"])
    for col in ["tmpf", "dewpf", "drct", "sknt", "lat", "lon", "mslp", "p01i"]:
        if col in display_latest.columns:
            display_latest[col] = display_latest[col].astype(float).round(2)

    now_utc = datetime.now(timezone.utc)
    now_ct = now_utc.astimezone(_CT)

    html = HTML_TEMPLATE.format(
        updated_at=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        updated_at_ct=now_ct.strftime("%I:%M %p"),
        total_rows=stats["total_rows"],
        station_count=stats["station_count"],
        first_obs=stats["first_obs"],
        first_obs_ct=stats["first_obs_ct"],
        last_obs=stats["last_obs"],
        last_obs_ct=stats["last_obs_ct"],
        latest_table=display_latest.to_html(index=False, classes="table", border=0),
        kdfw_chart=kdfw_temperature_chart(get_db(db_path)),
        hrrr_chart=hrrr_forecast_chart(get_db(db_path)),
        blended_chart=blended_forecast_chart(get_db(db_path)),
        hourly_chart=hourly_count_chart(get_db(db_path)),
        db_path=db_path,
        TARGET_ICAO=TARGET_ICAO,
        **comparison,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate static weather dashboard")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/data/DAlvarez101.HermesStocks.io/dfw-live-dashboard",
        help="Directory to write index.html",
    )
    args = parser.parse_args()

    path = generate_dashboard(args.db, args.output_dir)
    print(f"Dashboard written to: {path}")


if __name__ == "__main__":
    main()
