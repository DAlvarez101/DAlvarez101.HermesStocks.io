#!/usr/bin/env python3
"""Generate a static HTML database viewer page from the SQLite weather DB."""
import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

# Ensure project is importable when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dfw_temp_model.config import CACHE_DIR, TARGET_ICAO

_CT = ZoneInfo("America/Chicago")


def _dt_to_ct(dt, fmt="%m/%d %I:%M %p CT"):
    """Convert a timezone-aware datetime to Central Time string."""
    ct = dt.tz_convert(_CT)
    return ct.strftime(fmt)


def _table_html(df: pd.DataFrame, max_rows: int = 500) -> str:
    """Convert a DataFrame to an HTML table with inline CSS and JS sorting."""
    display = df.head(max_rows).copy()
    truncated = len(df) > max_rows

    # Round floats
    for col in display.select_dtypes(include=["float"]).columns:
        display[col] = display[col].round(2)

    html = display.to_html(index=False, classes="data-table", border=0, escape=False)
    if truncated:
        note = f'<p class="note">Showing latest {max_rows} of {len(df)} rows</p>'
    else:
        note = ""
    return note + html


def _metar_trend_chart(df: pd.DataFrame) -> str:
    """Plotly chart: temperature trend per station from METAR observations."""
    if df.empty:
        return "<p>No METAR data</p>"
    df = df.copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    stations = df["station"].unique()

    fig = go.Figure()
    for st in sorted(stations):
        sub = df[df["station"] == st].sort_values("valid")
        fig.add_trace(go.Scatter(
            x=sub["valid"],
            y=sub["tmpf"],
            mode="lines+markers",
            name=st,
            hovertemplate=f"<b>{st}</b><br>%{{x|%Y-%m-%d %H:%M UTC}}<br>Temp: %{{y:.1f}}°F<extra></extra>",
        ))

    fig.update_layout(
        title="METAR Temperature Trend — All Stations",
        xaxis_title="Time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        hovermode="x unified",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs="cdn", config={"displayModeBar": False})


def _hrrr_forecast_chart(df: pd.DataFrame) -> str:
    """Plotly chart: latest HRRR cycle forecast for target station."""
    target_df = df[df["station"] == TARGET_ICAO].copy()
    if target_df.empty:
        return f"<p>No HRRR data for {TARGET_ICAO}</p>"

    target_df["valid_dt"] = pd.to_datetime(target_df["valid_dt"], utc=True)
    target_df["init_dt"] = pd.to_datetime(target_df["init_dt"], utc=True)

    # Pick the latest init cycle with the most forecast hours
    latest_init = target_df["init_dt"].max()
    cycle_df = target_df[target_df["init_dt"] == latest_init].sort_values("forecast_hour")

    if cycle_df.empty:
        return "<p>No complete HRRR cycle available</p>"

    fig = go.Figure(data=[go.Scatter(
        x=cycle_df["valid_dt"],
        y=cycle_df["tmpf"],
        mode="lines+markers",
        name=f"HRRR {TARGET_ICAO}",
        line={"color": "#f59e0b", "width": 2},
        marker={"size": 6, "color": "#f59e0b"},
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.15)",
        hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M UTC}}</b><br>Temp: %{{y:.1f}}°F<br>f%{{text}}<extra></extra>",
        text=cycle_df["forecast_hour"].astype(int),
    )])

    init_label = latest_init.strftime("%Y-%m-%d %H:%M UTC")
    ct_label = _dt_to_ct(latest_init)

    fig.update_layout(
        title=f"HRRR Forecast — {TARGET_ICAO} 2m temp<br><sup>Latest cycle {init_label} · {ct_label}</sup>",
        xaxis_title="Valid time (UTC)",
        yaxis_title="Temperature (°F)",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font={"color": "#e2e8f0"},
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        showlegend=False,
        hovermode="x unified",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False, config={"displayModeBar": False})


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DFW Weather Database Viewer</title>
    <style>
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #38bdf8; margin-bottom: 0.25rem; }}
        h2 {{ color: #7dd3fc; margin-top: 2rem; }}
        p.updated {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin: 1rem 0; max-width: 900px; }}
        .card {{ background: #1e293b; padding: 1rem; border-radius: 0.5rem; }}
        .card h3 {{ margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.03em; }}
        .card p {{ margin: 0; font-size: 1.4rem; font-weight: 600; color: #38bdf8; }}
        .note {{ color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0; }}
        table.data-table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; max-width: 1200px; font-size: 0.82rem; }}
        table.data-table th, table.data-table td {{ border: 1px solid #334155; padding: 0.35rem 0.6rem; text-align: left; cursor: pointer; }}
        table.data-table th {{ background: #1e293b; position: sticky; top: 0; }}
        table.data-table th:hover {{ background: #334155; }}
        table.data-table tr:nth-child(even) {{ background: #162032; }}
        table.data-table tr:hover {{ background: #1e3a5f; }}
        .table-wrap {{ max-height: 500px; overflow-y: auto; margin: 1rem 0; max-width: 1200px; }}
        .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.85rem; max-width: 900px; }}
        a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>DFW Weather Database Viewer</h1>
    <p class="updated">Generated {generated_at} UTC · {generated_at_ct} CT · Source: {db_path}<br>This is a read-only snapshot refreshed hourly by the cron job.</p>

    <div class="stats">
        <div class="card"><h3>METAR rows</h3><p>{metar_count}</p></div>
        <div class="card"><h3>HRRR rows</h3><p>{hrrr_count}</p></div>
        <div class="card"><h3>Stations</h3><p>{station_count}</p></div>
        <div class="card"><h3>Date range</h3><p>{date_range}</p></div>
    </div>

    <h2>METAR Temperature Trends</h2>
    {metar_chart}

    <h2>HRRR Forecast — Latest Cycle ({TARGET_ICAO})</h2>
    {hrrr_chart}

    <h2>METAR Observations</h2>
    <div class="table-wrap">
    {metar_table}
    </div>

    <h2>HRRR Forecasts</h2>
    <div class="table-wrap">
    {hrrr_table}
    </div>

    <div class="footer">
        <a href="../dfw-live-dashboard/">← Back to Weather Dashboard</a><br>
        <a href="https://github.com/DAlvarez101/DAlvarez101.HermesStocks.io">View repo →</a>
    </div>

    <script>
    // Lightweight table sorter — click a header to sort by that column.
    document.querySelectorAll('table.data-table th').forEach((th, colIdx) => {{
        th.addEventListener('click', () => {{
            const table = th.closest('table');
            const tbody = table.querySelector('tbody') || table;
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const numeric = !isNaN(parseFloat(rows[0]?.querySelector(`td:nth-child(${{colIdx + 1}})`)?.textContent));
            const asc = th.dataset.sort !== 'asc';
            th.dataset.sort = asc ? 'asc' : 'desc';
            rows.sort((a, b) => {{
                let av = a.querySelector(`td:nth-child(${{colIdx + 1}})`)?.textContent ?? '';
                let bv = b.querySelector(`td:nth-child(${{colIdx + 1}})`)?.textContent ?? '';
                if (numeric) {{
                    av = parseFloat(av) || 0;
                    bv = parseFloat(bv) || 0;
                }}
                return asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
            }});
            rows.forEach(r => tbody.appendChild(r));
        }});
    }});
    </script>
</body>
</html>
"""


def generate_db_viewer(db_path: str, output_dir: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    metar_df = pd.read_sql_query("SELECT * FROM metar_observations ORDER BY valid DESC", conn)
    hrrr_df = pd.read_sql_query("SELECT * FROM hrrr_forecasts ORDER BY valid_dt DESC", conn)
    conn.close()

    # Stats
    metar_count = len(metar_df)
    hrrr_count = len(hrrr_df)
    station_count = metar_df["station"].nunique() if not metar_df.empty else 0

    if not metar_df.empty:
        first_valid = pd.to_datetime(metar_df["valid"].min(), utc=True)
        last_valid = pd.to_datetime(metar_df["valid"].max(), utc=True)
        date_range = f"{first_valid.strftime('%m/%d')} – {last_valid.strftime('%m/%d')}"
    else:
        date_range = "—"

    # Charts
    metar_chart = _metar_trend_chart(metar_df) if not metar_df.empty else "<p>No METAR data</p>"
    hrrr_chart = _hrrr_forecast_chart(hrrr_df) if not hrrr_df.empty else "<p>No HRRR data</p>"

    # Tables (prepare display columns)
    metar_cols = ["station", "valid", "tmpf", "dewpf", "drct", "sknt", "mslp", "p01i"]
    metar_display = metar_df[[c for c in metar_cols if c in metar_df.columns]].copy()
    metar_display.columns = ["Station", "Valid (UTC)", "Temp °F", "Dewpt °F", "Dir °", "Speed kt", "MSLP mb", "Precip in"]

    hrrr_cols = ["station", "init_dt", "forecast_hour", "valid_dt", "tmpf"]
    hrrr_display = hrrr_df[[c for c in hrrr_cols if c in hrrr_df.columns]].copy()
    hrrr_display.columns = ["Station", "Init (UTC)", "Fhr", "Valid (UTC)", "Temp °F"]

    now_utc = datetime.now(timezone.utc)
    now_ct = now_utc.astimezone(_CT)

    html = HTML_TEMPLATE.format(
        generated_at=now_utc.strftime("%Y-%m-%d %H:%M:%S"),
        generated_at_ct=now_ct.strftime("%I:%M %p"),
        db_path=db_path,
        metar_count=metar_count,
        hrrr_count=hrrr_count,
        station_count=station_count,
        date_range=date_range,
        metar_chart=metar_chart,
        hrrr_chart=hrrr_chart,
        metar_table=_table_html(metar_display),
        hrrr_table=_table_html(hrrr_display),
        TARGET_ICAO=TARGET_ICAO,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate static DB viewer HTML")
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(CACHE_DIR) / "db" / "weather_observations.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/opt/data/DAlvarez101.HermesStocks.io/dfw-data-viewer",
        help="Directory to write index.html",
    )
    args = parser.parse_args()

    path = generate_db_viewer(args.db, args.output_dir)
    print(f"DB viewer written to: {path}")


if __name__ == "__main__":
    main()