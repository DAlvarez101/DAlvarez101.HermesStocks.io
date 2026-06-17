"""Generate a static HTML dashboard from the observation database."""
import argparse
import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from dfw_temp_model.config import CACHE_DIR, TARGET_ICAO
from dfw_temp_model.storage.obs_db import (
    get_db,
    hrrr_forecast_range,
    hrrr_for_valid_hour,
    latest_by_station,
)

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
    </style>
</head>
<body>
    <h1>DFW Live Weather Dashboard</h1>
    <p>Updated at {updated_at} UTC · Sources: AviationWeather.gov METAR JSON, NOAA HRRR AWS Open Data</p>

    <div class="stats">
        <div class="card"><h3>Total METAR obs</h3><p>{total_rows}</p></div>
        <div class="card"><h3>Stations</h3><p>{station_count}</p></div>
        <div class="card"><h3>First observation</h3><p>{first_obs}</p></div>
        <div class="card"><h3>Latest observation</h3><p>{last_obs}</p></div>
    </div>

    <h2>METAR vs HRRR at {TARGET_ICAO} — current hour</h2>
    <div class="comparison">
        <div class="card">
            <h3>METAR observed</h3>
            <p>{metar_tmpf}°F</p>
            <small>{metar_valid}</small>
        </div>
        <div class="card">
            <h3>HRRR forecast</h3>
            <p>{hrrr_tmpf}°F</p>
            <small>Cycle {hrrr_init} · f{hrrr_fxx:02d} · valid {hrrr_valid}</small>
        </div>
        <div class="card">
            <h3>Difference</h3>
            <p>{delta_text}</p>
            <small>HRRR minus METAR</small>
        </div>
    </div>

    <h2>HRRR 18-hour forecast ({TARGET_ICAO})</h2>
    {hrrr_chart}

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
        }
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return {
        "total_rows": len(df),
        "station_count": df["station"].nunique(),
        "first_obs": df["valid"].min().strftime("%Y-%m-%d %H:%M UTC"),
        "last_obs": df["valid"].max().strftime("%Y-%m-%d %H:%M UTC"),
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
    """Interactive Plotly HRRR 2 m temperature forecast chart for the target station."""
    df = hrrr_forecast_range(conn, station=TARGET_ICAO, limit=18)
    if df.empty:
        return "<p>No HRRR forecast data yet</p>"

    df["valid_dt"] = pd.to_datetime(df["valid_dt"], utc=True)
    df["init_dt"] = pd.to_datetime(df["init_dt"], utc=True)
    df = df.sort_values("valid_dt")
    init_label = df["init_dt"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")

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
                    "Temp: %{y:.1f}°F<br>"
                    f"Cycle: {init_label}<br>"
                    "f%{text}<extra></extra>"
                ),
                text=df["forecast_hour"].astype(int),
            )
        ]
    )

    # Dynamic Y-axis with a little padding.
    ymin, ymax = df["tmpf"].min(), df["tmpf"].max()
    pad = max(1.0, (ymax - ymin) * 0.15)
    y_min = ymin - pad
    y_max = ymax + pad

    fig.update_layout(
        title=f"HRRR 18-hour forecast — {TARGET_ICAO} 2 m temp",
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
            "hrrr_tmpf": "—",
            "hrrr_init": "—",
            "hrrr_fxx": 0,
            "hrrr_valid": "No HRRR data",
            "delta_text": "—",
        }

    metar_tmpf = round(float(metar.iloc[0]["tmpf"]), 1)
    metar_valid = metar.iloc[0]["valid"]
    metar_dt = pd.to_datetime(metar_valid, utc=True)

    # Find the HRRR forecast valid in the same hour as the latest METAR.
    # Prefer the forecast from the most recent complete cycle for that valid hour.
    hrrr = hrrr_for_valid_hour(conn, station=TARGET_ICAO, valid_hour=metar_dt.floor("h"))

    if hrrr.empty:
        return {
            "metar_tmpf": metar_tmpf,
            "metar_valid": metar_valid,
            "hrrr_tmpf": "—",
            "hrrr_init": "—",
            "hrrr_fxx": 0,
            "hrrr_valid": "No HRRR data",
            "delta_text": "—",
        }

    # Pick the row with the latest init_dt as the best estimate for that hour.
    hrrr["init_dt"] = pd.to_datetime(hrrr["init_dt"], utc=True)
    hrrr["valid_dt"] = pd.to_datetime(hrrr["valid_dt"], utc=True)
    hrrr = hrrr.sort_values("init_dt", ascending=False).iloc[0]

    hrrr_tmpf = round(float(hrrr["tmpf"]), 1)
    hrrr_init = hrrr["init_dt"].strftime("%Y-%m-%d %H:%M UTC")
    hrrr_fxx = int(hrrr["forecast_hour"])
    hrrr_valid = hrrr["valid_dt"].strftime("%Y-%m-%d %H:%M UTC")
    delta = round(hrrr_tmpf - metar_tmpf, 1)

    return {
        "metar_tmpf": metar_tmpf,
        "metar_valid": metar_valid,
        "hrrr_tmpf": hrrr_tmpf,
        "hrrr_init": hrrr_init,
        "hrrr_fxx": hrrr_fxx,
        "hrrr_valid": hrrr_valid,
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

    html = HTML_TEMPLATE.format(
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        total_rows=stats["total_rows"],
        station_count=stats["station_count"],
        first_obs=stats["first_obs"],
        last_obs=stats["last_obs"],
        latest_table=display_latest.to_html(index=False, classes="table", border=0),
        kdfw_chart=kdfw_temperature_chart(get_db(db_path)),
        hrrr_chart=hrrr_forecast_chart(get_db(db_path)),
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
