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

from dfw_temp_model.config import CACHE_DIR, TARGET_ICAO
from dfw_temp_model.storage.obs_db import get_db, latest_by_station

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
        img {{ max-width: 100%; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #334155; }}
        .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.85rem; max-width: 900px; }}
        a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>DFW Live Weather Dashboard</h1>
    <p>Updated at {updated_at} UTC · Source: AviationWeather.gov METAR JSON</p>

    <div class="stats">
        <div class="card"><h3>Total observations</h3><p>{total_rows}</p></div>
        <div class="card"><h3>Stations</h3><p>{station_count}</p></div>
        <div class="card"><h3>First observation</h3><p>{first_obs}</p></div>
        <div class="card"><h3>Latest observation</h3><p>{last_obs}</p></div>
    </div>

    <h2>Latest readings per station</h2>
    {latest_table}

    <h2>Temperature trend ({TARGET_ICAO})</h2>
    <img src="data:image/png;base64,{kdfw_chart}" alt="KDFW temperature trend">

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
        params=(TARGET_ICAO,),
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


def generate_dashboard(db_path: str, output_dir: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_db(db_path)
    stats = summary_stats(conn)
    latest = latest_by_station(conn)
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
        hourly_chart=hourly_count_chart(get_db(db_path)),
        db_path=db_path,
        TARGET_ICAO=TARGET_ICAO,
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
