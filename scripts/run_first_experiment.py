"""First end-to-end experiment: baseline vs. advection on a 2024 holdout.

Usage:
    /opt/hermes/.venv/bin/python scripts/run_first_experiment.py

By default the date range is 2020-2024, but the script accepts CLI arguments to
override this. The user asked to verify on 2024-only data, so the simplest way is:

    /opt/hermes/.venv/bin/python scripts/run_first_experiment.py \
        --start-date 2024-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dfw_temp_model.config import CACHE_DIR, STATIONS, TARGET_ICAO
from dfw_temp_model.data.build_dataset import (
    build_residual_table,
    build_target_table,
    compute_daily_highs,
    compute_forecast_daily_highs,
)
from dfw_temp_model.data.iem_asos import fetch_all_stations as fetch_asos
from dfw_temp_model.data.openmeteo import fetch_all_stations as fetch_openmeteo
from dfw_temp_model.evaluation.metrics import evaluate_correction, mae, rmse
from dfw_temp_model.features.geometry import station_geometry_table
from dfw_temp_model.models.advection import advection_predict, advection_predict_with_fronts
from dfw_temp_model.models.baseline import inverse_distance_predict
from dfw_temp_model.training.splits import time_based_split

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / CACHE_DIR
RESULTS_PATH = ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs. advection experiment")
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="Start date for data fetch (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date for data fetch (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch and overwrite cached data even if cache exists.",
    )
    parser.add_argument(
        "--p-baseline",
        type=float,
        default=2.0,
        help="Inverse-distance exponent for the baseline model.",
    )
    parser.add_argument(
        "--p-advection",
        type=float,
        default=2.0,
        help="Inverse-distance exponent for the advection model.",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=3.0,
        help="Upwind boost for the advection model.",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=45.0,
        help="Upwind cone half-width in degrees.",
    )
    parser.add_argument(
        "--l-adv-km",
        type=float,
        default=50.0,
        help="Advection decay length in kilometers.",
    )
    parser.add_argument(
        "--wind-station",
        default=TARGET_ICAO,
        help="Station used to derive daily wind for advection weighting.",
    )
    parser.add_argument(
        "--output-stem",
        default="2024_holdout",
        help="Stem used for CSV and JSON result files.",
    )
    parser.add_argument(
        "--use-fronts",
        action="store_true",
        help="Enable front-detection fallback branch in the advection model.",
    )
    parser.add_argument(
        "--gradient-threshold",
        type=float,
        default=0.05,
        help="Residual gradient threshold (°F/km) for front detection.",
    )
    parser.add_argument(
        "--front-fallback",
        default="mean",
        choices=["mean", "idw"],
        help="Fallback predictor used on flagged front days.",
    )
    parser.add_argument(
        "--front-uncertainty-mult",
        type=float,
        default=2.0,
        help="Uncertainty multiplier recorded for front-day predictions.",
    )
    return parser.parse_args()


def load_or_fetch_asos(start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load ASOS obs from cache if present, otherwise fetch and cache."""
    cache_file = CACHE_PATH / f"asos_{start}_{end}.parquet"
    if not force_refresh and cache_file.exists():
        print(f"Loading ASOS obs from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Fetching ASOS obs for {start} to {end} ...")
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    df = fetch_asos(start, end, STATIONS, cache_path=str(cache_file))
    print(f"ASOS obs cached: {cache_file} ({len(df)} rows)")
    return df


def load_or_fetch_openmeteo(start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load Open-Meteo forecasts from cache if present, otherwise fetch and cache."""
    cache_file = CACHE_PATH / f"openmeteo_{start}_{end}.parquet"
    if not force_refresh and cache_file.exists():
        print(f"Loading Open-Meteo forecasts from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Fetching Open-Meteo forecasts for {start} to {end} ...")
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    df = fetch_openmeteo(STATIONS, start, end, cache_path=str(cache_file))
    print(f"Open-Meteo forecasts cached: {cache_file} ({len(df)} rows)")
    return df


def get_daily_wind(obs_df: pd.DataFrame, station: str = TARGET_ICAO) -> pd.DataFrame:
    """Derive a daily wind vector from ASOS observations.

    For each calendar date, select the observation row at the hour of maximum
    temperature for the requested station and report its wind direction and
    speed. This matches the time of day when the temperature residual is most
    relevant to the advection model. If the hour of max temperature cannot be
    found, fall back to the last available observation of the day.
    """
    df = obs_df[obs_df["station"] == station].copy()
    if df.empty:
        raise ValueError(f"No observations for wind station {station!r}")

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["date"] = df["valid"].dt.tz_localize(None).dt.date.astype(str)
    df = df.dropna(subset=["drct", "sknt"])
    if df.empty:
        raise ValueError(f"No valid wind drct/sknt values for station {station!r}")

    # Pick the row with the highest tmpf per day; tie-break by latest valid time.
    daily_rows = (
        df.sort_values(["tmpf", "valid"], ascending=[False, False])
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")
    )

    # Fallback: if tmpf is missing for a day, use the last valid observation.
    fallback = df.sort_values("valid").drop_duplicates(subset=["date"], keep="last").set_index("date")

    wind_df = pd.DataFrame(index=daily_rows.index)
    wind_df["wind_dir_deg"] = daily_rows["drct"].fillna(fallback["drct"]).astype(float)
    wind_df["wind_speed_kts"] = daily_rows["sknt"].fillna(fallback["sknt"]).astype(float)
    wind_df.index.name = "date"
    return wind_df


def main() -> None:
    args = parse_args()

    start_date = args.start_date
    end_date = args.end_date

    # Fetch / load data.
    obs_df = load_or_fetch_asos(start_date, end_date, force_refresh=args.force_refresh)
    fcst_df = load_or_fetch_openmeteo(start_date, end_date, force_refresh=args.force_refresh)

    # Build daily highs and residual/target tables.
    obs_daily = compute_daily_highs(obs_df)
    fcst_daily = compute_forecast_daily_highs(fcst_df)
    residuals = build_residual_table(obs_daily, fcst_daily, STATIONS)
    target_table = build_target_table(obs_daily, fcst_daily, residuals)

    # The index is a date string; ensure it is sorted.
    target_table = target_table.sort_index()

    # Time-based split.
    train, val, test = time_based_split(target_table, "2023-12-31", "2024-12-31")

    print(f"\nDate ranges: train {len(train)} days, val {len(val)} days, test {len(test)} days")

    # Geometry table.
    geom = station_geometry_table(STATIONS).reset_index()

    # Baseline inverse-distance model on val set.
    # The residual target is KDFW's own observation-minus-forecast residual.
    residuals_full = target_table[["residual_target"] + residuals.columns.tolist()].copy()
    residuals_full = residuals_full.rename(columns={"residual_target": TARGET_ICAO})

    baseline_pred = inverse_distance_predict(
        residuals_full.loc[val.index],
        geom,
        target_col=TARGET_ICAO,
        p=args.p_baseline,
    )

    # Wind for advection model.
    wind_df = get_daily_wind(obs_df, station=args.wind_station)
    wind_df = wind_df.loc[wind_df.index.intersection(val.index)]

    if args.use_fronts:
        advection_pred = advection_predict_with_fronts(
            residuals_full.loc[val.index],
            geom,
            wind_df,
            target_col=TARGET_ICAO,
            p=args.p_advection,
            boost=args.boost,
            half_width=args.half_width,
            l_adv_km=args.l_adv_km,
            front_params={
                "gradient_threshold": args.gradient_threshold,
                "front_fallback": args.front_fallback,
                "uncertainty_multiplier": args.front_uncertainty_mult,
            },
        )
        front_day_count = int(advection_pred["front_day"].sum())
        print(f"\nFront days detected: {front_day_count} / {len(val)}")
    else:
        advection_pred = advection_predict(
            residuals_full.loc[val.index],
            geom,
            wind_df,
            target_col=TARGET_ICAO,
            p=args.p_advection,
            boost=args.boost,
            half_width=args.half_width,
            l_adv_km=args.l_adv_km,
        )
        front_day_count = None

    # Pull observed high, raw forecast, and corrected forecasts for val dates.
    val_obs = target_table.loc[val.index, "kdfw_obs"]
    val_fcst = target_table.loc[val.index, "kdfw_fcst"]
    baseline_corrected = val_fcst + baseline_pred["predicted_residual"]
    advection_corrected = val_fcst + advection_pred["predicted_residual"]

    # Compute metrics.
    baseline_metrics = evaluate_correction(val_obs, val_fcst, baseline_corrected)
    advection_metrics = evaluate_correction(val_obs, val_fcst, advection_corrected)

    # Per-sample residual-prediction metrics (residual target space).
    residual_target = target_table.loc[val.index, "residual_target"]
    baseline_residual_rmse = rmse(residual_target, baseline_pred["predicted_residual"])
    advection_residual_rmse = rmse(residual_target, advection_pred["predicted_residual"])

    # Print results.
    print("\n=== Baseline (inverse-distance) ===")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  residual_rmse: {baseline_residual_rmse:.4f}")

    print("\n=== Advection ===")
    for k, v in advection_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  residual_rmse: {advection_residual_rmse:.4f}")

    # Save per-date comparison CSV.
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        {
            "date": val.index,
            "kdfw_obs": val_obs.values,
            "kdfw_fcst": val_fcst.values,
            "baseline_pred_residual": baseline_pred["predicted_residual"].values,
            "advection_pred_residual": advection_pred["predicted_residual"].values,
            "baseline_corrected": baseline_corrected.values,
            "advection_corrected": advection_corrected.values,
            "baseline_error": (val_obs - baseline_corrected).values,
            "advection_error": (val_obs - advection_corrected).values,
        }
    )
    if args.use_fronts:
        comparison["front_day"] = advection_pred["front_day"].values
        comparison["front_uncertainty_multiplier"] = advection_pred[
            "front_uncertainty_multiplier"
        ].values
    csv_path = RESULTS_PATH / f"{args.output_stem}_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV: {csv_path}")

    # Save metrics JSON.
    metrics_payload = {
        "date_range": {"start": start_date, "end": end_date},
        "split": {
            "train_days": len(train),
            "val_days": len(val),
            "test_days": len(test),
        },
        "baseline": {
            **baseline_metrics,
            "residual_rmse": baseline_residual_rmse,
        },
        "advection": {
            **advection_metrics,
            "residual_rmse": advection_residual_rmse,
        },
        "parameters": {
            "p_baseline": args.p_baseline,
            "p_advection": args.p_advection,
            "boost": args.boost,
            "half_width": args.half_width,
            "l_adv_km": args.l_adv_km,
            "wind_station": args.wind_station,
            "use_fronts": args.use_fronts,
            "gradient_threshold": args.gradient_threshold if args.use_fronts else None,
            "front_fallback": args.front_fallback if args.use_fronts else None,
            "front_uncertainty_mult": args.front_uncertainty_mult if args.use_fronts else None,
            "front_day_count": front_day_count,
        },
    }
    json_path = RESULTS_PATH / f"{args.output_stem}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
