"""Run a parameter sweep over the advection model on the validation set.

Usage:
    uv run python scripts/run_parameter_sweep.py

This script:
1. Loads cached ASOS observations and Open-Meteo forecasts for 2020-2024.
2. Builds daily high temperature residual tables.
3. Splits into train (2020-2023) and validation (2024) sets.
4. Runs a grid search over p, boost, half_width, and l_adv_km on the validation set.
5. Saves all results and the best parameter set to data/results/.
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
from dfw_temp_model.evaluation.metrics import rmse
from dfw_temp_model.features.geometry import station_geometry_table
from dfw_temp_model.models.baseline import inverse_distance_predict
from dfw_temp_model.training.splits import time_based_split
from dfw_temp_model.training.tune import DEFAULT_PARAM_GRID, grid_search_advection

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / CACHE_DIR
RESULTS_PATH = ROOT / "data" / "results"

ASOS_CACHE = CACHE_PATH / "asos_2020-01-01_2024-12-31.parquet"
OPENMETEO_CACHE = CACHE_PATH / "openmeteo_2020-01-01_2024-12-31.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advection parameter grid search")
    parser.add_argument(
        "--train-end",
        default="2023-12-31",
        help="End date for training set (validation starts the next day).",
    )
    parser.add_argument(
        "--val-end",
        default="2024-12-31",
        help="End date for validation set.",
    )
    parser.add_argument(
        "--output-stem",
        default="advection_param_sweep",
        help="Stem for CSV and JSON result files.",
    )
    return parser.parse_args()


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ASOS_CACHE.exists():
        raise FileNotFoundError(f"ASOS cache not found: {ASOS_CACHE}")
    if not OPENMETEO_CACHE.exists():
        raise FileNotFoundError(f"Open-Meteo cache not found: {OPENMETEO_CACHE}")

    print(f"Loading ASOS obs from cache: {ASOS_CACHE}")
    obs_df = pd.read_parquet(ASOS_CACHE)
    print(f"Loading Open-Meteo forecasts from cache: {OPENMETEO_CACHE}")
    fcst_df = pd.read_parquet(OPENMETEO_CACHE)
    return obs_df, fcst_df


def get_daily_wind(obs_df: pd.DataFrame, station: str = TARGET_ICAO) -> pd.DataFrame:
    """Derive a daily wind vector from ASOS observations at the hour of max temp."""
    df = obs_df[obs_df["station"] == station].copy()
    if df.empty:
        raise ValueError(f"No observations for wind station {station!r}")

    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["date"] = df["valid"].dt.tz_localize(None).dt.date.astype(str)
    df = df.dropna(subset=["drct", "sknt"])
    if df.empty:
        raise ValueError(f"No valid wind drct/sknt values for station {station!r}")

    daily_rows = (
        df.sort_values(["tmpf", "valid"], ascending=[False, False])
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")
    )
    fallback = (
        df.sort_values("valid").drop_duplicates(subset=["date"], keep="last").set_index("date")
    )

    wind_df = pd.DataFrame(index=daily_rows.index)
    wind_df["wind_dir_deg"] = daily_rows["drct"].fillna(fallback["drct"]).astype(float)
    wind_df["wind_speed_kts"] = daily_rows["sknt"].fillna(fallback["sknt"]).astype(float)
    wind_df.index.name = "date"
    return wind_df


def main() -> None:
    args = parse_args()

    obs_df, fcst_df = load_cached_data()

    # Build daily highs, residuals, and target table.
    obs_daily = compute_daily_highs(obs_df)
    fcst_daily = compute_forecast_daily_highs(fcst_df)
    residuals = build_residual_table(obs_daily, fcst_daily, STATIONS)
    target_table = build_target_table(obs_daily, fcst_daily, residuals).sort_index()

    # Split by date.
    train, val, test = time_based_split(target_table, args.train_end, args.val_end)
    print(f"\nDate ranges: train {len(train)} days, val {len(val)} days, test {len(test)} days")

    # Residual table used by the model: target residual plus neighbor residuals.
    residuals_full = target_table[["residual_target"] + residuals.columns.tolist()].copy()
    residuals_full = residuals_full.rename(columns={"residual_target": TARGET_ICAO})

    # Geometry table.
    geom = station_geometry_table(STATIONS).reset_index()

    # Grid search uses only validation dates.
    residuals_val = residuals_full.loc[val.index]
    wind_df_val = get_daily_wind(obs_df).loc[val.index]

    print(
        f"\nRunning grid search over {DEFAULT_PARAM_GRID} ({300} combinations) ..."
    )
    best_params, best_score, all_results = grid_search_advection(
        residuals_val,
        geom,
        wind_df_val,
        val.index,
        param_grid=DEFAULT_PARAM_GRID,
        target_col=TARGET_ICAO,
    )

    print(f"\nBest validation score (negative RMSE): {best_score:.4f}")
    print(f"Best validation residual RMSE: {-best_score:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Save results.
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_PATH / f"{args.output_stem}.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"\nSaved sweep CSV: {csv_path}")

    json_path = RESULTS_PATH / "best_params.json"
    payload = {
        "best_params": best_params,
        "best_score": best_score,
        "best_rmse": -best_score,
        "param_grid": DEFAULT_PARAM_GRID,
        "split": {
            "train_start": str(train.index[0]) if len(train) else None,
            "train_end": str(train.index[-1]) if len(train) else None,
            "train_days": len(train),
            "val_start": str(val.index[0]) if len(val) else None,
            "val_end": str(val.index[-1]) if len(val) else None,
            "val_days": len(val),
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved best params JSON: {json_path}")


if __name__ == "__main__":
    main()
