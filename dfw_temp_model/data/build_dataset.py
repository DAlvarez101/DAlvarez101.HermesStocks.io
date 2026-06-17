import pandas as pd


def compute_daily_highs(obs_df: pd.DataFrame) -> pd.DataFrame:
    df = obs_df[["station", "valid", "tmpf"]].copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["date"] = df["valid"].dt.tz_localize(None).dt.date.astype(str)
    daily = df.groupby(["date", "station"])["tmpf"].max().reset_index()
    return daily.pivot(index="date", columns="station", values="tmpf")


def compute_forecast_daily_highs(fcst_df: pd.DataFrame) -> pd.DataFrame:
    df = fcst_df[["station", "valid", "fcst_temp_f"]].copy()
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df["date"] = df["valid"].dt.tz_localize(None).dt.date.astype(str)
    daily = df.groupby(["date", "station"])["fcst_temp_f"].max().reset_index()
    return daily.pivot(index="date", columns="station", values="fcst_temp_f")


def build_residual_table(obs_daily: pd.DataFrame, fcst_daily: pd.DataFrame, stations) -> pd.DataFrame:
    merged = obs_daily.join(fcst_daily, how="inner", rsuffix="_fcst")
    residuals = pd.DataFrame(index=merged.index)
    for st in stations:
        if st.icao == "KDFW":
            continue
        if st.icao in obs_daily.columns and f"{st.icao}_fcst" in merged.columns:
            residuals[st.icao] = merged[st.icao] - merged[f"{st.icao}_fcst"]
    return residuals


def build_target_table(obs_daily, fcst_daily, residuals) -> pd.DataFrame:
    target = pd.DataFrame(index=residuals.index)
    target["kdfw_obs"] = obs_daily["KDFW"]
    target["kdfw_fcst"] = fcst_daily["KDFW"]
    target["residual_target"] = target["kdfw_obs"] - target["kdfw_fcst"]
    for col in residuals.columns:
        target[col] = residuals[col]
    return target
