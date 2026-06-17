import requests
import pandas as pd

API = "https://archive-api.open-meteo.com/v1/archive"


def build_url(lat: float, lon: float, start: str, end: str) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    r = requests.Request("GET", API, params=params).prepare()
    return r.url


def fetch_hourly_temp(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    url = build_url(lat, lon, start, end)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload["hourly"]
    df = pd.DataFrame(
        {
            "valid": pd.to_datetime(hourly["time"]),
            "fcst_temp_f": hourly["temperature_2m"],
        }
    )
    df["lat"] = lat
    df["lon"] = lon
    return df


def fetch_all_stations(stations, start: str, end: str, cache_path=None) -> pd.DataFrame:
    frames = []
    for st in stations:
        df = fetch_hourly_temp(st.lat, st.lon, start, end)
        df["station"] = st.icao
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if cache_path:
        out.to_parquet(cache_path)
    return out
