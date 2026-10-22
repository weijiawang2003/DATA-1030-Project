"""Collect Brown University daily reanalysis only when explicitly requested."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLES = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "pressure_msl_mean", "cloud_cover_mean",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "daylight_duration", "weather_code",
]


def weather_types(frame, precipitation_threshold=1.0, cloud_threshold=40.0):
    """Apply precipitation first; missing measurements never become Clear."""
    precip, cloud = frame["precipitation_sum"], frame["cloud_cover_mean"]
    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    labels.loc[precip.ge(precipitation_threshold)] = "Precipitation"
    dry = precip.notna() & precip.lt(precipitation_threshold) & cloud.notna()
    labels.loc[dry & cloud.ge(cloud_threshold)] = "Cloudy"
    labels.loc[dry & cloud.lt(cloud_threshold)] = "Clear"
    return labels


def collect_data(config, raw_dir):
    """Return daily data, quality metadata, and API units; cache the response.

    An identical cached query avoids a new request. API days follow the requested
    timezone convention; retain utc_offset_seconds when reproducing the data.
    """
    location = config["location"]
    start, end = config["start_date"], config["end_date"]
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start_date must not follow end_date")
    params = {
        "latitude": location["latitude"], "longitude": location["longitude"],
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARIABLES), "models": config["model"],
        "timezone": config["timezone"], "temperature_unit": "celsius",
        "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    raw_dir = Path(raw_dir)
    cache = raw_dir / f"{location['id']}_{start}_{end}.json"
    cached = cache.exists()
    if cached:
        payload = json.loads(cache.read_text())
        if payload.get("query") != params:
            raise ValueError(f"Cached query differs from configuration: {cache}")
    else:
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504],
                      allowed_methods=["GET"], respect_retry_after_header=True)
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            response = session.get(API_URL, params=params, timeout=(15, 180))
            if response.status_code == 429:
                raise RuntimeError("Open-Meteo rate limit reached; try collection later.")
            response.raise_for_status()
            payload = {"api_url": API_URL, "query": params,
                       "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                       "response": response.json()}
    js = payload["response"]
    if js.get("error") or "daily" not in js:
        raise ValueError(f"API response error: {js.get('reason', 'missing daily data')}")
    if set(js["daily"]) != {"time", *DAILY_VARIABLES}:
        raise ValueError("API daily columns differ from the requested 15 variables")
    daily = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    expected = pd.Series(pd.date_range(start, end))
    dates = daily["date"].reset_index(drop=True)
    if dates.duplicated().any() or len(dates) != len(expected) or not dates.eq(expected).all():
        raise ValueError("API dates must be complete, unique, ordered daily observations")
    for column in DAILY_VARIABLES:
        daily[column] = pd.to_numeric(daily[column], errors="raise")
    if (daily["temperature_2m_max"] < daily["temperature_2m_min"]).any():
        raise ValueError("Daily maximum temperature is below the minimum")
    if (daily[["precipitation_sum", "rain_sum", "snowfall_sum"]] < 0).any().any():
        raise ValueError("Negative precipitation amount")
    daily.insert(0, "city", location["name"])
    daily["temp_range"] = daily["temperature_2m_max"] - daily["temperature_2m_min"]
    daily["weather_type"] = weather_types(
        daily, config["precipitation_threshold_mm"], config["cloud_threshold_percent"],
    )
    quality = {
        "city": location["name"], "rows": len(daily), "api_variable_count": len(DAILY_VARIABLES),
        "start_date": start, "end_date": end, "duplicate_dates": 0,
        "missing_weather_values": int(daily[DAILY_VARIABLES].isna().sum().sum()),
        "requested_latitude": location["latitude"], "requested_longitude": location["longitude"],
        "grid_latitude": js["latitude"], "grid_longitude": js["longitude"],
        "grid_elevation_m": js.get("elevation"), "timezone": js.get("timezone"),
        "utc_offset_seconds": js.get("utc_offset_seconds"),
        "cache_file": cache.name, "fetched_at_utc": payload["fetched_at_utc"],
    }
    if not cached:
        raw_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        temporary.replace(cache)
    return daily, quality, js.get("daily_units", {})
