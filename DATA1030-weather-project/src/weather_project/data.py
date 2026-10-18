"""Collect versioned API responses and construct predictors without future values."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
DESCRIPTIONS = {
    "temperature_2m_mean": "Daily mean air temperature at 2 m",
    "temperature_2m_max": "Daily maximum air temperature at 2 m",
    "temperature_2m_min": "Daily minimum air temperature at 2 m",
    "relative_humidity_2m_mean": "Daily mean relative humidity at 2 m",
    "pressure_msl_mean": "Daily mean sea-level pressure",
    "cloud_cover_mean": "Daily mean total cloud cover",
    "precipitation_sum": "Total daily precipitation including snow water equivalent",
    "rain_sum": "Daily liquid rain amount",
    "snowfall_sum": "Daily snowfall depth",
    "precipitation_hours": "Number of hours with precipitation",
    "wind_speed_10m_max": "Daily maximum wind speed at 10 m",
    "wind_direction_10m_dominant": "Dominant wind direction at 10 m; encoded as sine and cosine for modeling",
    "shortwave_radiation_sum": "Daily total shortwave solar radiation",
    "daylight_duration": "Duration of daylight",
    "weather_code": "Daily most severe WMO weather code; categorical metadata, excluded from model inputs",
}


def weather_types(frame, precipitation_threshold=1.0, cloud_threshold=40.0):
    """Precipitation takes precedence; missing measurements never become Clear."""
    precip = frame["precipitation_sum"]
    cloud = frame["cloud_cover_mean"]
    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    labels.loc[precip.ge(precipitation_threshold)] = "Precipitation"
    dry = precip.notna() & precip.lt(precipitation_threshold) & cloud.notna()
    labels.loc[dry & cloud.ge(cloud_threshold)] = "Cloudy"
    labels.loc[dry & cloud.lt(cloud_threshold)] = "Clear"
    return labels


def _session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def collect_city(city, period, config, raw_dir, offline=False, refresh=False):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"{city['id']}_{period['start']}_{period['end']}.json"
    params = {
        "latitude": city["latitude"], "longitude": city["longitude"],
        "start_date": period["start"], "end_date": period["end"],
        "daily": ",".join(DAILY_VARIABLES), "models": config["model"],
        "timezone": config["timezone"], "temperature_unit": "celsius",
        "wind_speed_unit": "kmh", "precipitation_unit": "mm",
    }
    if offline and refresh:
        raise ValueError("--offline and --refresh cannot be combined")
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
        if payload.get("query") != params:
            raise ValueError(f"Cache parameters differ from config: {cache}; use --refresh")
        print(f"  {city['name']}: cached API response", flush=True)
    else:
        # A full-period cache can also serve a shorter smoke run offline.
        payload = None
        if not refresh:
            for candidate in sorted(raw_dir.glob(f"{city['id']}_*.json")):
                saved = json.loads(candidate.read_text())
                q = saved.get("query", {})
                fixed = [k for k in params if k not in {"start_date", "end_date"}]
                if all(q.get(k) == params[k] for k in fixed) and q.get("start_date", "9999") <= period["start"] and q.get("end_date", "") >= period["end"]:
                    payload = saved
                    cache = candidate
                    print(f"  {city['name']}: using covering cache", flush=True)
                    break
        if payload is None:
            if offline:
                raise FileNotFoundError(f"No matching cached response for {city['name']}; run once without --offline")
            print(f"  {city['name']}: requesting {period['start']} to {period['end']}", flush=True)
            with _session() as session:
                for attempt in range(5):
                    response = session.get(API_URL, params=params, timeout=(15, 180))
                    if response.status_code != 429:
                        break
                    try:
                        reason = response.json().get("reason", response.text[:160])
                    except ValueError:
                        reason = response.text[:160]
                    if attempt == 4 or "daily" in reason.lower() or "hourly" in reason.lower():
                        raise RuntimeError(f"Open-Meteo rate limit: {reason}. Cached cities are retained; rerun later.")
                    delay = max(1, min(int(response.headers.get("Retry-After", "60")), 60))
                    print(f"  API rate limit ({reason}); waiting {delay}s, then retrying {city['name']}", flush=True)
                    time.sleep(delay)
                response.raise_for_status()
                js = response.json()
            if js.get("error") or "daily" not in js:
                raise ValueError(f"API error for {city['name']}: {js.get('reason', 'missing daily data')}")
            payload = {"query": params, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                       "api_url": API_URL, "response": js}
            temp = cache.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False))
            temp.replace(cache)
    js = payload["response"]
    expected_columns = {"time", *DAILY_VARIABLES}
    if set(js["daily"]) != expected_columns:
        raise ValueError(f"Unexpected API schema for {city['name']}: {set(js['daily']) ^ expected_columns}")
    daily = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.loc[daily["date"].between(period["start"], period["end"])].copy()
    for col in DAILY_VARIABLES:
        daily[col] = pd.to_numeric(daily[col], errors="raise")
    dates = pd.date_range(period["start"], period["end"])
    if daily["date"].duplicated().any() or not daily["date"].reset_index(drop=True).equals(pd.Series(dates)):
        raise ValueError(f"API dates are incomplete, duplicated or unordered for {city['name']}")
    if (daily["temperature_2m_max"] < daily["temperature_2m_min"]).any():
        raise ValueError(f"Daily maximum below minimum in {city['name']}")
    if (daily[["precipitation_sum", "rain_sum", "snowfall_sum"]] < 0).any().any():
        raise ValueError(f"Negative precipitation in {city['name']}")
    daily.insert(0, "city", city["name"])
    daily["temp_range"] = daily["temperature_2m_max"] - daily["temperature_2m_min"]
    daily["weather_type"] = weather_types(daily, config["precipitation_threshold_mm"], config["cloud_threshold_percent"])
    quality = {"city": city["name"], "rows": len(daily), "api_variable_count": len(DAILY_VARIABLES),
               "start_date": daily["date"].min().date().isoformat(), "end_date": daily["date"].max().date().isoformat(),
               "missing_weather_values": int(daily[DAILY_VARIABLES].isna().sum().sum()), "duplicate_dates": 0,
               "requested_latitude": city["latitude"], "requested_longitude": city["longitude"],
               "grid_latitude": js["latitude"], "grid_longitude": js["longitude"],
               "grid_elevation_m": js.get("elevation"), "timezone": config["timezone"],
               "cache_file": str(cache.name), "fetched_at_utc": payload["fetched_at_utc"]}
    return daily, quality, js.get("daily_units", {})


def build_supervised(daily):
    """Pair day t predictors with day t+1 outcomes within the same city."""
    frames = []
    features = [c for c in DAILY_VARIABLES if c not in {"weather_code", "wind_direction_10m_dominant"}]
    features += ["temp_range", "wind_direction_sin", "wind_direction_cos"]
    features += [f"{v}_lag_{lag}" for v in ["temperature_2m_mean", "precipitation_sum"] for lag in [1, 2, 7]]
    features += ["temperature_rolling_7", "precipitation_rolling_7", "target_doy_sin", "target_doy_cos"]
    for city, group in daily.groupby("city", sort=True):
        frame = group.sort_values("date").copy()
        if frame["date"].duplicated().any() or not frame["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Dates must be unique and consecutive within {city}")
        frame["target_date"] = frame["date"] + pd.Timedelta(days=1)
        frame["target_mean_temperature"] = frame["temperature_2m_mean"].shift(-1)
        frame["target_temperature_range"] = frame["temp_range"].shift(-1)
        frame["target_weather_type"] = frame["weather_type"].shift(-1)
        frame["persistence_temperature"] = frame["temperature_2m_mean"]
        frame["persistence_range"] = frame["temp_range"]
        frame["persistence_weather"] = frame["weather_type"]
        direction = np.deg2rad(frame["wind_direction_10m_dominant"])
        frame["wind_direction_sin"] = np.sin(direction)
        frame["wind_direction_cos"] = np.cos(direction)
        for variable in ["temperature_2m_mean", "precipitation_sum"]:
            for lag in [1, 2, 7]:
                frame[f"{variable}_lag_{lag}"] = frame[variable].shift(lag)
        frame["temperature_rolling_7"] = frame["temperature_2m_mean"].rolling(7, min_periods=7).mean()
        frame["precipitation_rolling_7"] = frame["precipitation_sum"].rolling(7, min_periods=7).mean()
        phase = 2 * np.pi * (frame["target_date"].dt.dayofyear - 1) / 365.25
        frame["target_doy_sin"], frame["target_doy_cos"] = np.sin(phase), np.cos(phase)
        # Every model and baseline uses identical samples. Exclude unlabeled rows
        # and unavailable persistence values; other predictors can be imputed.
        frame = frame.iloc[7:].dropna(subset=[
            "target_mean_temperature", "target_temperature_range", "target_weather_type",
            "persistence_temperature", "persistence_range", "persistence_weather",
        ])
        frames.append(frame)
    if not frames:
        raise ValueError("No city data to build supervised samples")
    supervised = pd.concat(frames, ignore_index=True)
    if supervised.empty:
        raise ValueError("No labeled rows remain after warmup and next-day alignment")
    return supervised, features


def data_dictionary(daily, supervised, feature_columns, api_units, config):
    records = []
    for column in dict.fromkeys([*daily.columns, *supervised.columns]):
        unit = api_units.get(column, "unitless")
        role = "predictor" if column in feature_columns else "metadata"
        description = DESCRIPTIONS.get(column, "")
        availability = "Finalized day t reanalysis; retrospective input only"
        if column == "city":
            description = "One of the configured seven cities; models fitted separately per city"
            unit = "category"
        elif column == "split":
            description, unit, availability = "Train, validation or test, assigned by target_date", "category", "Known evaluation partition; excluded from predictors"
        elif column in {"date", "target_date"}:
            description = "UTC input day t" if column == "date" else "UTC target day t+1; split boundaries use this date"
            unit = "ISO date"
        elif column in {"weather_type", "target_weather_type", "persistence_weather"}:
            description = f"Precipitation if total >= {config['precipitation_threshold_mm']} mm; else Cloudy if cloud >= {config['cloud_threshold_percent']}%; else Clear; missing input remains missing"
            unit = "category"
        elif column.startswith("target_") and column not in {"target_doy_sin", "target_doy_cos"}:
            description = description or ("Next-day mean temperature" if column == "target_mean_temperature" else "Next-day maximum minus next-day minimum temperature")
            role, availability = "target", "Day t+1 outcome, never a model input"
            if column != "target_weather_type":
                unit = "°C"
        if column.startswith("persistence_"):
            role = "baseline input"
            description = description or "Current-day value of the corresponding target"
            if column != "persistence_weather":
                unit = "°C"
        if "_lag_" in column:
            variable, lag = column.rsplit("_lag_", 1)
            description = f"{DESCRIPTIONS[variable]} at t minus {lag} days"
            unit = api_units[variable]
            availability = f"Day t-{lag}"
        if column.endswith("rolling_7"):
            description = "Mean over t-6 through t, inclusive; no centered or future window"
            unit = "°C" if column.startswith("temperature") else "mm/day"
        if column in {"target_doy_sin", "target_doy_cos"}:
            description, availability = "Cyclic day-of-year feature for target date using period 365.25 days", "Known calendar information"
        if column == "temp_range":
            description, unit = "Current-day maximum minus minimum temperature", "°C"
        if column.startswith("wind_direction_") and column != "wind_direction_10m_dominant":
            description = "Sine or cosine of current dominant wind direction in radians"
        if column == "target_weather_type":
            role, availability = "target", "Day t+1 outcome, never a model input"
        records.append({"column": column, "unit": unit, "role": role, "description": description, "availability": availability})
    return pd.DataFrame(records)
