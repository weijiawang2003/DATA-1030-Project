import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


def get_API(LAT, LON, START, END, KEYS, TIME_ZONE="America/New_York",
            temperature_unit="celsius", wind_speed_unit="kmh", precipitation_unit="mm",
            MODELS="era5"):
    """Download daily weather data from the Open-Meteo Historical Weather API
    and return it as a DataFrame (one row per day, one column per key)."""

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # The order of variables in daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "daily": KEYS,
        "models": MODELS,
        "timezone": TIME_ZONE,
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "precipitation_unit": precipitation_unit,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }

    for i, key in enumerate(KEYS):
        daily_data[key] = daily.Variables(i).ValuesAsNumpy()

    daily_dataframe = pd.DataFrame(data=daily_data)
    daily_dataframe["date"] = daily_dataframe["date"].dt.date

    return daily_dataframe
