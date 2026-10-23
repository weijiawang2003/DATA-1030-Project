# %% [markdown]
# # Brown University 日度天气数据：次日预测（2001–2025）
#
# - 数据源：Open-Meteo Historical Weather API（`archive-api.open-meteo.com`，免费、无需 key）
# - 模型：ERA5 再分析（`models=era5`，0.25° 网格，25 年同一套模型，避免数据源切换）
# - 地点：Brown University，41.82682° N, -71.40293° W（latlong.net）；API 实际返回最近的 ERA5 格点 41.75, -71.50
# - 时间：2001-01-01 至 2025-12-31，日数据，按 America/New_York 当地日切分
# - 单位：全部使用 API 默认单位（°C、km/h、mm、cm、hPa）
# - 数据输出：`data/BROWN-2001-2025.csv`
# - 图片输出：`figures/EDA/`（`single_attribute_EDA/`、`correlation_EDA/`、`XX_EDA/`、`Xy_EDA/`）

import pandas as pd
import os 
import openmeteo_requests
import requests_cache
from retry_requests import retry
from src.utils import create_folder, singular_attribute_EDA_all
from src.data_api import get_API
from src.test_dataframe import dataframe_test
from src.EDA import singular_attribute_EDA, correlation_EDA, EDA00

# %% Variables
# inputs



LAT, LON = 41.82682, -71.40293
START, END = "2006-01-01", "2025-01-01"

folder_name = "data"
file_name = "BROWN"+START+"-"+END+"".csv"

figure_folder = "figures"
EDA_folder = "figures/EDA"
temperature_folder = "figures/EDA/temperature"     # EDA of X1 y1
weather_code_folder = "figures/EDA/weather_code"   # EDA of X2 y2

single_attribute_EDA_folder = weather_code_folder + "/single_attribute_EDA"
correlation_EDA_folder = weather_code_folder + "/correlation_EDA"
XX_EDA_folder = weather_code_folder + "/XX_EDA"
Xy_EDA_folder = weather_code_folder + "/Xy_EDA"
time_series_EDA_folder = weather_code_folder + "/time_series_EDA"


KEYS = [
    "weather_code",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "pressure_msl_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "daylight_duration",
]

# readable name of each key, with unit if it has one
names = {
    "weather_code": "WMO weather code",
    "temperature_2m_mean": "Mean temperature (°C)",
    "temperature_2m_max": "Max temperature (°C)",
    "temperature_2m_min": "Min temperature (°C)",
    "cloud_cover_mean": "Mean cloud cover (%)",
    "relative_humidity_2m_mean": "Mean relative humidity (%)",
    "surface_pressure_mean": "Mean surface pressure (hPa)",
    "pressure_msl_mean": "Mean sea-level pressure (hPa)",
    "precipitation_sum": "Precipitation (mm)",
    "rain_sum": "Rain (mm)",
    "snowfall_sum": "Snowfall (cm)",
    "precipitation_hours": "Precipitation hours (h)",
    "wind_speed_10m_max": "Max wind speed (km/h)",
    "wind_gusts_10m_max": "Max wind gust (km/h)",
    "wind_direction_10m_dominant": "Dominant wind direction (°)",
    "shortwave_radiation_sum": "Shortwave radiation (MJ/m²)",
    "et0_fao_evapotranspiration": "Reference evapotranspiration ET0 (mm)",
    "sunshine_duration": "Sunshine duration (s)",
    "daylight_duration": "Daylight duration (s)",
}

# settings for singular_attribute_EDA, one entry per key
singular_attribute_EDA_input = {
    "weather_code": {
        "kind": "categorical",
        "bin_set": [0, 1, 2, 3, 51, 53, 55, 61, 63, 65, 71, 73, 75],
        "title": names["weather_code"],
        "xlabel": names["weather_code"],
        "ylabel": "count (days)",
        "log_y": True,
        "description": "WMO weather code of each day. 0 clear, 1 mainly clear, 2 partly cloudy, 3 overcast; "
                       "51/53/55 drizzle (light/moderate/dense); 61/63/65 rain (slight/moderate/heavy); "
                       "71/73/75 snow (slight/moderate/heavy).",},
    "temperature_2m_mean": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["temperature_2m_mean"],
        "xlabel": names["temperature_2m_mean"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily mean air temperature at 2 m above ground, °C.",},
    "temperature_2m_max": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["temperature_2m_max"],
        "xlabel": names["temperature_2m_max"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily maximum air temperature at 2 m above ground, °C.",},
    "temperature_2m_min": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["temperature_2m_min"],
        "xlabel": names["temperature_2m_min"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily minimum air temperature at 2 m above ground, °C.",},
    "cloud_cover_mean": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["cloud_cover_mean"],
        "xlabel": names["cloud_cover_mean"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily mean total cloud cover, % of the sky.",},
    "relative_humidity_2m_mean": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["relative_humidity_2m_mean"],
        "xlabel": names["relative_humidity_2m_mean"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily mean relative humidity at 2 m above ground, %.",},
    "surface_pressure_mean": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["surface_pressure_mean"],
        "xlabel": names["surface_pressure_mean"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily mean air pressure at the surface, hPa.",},
    "pressure_msl_mean": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["pressure_msl_mean"],
        "xlabel": names["pressure_msl_mean"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily mean air pressure reduced to mean sea level, hPa.",},
    "precipitation_sum": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["precipitation_sum"],
        "xlabel": names["precipitation_sum"],
        "ylabel": "count (days, log scale)",
        "log_y": True,
        "description": "Daily total precipitation (rain + snow water equivalent), mm. About half of the days are 0.",},
    "rain_sum": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["rain_sum"],
        "xlabel": names["rain_sum"],
        "ylabel": "count (days, log scale)",
        "log_y": True,
        "description": "Daily total liquid rain, mm.",},
    "snowfall_sum": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["snowfall_sum"],
        "xlabel": names["snowfall_sum"],
        "ylabel": "count (days, log scale)",
        "log_y": True,
        "description": "Daily total snowfall, cm. Most days have no snow.",},
    "precipitation_hours": {
        "kind": "continuous",
        "bin_set": [24],
        "title": names["precipitation_hours"],
        "xlabel": names["precipitation_hours"],
        "ylabel": "count (days, log scale)",
        "log_y": True,
        "description": "Number of hours in the day with precipitation, h (0-24).",},
    "wind_speed_10m_max": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["wind_speed_10m_max"],
        "xlabel": names["wind_speed_10m_max"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily maximum wind speed at 10 m above ground, km/h.",},
    "wind_gusts_10m_max": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["wind_gusts_10m_max"],
        "xlabel": names["wind_gusts_10m_max"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily maximum wind gust at 10 m above ground, km/h.",},
    "wind_direction_10m_dominant": {
        "kind": "continuous",
        "bin_set": [36],
        "title": names["wind_direction_10m_dominant"],
        "xlabel": names["wind_direction_10m_dominant"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Dominant wind direction of the day, degrees clockwise from north (0/360 = north, 90 = east).",},
    "shortwave_radiation_sum": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["shortwave_radiation_sum"],
        "xlabel": names["shortwave_radiation_sum"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily sum of incoming shortwave solar radiation, MJ/m².",},
    "et0_fao_evapotranspiration": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["et0_fao_evapotranspiration"],
        "xlabel": names["et0_fao_evapotranspiration"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Daily FAO-56 reference evapotranspiration (ET0), mm.",},
    "sunshine_duration": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["sunshine_duration"],
        "xlabel": names["sunshine_duration"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Seconds of sunshine in the day, s.",},
    "daylight_duration": {
        "kind": "continuous",
        "bin_set": [50],
        "title": names["daylight_duration"],
        "xlabel": names["daylight_duration"],
        "ylabel": "count (days)",
        "log_y": False,
        "description": "Seconds between sunrise and sunset, s.",},
}

# columns that use a log axis in Xy_EDA
Xy_EDA_input = [
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]

X1 = [
    "date",
    "weather_code",
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "pressure_msl_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "daylight_duration",
]

y1 = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
]

X1_continuous = [
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "pressure_msl_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "daylight_duration",
]

y1_continuous = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
]

X1_categorical = [
    "weather_code",
]

y1_categorical = []

X1_ordinal = []

y1_ordinal = []

X2 = [
    "date",
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "pressure_msl_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "daylight_duration",
]

y2 = [
    "weather_code",
]

X2_continuous = [
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "pressure_msl_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "daylight_duration",
]

y2_continuous = []

X2_categorical = []

y2_categorical = [
    "weather_code",
]

X2_ordinal = []

y2_ordinal = []

# %% API
create_folder("data")
create_folder(single_attribute_EDA_folder)
create_folder(correlation_EDA_folder)

dataset = get_API(LAT, LON, START, END, KEYS)

dataset.head()

dataset.to_csv(folder_name + "/" + file_name, index=False)

#EDA

# %% Input
data = pd.read_csv(folder_name + "/" + file_name)

# dates downloaded from the API
dates = list(data["date"])

data.head()

# %% EDA
dataframe_test(data)

singular_attribute_EDA_all(data, singular_attribute_EDA_input, single_attribute_EDA_folder)

correlation_EDA(data, X2_continuous, y2_continuous, "Absolute correlation: X2 vs y2", "X2_y2", correlation_EDA_folder, names)

eda = EDA00(data, names, dates)
eda.XX_EDA(X2_continuous, X2_categorical, X2_ordinal, ["precipitation_sum", "rain_sum", "snowfall_sum"], folder_name = XX_EDA_folder)
eda.Xy_EDA(X2_continuous, X2_categorical, X2_ordinal, y2_continuous, y2_categorical, y2_ordinal, Xy_EDA_input, folder_name = Xy_EDA_folder)
eda.timeseries_EDA(X2_continuous, X2_categorical, X2_ordinal, y2_continuous, y2_categorical, y2_ordinal, folder_name = time_series_EDA_folder)

# %% Split

# %% Preprocess

# %% Model

# %% CV

# %% Test

# %% Explanation
