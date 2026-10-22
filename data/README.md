# Data dictionary

This describes the configured schema, not collected results: **2001-01-01 through 2025-12-31**, Brown University (**41.82682, -71.40293**), `models=era5` through the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). ERA5 is gridded reanalysis. Requested and returned coordinates are recorded separately.

Timezone is `America/New_York`. Retain `utc_offset_seconds`: the API documents fixed-offset boundaries, not DST-exact historical local days. Metadata records the query, retrieval timestamp, units, grid, and quality checks. Observed counts, missingness, and categories require collection.

## Daily weather fields

Requests select Celsius, km/h, and mm. These **15 fields** appear at every historical position. Measurements are continuous quantities even when returned as rounded integers; `weather_code` is categorical.

| Exact field | Meaning | Unit / type |
| --- | --- | --- |
| `temperature_2m_mean` | Mean daily air temperature, 2 m | °C |
| `temperature_2m_max` | Maximum daily air temperature, 2 m | °C |
| `temperature_2m_min` | Minimum daily air temperature, 2 m | °C |
| `relative_humidity_2m_mean` | Mean relative humidity, 2 m | % |
| `pressure_msl_mean` | Mean sea-level pressure | hPa |
| `cloud_cover_mean` | Mean total cloud coverage | % |
| `precipitation_sum` | Total precipitation, including snow water equivalent | mm |
| `rain_sum` | Liquid rainfall total | mm |
| `snowfall_sum` | Snowfall depth total | cm |
| `precipitation_hours` | Hours with precipitation | h |
| `wind_speed_10m_max` | Maximum wind speed, 10 m | km/h |
| `wind_direction_10m_dominant` | Dominant wind direction; cyclic, 0° = 360° | degrees |
| `shortwave_radiation_sum` | Daily incoming shortwave energy per area | MJ/m² |
| `daylight_duration` | Daylight length | seconds |
| `weather_code` | Most severe daily WMO weather category | categorical code |

API `time` is renamed `date` during collection. Additional daily columns are `city` (category, currently Brown University), `temp_range` (maximum minus minimum temperature, °C), and `weather_type` (derived category). Apply these rules in order: precipitation at least **1 mm** → `Precipitation`; otherwise cloud coverage at least **40%** → `Cloudy`; otherwise → `Clear`. Missing required measurements remain missing. These configurable labels are for EDA, not the current regression target.

## WMO categories

The [official API code table](https://open-meteo.com/en/docs/historical-weather-api) defines these possibilities, **not observed categories**. Availability depends on the model; some may never occur in ERA5.

| Codes | Categories, respectively |
| --- | --- |
| 0; 1; 2; 3 | Clear; mostly clear; partly cloudy; overcast |
| 45; 48 | Fog; fog depositing rime |
| 51; 53; 55 | Drizzle: light; moderate; dense |
| 56; 57 | Freezing drizzle: light; dense |
| 61; 63; 65 | Rain: slight; moderate; heavy |
| 66; 67 | Freezing rain: light; heavy |
| 71; 73; 75 | Snowfall: slight; moderate; heavy |
| 77 | Snow grains |
| 80; 81; 82 | Rain showers: slight; moderate; violent |
| 85; 86 | Snow showers: slight; heavy |
| 95; 97 | Thunderstorm; heavy thunderstorm |
| 96; 99 | Thunderstorm with slight hail; with heavy hail |

Codes have no ordinal scale. Prepared values are integer strings or missing `NaN`; preprocessing later applies one-hot encoding.

## Prepared samples and feature names

For target day **d**, defaults retain **28 recent days** (d−1 through d−28) and **35 prior-year days**: five previous years, each contributing offsets −3 through +3 around d's month/day. A February 29 anchor becomes February 28 in a non-leap year. Windows may cross year boundaries; every source date must precede d. Incomplete date windows and non-finite targets are excluded; missing predictor measurements remain missing.

Every position stores the 15 fields above plus `calendar_sin` and `calendar_cos`. For source month/day s, θ = 2π × (day-of-year of s in leap year 2000 − 1) / 366; store sin(θ), cos(θ), both unitless. `target_calendar_sin` and `target_calendar_cos` use the same formula for d and are known calendar inputs.

Column templates are `recent_lag_01__<field>` through `recent_lag_28__<field>`, and `history_y1_offset_m3__<field>` through year 5. Offset suffixes `m3`, `m2`, `m1`, `z0`, `p1`, `p2`, `p3` mean −3, −2, −1, 0, +1, +2, +3 days. Fields retain their original units.

Thus raw **X has (28 + 5×7)×17 + 2 = 1,073 columns**. It excludes metadata and y. Later wind sine/cosine conversion and WMO one-hot encoding change the transformed column count.

| Other sample column | Meaning / type |
| --- | --- |
| `city` | Location identifier; category |
| `date` | Last input-day boundary, d−1; ISO date |
| `target_date` | Predicted day d; ISO date |
| `recent_days` | Recent context count, normally 28; integer days |
| `history_days` | Prior-year context count, normally 35; integer days |
| `split` | `train`, `validation`, or `test`, assigned by target date; category |
| `target_mean_temperature` | y: actual mean temperature on d, °C; never an input |

Current split cutoffs are training through 2019, validation 2020–2022, and testing 2023–2025, subject to available complete windows. Finalized reanalysis inputs define a retrospective experiment, not demonstrated live availability. Weather data by [Open-Meteo.com](https://open-meteo.com/) and Copernicus ERA5, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); features and labels are project transformations.
