# Data dictionary

## Collections

Both collections request ERA5 through the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) at Brown University (**41.82682, -71.40293**). The modeling response selects grid **41.75, -71.50**, elevation **42 m**. These are gridded reanalysis values, not campus station measurements.

| Collection | Inclusive dates | Rows | Weather fields | Total columns |
| --- | --- | ---: | ---: | ---: |
| Modeling raw API daily response | 2001-01-01–2025-12-31 | 9,131 | 15 | 16, including `time` |
| Modeling `data/processed/daily.csv` | Same period | 9,131 | 15 | 19, including metadata and derived fields |
| EDA `team_project_bill/data/BROWN-2006-2025.csv` | 2006-01-01–2025-01-01 | 6,941 | 19 | 20, including `date` |

The timezone request is `America/New_York`. The modeling cache and collection metadata retain the query, retrieval timestamp, units, returned grid, quality checks, and `utc_offset_seconds` (**-14400** in this response), documenting the API's daily boundaries. The modeling daily dates are complete and unique, with no missing requested weather values.

## Weather fields and units

The modeling request explicitly selects Celsius, km/h, and mm. All fields below are numeric measurements except the categorical WMO code. Rounded integers do not make humidity or cloud cover ordinal categories.

| Modeling field | Meaning | Unit / type |
| --- | --- | --- |
| `temperature_2m_mean` | Daily mean air temperature at 2 m | °C |
| `temperature_2m_max` | Daily maximum air temperature at 2 m | °C |
| `temperature_2m_min` | Daily minimum air temperature at 2 m | °C |
| `relative_humidity_2m_mean` | Mean relative humidity at 2 m | % |
| `pressure_msl_mean` | Mean sea-level pressure | hPa |
| `cloud_cover_mean` | Mean total cloud coverage | % |
| `precipitation_sum` | Total precipitation, including snow water equivalent | mm |
| `rain_sum` | Liquid rainfall total | mm |
| `snowfall_sum` | Snowfall depth total | cm |
| `precipitation_hours` | Precipitation duration | h |
| `wind_speed_10m_max` | Maximum wind speed at 10 m | km/h |
| `wind_direction_10m_dominant` | Dominant wind direction; circular, 0° = 360° | degrees |
| `shortwave_radiation_sum` | Incoming shortwave energy per area | MJ/m² |
| `daylight_duration` | Daylight length | seconds |
| `weather_code` | Most severe daily WMO condition | category |

The EDA collection includes these 15 fields and four extras, which are **not** included in the current 1,073-predictor selection:

| EDA-only field | Meaning | Unit |
| --- | --- | --- |
| `surface_pressure_mean` | Mean pressure at the surface | hPa |
| `wind_gusts_10m_max` | Maximum wind gust at 10 m | km/h |
| `et0_fao_evapotranspiration` | Reference evapotranspiration total | mm |
| `sunshine_duration` | Sunshine length | seconds |

## Dates, derived fields, and categories

API `time` is renamed `date`. It identifies the observation day in the daily tables; in prepared samples, `date` means the last input-day boundary, d−1. `city` is the location category, currently `Brown University`. `temp_range` is maximum minus minimum temperature (°C).

`weather_type` is an EDA-only derived category, evaluated in order: total precipitation ≥1 mm → `Precipitation`; otherwise cloud cover ≥40% → `Cloudy`; otherwise → `Clear`. Required missing measurements remain missing. These thresholds are in `config.json`; this label is not y.

Both collected datasets contain the following **13 WMO codes**. Codes are nominal labels, not an ordinal severity scale. Prepared code values are integer strings or missing `NaN`, later one-hot encoded.

| Codes | Meanings, respectively |
| --- | --- |
| 0; 1; 2; 3 | Clear; mainly clear; partly cloudy; overcast |
| 51; 53; 55 | Drizzle: light; moderate; dense |
| 61; 63; 65 | Rain: slight; moderate; heavy |
| 71; 73; 75 | Snowfall: slight; moderate; heavy |

These are observed categories; the [API's complete code table](https://open-meteo.com/en/docs/historical-weather-api) also lists other model-dependent possibilities.

## Prepared samples

For target d, retain 28 recent observations (d−1 through d−28) and 35 prior-year observations: five years × offsets −3 through +3 around the corresponding month/day. February 29 anchors become February 28 in non-leap years. Windows may cross year boundaries; all source dates must precede d. Missing context dates exclude a sample; missing predictor measurements remain missing.

Each position contains the 15 modeling fields plus unitless `calendar_sin` and `calendar_cos`. For source month/day s, θ = 2π × (day-of-year of s in leap year 2000 − 1) / 366. Store sin(θ) and cos(θ). `target_calendar_sin` and `target_calendar_cos` apply the same formula to d; these calendar inputs are known in advance.

Names follow `recent_lag_01__<field>` through lag 28, or `history_y1_offset_m3__<field>` through year 5. Offset names `m3`, `m2`, `m1`, `z0`, `p1`, `p2`, `p3` mean −3 through +3 days. Measurements retain their original units.

Raw X has **63 × 17 + 2 = 1,073** columns, excluding metadata and y. Generated `feature_columns.json` identifies X. Wind sine/cosine conversion and WMO one-hot encoding later change its dimension.

| Other sample column | Meaning / type |
| --- | --- |
| `city` | Location category |
| `date` | Last input-day boundary, d−1; ISO date |
| `target_date` | Predicted day d; ISO date |
| `recent_days` | Recent context count: 28; integer days |
| `history_days` | Prior-year context count: 35; integer days |
| `split` | `train`, `validation`, or `test`, assigned by target date |
| `target_mean_temperature` | y: actual daily mean temperature on d, °C; never an input |

There are **7,302 eligible targets**: training **2006-01-04–2017-12-31 (4,380)**, validation **2018–2021 (1,461)**, and testing **2022–2025 (1,461)**. `prepare` generates the sample matrix locally; it is not committed. Full-period Bill EDA has already inspected the exploratory years, including validation and test years.

Weather data by [Open-Meteo.com](https://open-meteo.com/) and Copernicus ERA5 under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Features and labels are project transformations.
