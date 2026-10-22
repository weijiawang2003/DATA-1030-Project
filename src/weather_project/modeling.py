"""Unfitted models for numeric weather, WMO categories, and circular wind."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

WEATHER_SUFFIX = "__weather_code"
WIND_SUFFIX = "__wind_direction_10m_dominant"


def _weather_columns(frame):
    return [name for name in frame.columns if name.endswith(WEATHER_SUFFIX)]


def _wind_columns(frame):
    return [name for name in frame.columns if name.endswith(WIND_SUFFIX)]


def _numeric_columns(frame):
    return [name for name in frame.columns if not name.endswith((WEATHER_SUFFIX, WIND_SUFFIX))]


def _canonical_weather_codes(values):
    result = np.asarray(values, dtype=object).copy()
    for index, value in np.ndenumerate(result):
        if pd.isna(value):
            result[index] = np.nan
            continue
        try:
            number = float(value)
            result[index] = str(int(number)) if np.isfinite(number) and number.is_integer() else str(value)
        except (TypeError, ValueError):
            result[index] = str(value)
    return result


def _wind_sin_cos(values):
    radians = np.deg2rad(np.asarray(values, dtype=float))
    return np.column_stack([np.sin(radians), np.cos(radians)])


def _wind_feature_names(transformer, input_features):
    return [f"{name}_{part}" for part in ("sin", "cos") for name in input_features]


def _preprocessor(scale):
    numeric = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    # Encode circular angles before imputing; 359 and 1 degrees are neighbors.
    wind = [
        ("circular", FunctionTransformer(_wind_sin_cos, feature_names_out=_wind_feature_names)),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale:
        numeric.append(("scaler", StandardScaler()))
        wind.append(("scaler", StandardScaler()))
    weather = Pipeline([
        ("canonical_codes", FunctionTransformer(_canonical_weather_codes, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric), _numeric_columns),
        ("weather", weather, _weather_columns),
        ("wind", Pipeline(wind), _wind_columns),
    ], remainder="drop")


def build_models(task="regression", seed=1030):
    """Return two unfitted pipelines; suffixes determine feature treatment."""
    if task == "regression":
        linear = Ridge(alpha=1.0)
        forest = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        linear_name = "ridge"
    elif task == "classification":
        linear = LogisticRegression(max_iter=1000, random_state=seed)
        forest = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        linear_name = "logistic_regression"
    else:
        raise ValueError("task must be 'regression' or 'classification'")
    return {
        linear_name: Pipeline([
            ("preprocessing", _preprocessor(scale=True)),
            ("model", linear),
        ]),
        "random_forest": Pipeline([
            ("preprocessing", _preprocessor(scale=False)),
            ("model", forest),
        ]),
    }
