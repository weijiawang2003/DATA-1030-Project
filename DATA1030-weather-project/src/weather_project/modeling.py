"""Per-city chronological model comparisons and forecast baselines."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


WEATHER_LABELS = ("Clear", "Cloudy", "Precipitation")
TARGETS = {
    "temperature": {
        "column": "target_mean_temperature",
        "persistence": ("persistence_temperature", "temperature_2m_mean"),
        "task": "regression",
    },
    "range": {
        "column": "target_temperature_range",
        "persistence": ("persistence_range", "temp_range"),
        "task": "regression",
    },
    "weather": {
        "column": "target_weather_type",
        "persistence": ("persistence_weather", "weather_type"),
        "task": "classification",
    },
}
METRIC_COLUMNS = [
    "city", "target", "model", "split", "n_samples", "mae", "rmse",
    "macro_f1", "recall_clear", "recall_cloudy", "recall_precipitation",
    "accuracy",
]
PREDICTION_COLUMNS = [
    "city", "date", "target_date", "target", "model", "split", "y_true", "y_pred",
]


def _date_series(values: pd.Series, name: str) -> pd.Series:
    try:
        result = pd.to_datetime(values, errors="raise")
        if result.dt.tz is not None:
            result = result.dt.tz_localize(None)
        result = result.dt.normalize()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{name} must contain consistent daily dates.") from exc
    if result.isna().any():
        raise ValueError(f"{name} contains missing dates.")
    return result


def _cutoff(value: str, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}.") from exc
    if pd.isna(result):
        raise ValueError(f"Invalid {name}: {value!r}.")
    return result.tz_localize(None).normalize()


def _numeric(values: pd.Series, name: str) -> pd.Series:
    try:
        result = pd.to_numeric(values, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} contains missing or infinite target/baseline values.")
    return result


def _weather(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} contains missing weather labels.")
    result = values.astype(str)
    unexpected = sorted(set(result) - set(WEATHER_LABELS))
    if unexpected:
        raise ValueError(
            f"{name} has unsupported labels {unexpected}; expected {list(WEATHER_LABELS)}."
        )
    return result


def _preprocessor(frame: pd.DataFrame, features: list[str], scale: bool) -> ColumnTransformer:
    numeric = list(frame[features].select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [name for name in features if name not in numeric]
    transformers = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def _families(task: str, include_svm: bool, seed: int) -> list[tuple]:
    """Two small configurations per family; selection uses validation only."""
    forest_common = {"n_estimators": 100, "random_state": seed, "n_jobs": -1}
    forest_configs = [
        {**forest_common, "max_depth": 6, "min_samples_leaf": 3},
        {**forest_common, "max_depth": 12, "min_samples_leaf": 5},
    ]
    if task == "regression":
        families = [
            ("ridge", Ridge, [{"alpha": 1.0}, {"alpha": 10.0}], True),
            ("random_forest", RandomForestRegressor, forest_configs, False),
        ]
        if include_svm:
            families.append(("svm", SVR, [
                {"kernel": "rbf", "C": c, "gamma": "scale", "epsilon": 0.1}
                for c in (1.0, 10.0)
            ], True))
    else:
        families = [
            ("logistic_regression", LogisticRegression, [
                {"C": c, "max_iter": 3000, "random_state": seed} for c in (0.1, 1.0)
            ], True),
            ("random_forest", RandomForestClassifier, forest_configs, False),
        ]
        if include_svm:
            families.append(("svm", SVC, [
                {"kernel": "rbf", "C": c, "gamma": "scale", "random_state": seed}
                for c in (1.0, 10.0)
            ], True))
    return families


def _scores(y_true, y_pred, task: str) -> dict:
    result = {key: np.nan for key in METRIC_COLUMNS[5:]}
    if task == "regression":
        result["mae"] = float(mean_absolute_error(y_true, y_pred))
        result["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        result["macro_f1"] = float(f1_score(
            y_true, y_pred, labels=list(WEATHER_LABELS), average="macro", zero_division=0,
        ))
        recalls = recall_score(
            y_true, y_pred, labels=list(WEATHER_LABELS), average=None, zero_division=0,
        )
        for key, value in zip(
            ("recall_clear", "recall_cloudy", "recall_precipitation"), recalls,
        ):
            result[key] = float(value)
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
    return result


def _selection_score(y_true, y_pred, task: str) -> float:
    if task == "regression":
        return float(mean_absolute_error(y_true, y_pred))
    return float(f1_score(
        y_true, y_pred, labels=list(WEATHER_LABELS), average="macro", zero_division=0,
    ))


def _better(score: float, current: float | None, task: str) -> bool:
    if current is None:
        return True
    return score < current if task == "regression" else score > current


def _mode(values: pd.Series) -> str:
    # A fixed label order resolves ties reproducibly, including unobserved labels.
    return str(values.value_counts().reindex(WEATHER_LABELS, fill_value=0).idxmax())


def _climatology(training: pd.DataFrame, holdout: pd.DataFrame, target: str, task: str):
    """Use the target month's mean/mode and a training-only global fallback."""
    groups = training.groupby(training["target_date"].dt.month)[target]
    if task == "regression":
        monthly = groups.mean()
        fallback = float(training[target].mean())
    else:
        monthly = groups.agg(_mode)
        fallback = _mode(training[target])
    return holdout["target_date"].dt.month.map(monthly).fillna(fallback).to_numpy()


def run_models(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: str | Path,
    train_end: str = "2019-12-31",
    validation_end: str = "2022-12-31",
    include_svm: bool = False,
    targets: Sequence[str] = ("temperature", "weather"),
    seed: int = 1030,
) -> pd.DataFrame:
    """Compare models within each city using fixed target-date holdouts.

    Required identifiers are ``city``, ``date`` and ``target_date`` (next day).
    Targets are ``temperature``, ``range`` and/or ``weather``; corresponding
    outcome columns are listed in ``TARGETS``. Persistence inputs use
    ``persistence_temperature``, ``persistence_range``, ``persistence_weather``;
    original source columns are accepted as fallbacks. Weather labels must be
    Clear, Cloudy or Precipitation. Numeric and text categorical features are
    supported; imputation, scaling and encoding are fitted inside pipelines.

    Each family chooses parameters on validation, is refitted on train plus
    validation, then is scored on test. The saved family is chosen solely by
    validation MAE (regression) or macro-F1 (classification), excluding baselines.
    Writes metrics.csv, predictions.csv, model_selection.json and one best
    joblib pipeline per city/target. Every city must have all three time splits;
    a classification training split must contain at least two classes.
    """
    features = list(feature_columns)
    target_names = [targets] if isinstance(targets, str) else list(targets)
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be nonempty and contain no duplicates.")
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("targets must be nonempty and contain no duplicates.")
    unknown = set(target_names) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown targets {sorted(unknown)}; choose {list(TARGETS)}.")
    forbidden = {"city", "date", "target_date"} | {
        spec["column"] for spec in TARGETS.values()
    }
    leaked = set(features) & forbidden
    if leaked:
        raise ValueError(f"Identifiers and target columns cannot be features: {sorted(leaked)}.")
    required = {"city", "date", "target_date", *features} | {
        TARGETS[name]["column"] for name in target_names
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing modeling columns: {sorted(missing)}.")
    if df.empty or df["city"].isna().any():
        raise ValueError("Modeling data must be nonempty with a city on every row.")
    if not df.columns.is_unique:
        raise ValueError("Modeling data contains duplicate column names.")
    training_cutoff = _cutoff(train_end, "train_end")
    validation_cutoff = _cutoff(validation_end, "validation_end")
    if training_cutoff >= validation_cutoff:
        raise ValueError("train_end must precede validation_end.")

    frame = df.copy()
    frame["city"] = frame["city"].astype(str)
    frame["date"] = _date_series(frame["date"], "date")
    frame["target_date"] = _date_series(frame["target_date"], "target_date")
    if not (frame["target_date"] == frame["date"] + pd.Timedelta(days=1)).all():
        raise ValueError("Each target_date must be exactly one day after its feature date.")
    if frame.duplicated(["city", "target_date"]).any():
        raise ValueError("Duplicate city/target_date rows would distort evaluation.")
    persistence_columns = {}
    for name in target_names:
        spec = TARGETS[name]
        persistence = next((col for col in spec["persistence"] if col in frame), None)
        if persistence is None:
            raise ValueError(f"Target {name!r} requires one of {spec['persistence']} for persistence.")
        persistence_columns[name] = persistence
        validate = _weather if spec["task"] == "classification" else _numeric
        for column in (spec["column"], persistence):
            frame[column] = validate(frame[column], column)
    for column in features:
        if pd.api.types.is_datetime64_any_dtype(frame[column].dtype):
            raise ValueError(f"Encode datetime feature {column!r} as calendar features first.")
        if pd.api.types.is_numeric_dtype(frame[column].dtype):
            frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)
        else:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    frame = frame.sort_values(["city", "target_date"]).reset_index(drop=True)

    city_splits = {}
    for city, city_frame in frame.groupby("city", sort=True):
        splits = {
            "train": city_frame.loc[city_frame["target_date"] <= training_cutoff].copy(),
            "validation": city_frame.loc[
                (city_frame["target_date"] > training_cutoff)
                & (city_frame["target_date"] <= validation_cutoff)
            ].copy(),
            "test": city_frame.loc[city_frame["target_date"] > validation_cutoff].copy(),
        }
        empty = [split for split, values in splits.items() if values.empty]
        if empty:
            raise ValueError(
                f"City {city!r} has empty splits {empty}; check coverage and date cutoffs."
            )
        if "weather" in target_names and splits["train"]["target_weather_type"].nunique() < 2:
            raise ValueError(f"City {city!r} needs at least two weather classes in training.")
        city_splits[city] = splits

    destination = Path(output_dir)
    model_dir = destination / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_frames = [], []
    metadata = {
        "seed": seed,
        "train_end": training_cutoff.date().isoformat(),
        "validation_end": validation_cutoff.date().isoformat(),
        "split_basis": "target_date",
        "feature_columns": features,
        "targets": target_names,
        "weather_labels": list(WEATHER_LABELS),
        "include_svm": include_svm,
        "versions": {
            "python": sys.version.split()[0], "scikit-learn": sklearn_version,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
        "selection": "Validation MAE minimized or validation macro-F1 maximized; ties use listed order.",
        "classification_zero_division": 0,
        "classification_note": "Macro-F1 includes all three labels even when a holdout lacks a class.",
        "climatology": "Per-city target-month mean/mode; unseen months use training global mean/mode.",
        "cities": [],
    }

    def record(city, target_name, model, split, holdout, column, predictions, task):
        predictions = np.asarray(predictions)
        metric_rows.append({
            "city": city, "target": target_name, "model": model, "split": split,
            "n_samples": len(holdout), **_scores(holdout[column], predictions, task),
        })
        values = holdout[["city", "date", "target_date"]].copy()
        values["target"] = target_name
        values["model"] = model
        values["split"] = split
        values["y_true"] = holdout[column].to_numpy()
        values["y_pred"] = predictions
        prediction_frames.append(values[PREDICTION_COLUMNS])

    for city, splits in city_splits.items():
        train, validation, test = (splits[key] for key in ("train", "validation", "test"))
        development = pd.concat([train, validation], ignore_index=True)
        city_metadata = {
            "city": city,
            "splits": {
                name: {
                    "n_samples": len(values),
                    "first_target_date": values["target_date"].min().date().isoformat(),
                    "last_target_date": values["target_date"].max().date().isoformat(),
                } for name, values in splits.items()
            },
            "targets": [],
        }
        for name in target_names:
            print(f"  Modeling {city}: {name}", flush=True)
            spec = TARGETS[name]
            column, task = spec["column"], spec["task"]
            target_metadata = {
                "target": name, "column": column, "persistence_column": persistence_columns[name],
                "selection_metric": "mae" if task == "regression" else "macro_f1",
                "families": [],
            }
            for split, fitting, holdout in (
                ("validation", train, validation), ("test", development, test),
            ):
                record(city, name, "persistence", split, holdout, column,
                       holdout[persistence_columns[name]], task)
                record(city, name, "climatology", split, holdout, column,
                       _climatology(fitting, holdout, column, task), task)

            selected_score, selected_model = None, None
            selected_name, selected_params = None, None
            for family, estimator_class, configurations, scale in _families(task, include_svm, seed):
                family_score, family_params, family_prediction = None, None, None
                trials = []
                for params in configurations:
                    candidate = Pipeline([
                        ("preprocessing", _preprocessor(train, features, scale)),
                        ("model", estimator_class(**params)),
                    ])
                    try:
                        candidate.fit(train[features], train[column])
                        prediction = candidate.predict(validation[features])
                    except (ValueError, TypeError) as exc:
                        raise ValueError(
                            f"Model fit failed for city={city!r}, target={name!r}, "
                            f"family={family!r}, params={params}: {exc}"
                        ) from exc
                    score = _selection_score(validation[column], prediction, task)
                    trials.append({"params": params, "validation_score": score})
                    if _better(score, family_score, task):
                        family_score, family_params, family_prediction = score, params, prediction
                record(city, name, family, "validation", validation, column, family_prediction, task)
                fitted = Pipeline([
                    ("preprocessing", _preprocessor(development, features, scale)),
                    ("model", estimator_class(**family_params)),
                ])
                fitted.fit(development[features], development[column])
                record(city, name, family, "test", test, column, fitted.predict(test[features]), task)
                target_metadata["families"].append({
                    "model": family, "trials": trials,
                    "selected_params": family_params, "validation_score": family_score,
                })
                if _better(family_score, selected_score, task):
                    selected_score, selected_model = family_score, fitted
                    selected_name, selected_params = family, family_params

            slug = re.sub(r"[^A-Za-z0-9_-]+", "_", city).strip("_").lower() or "city"
            artifact = model_dir / f"{slug}__{name}.joblib"
            joblib.dump(selected_model, artifact)
            target_metadata.update({
                "selected_family": selected_name,
                "selected_params": selected_params,
                "selected_validation_score": selected_score,
                "saved_model": str(artifact.relative_to(destination)),
                "refit_on": "train_plus_validation",
            })
            city_metadata["targets"].append(target_metadata)
        metadata["cities"].append(city_metadata)

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    predictions = pd.concat(prediction_frames, ignore_index=True)[PREDICTION_COLUMNS]
    for column in ("date", "target_date"):
        predictions[column] = predictions[column].dt.strftime("%Y-%m-%d")
    metrics.to_csv(destination / "metrics.csv", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    (destination / "model_selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics
