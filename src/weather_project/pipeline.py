"""Preview configuration or explicitly collect and prepare local data."""

import json
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("plan", "collect", "prepare")
COLLECTION_KEYS = (
    "location", "model", "timezone", "start_date", "end_date",
    "precipitation_threshold_mm", "cloud_threshold_percent",
)


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _dictionary(daily, samples, features, units, config):
    """Describe saved sample columns without fitting a transformation."""
    import pandas as pd

    records = []
    units = {**units, "temp_range": "°C"}
    for column in dict.fromkeys([*daily.columns, *samples.columns]):
        role = "predictor" if column in features else "metadata"
        unit, description = units.get(column, "unitless"), column.replace("_", " ")
        data_type = "numeric"
        if column in units and column not in features:
            role = "raw weather"
        if column in features and "__" in column:
            position, field = column.split("__", 1)
            unit = units.get(field, "unitless")
            if position.startswith("recent_lag_"):
                context = f"target_date minus {int(position.removeprefix('recent_lag_'))} days"
            else:
                year, offset = position.removeprefix("history_y").split("_offset_")
                signed_offset = int(offset[1:]) * (-1 if offset.startswith("m") else 1)
                context = f"same-calendar anchor {year} years earlier, offset {signed_offset:+d} calendar days"
            description = f"{field} at {context}"
            if field == "weather_code":
                data_type = "categorical"
                description += "; WMO category, one-hot encoded later"
            elif field in {"calendar_sin", "calendar_cos"}:
                description += "; actual source month/day encoded on a fixed 366-day calendar"
        elif column in {"target_calendar_sin", "target_calendar_cos"}:
            description = "Known target month/day encoded on a fixed 366-day calendar; model input"
        elif column == "weather_code":
            role, data_type, description = "raw categorical", "categorical", "WMO weather code; copied as a category at each historical input position"
        elif column == "weather_type":
            unit, data_type = "category", "categorical"
            description = "Derived Clear, Cloudy, or Precipitation label for EDA only; thresholds are recorded in config.json"
        elif column == "target_mean_temperature":
            role = "target"
            unit, description = "°C", "Target-day actual mean temperature; never a model input"
        elif column in {"date", "target_date"}:
            unit, data_type = "ISO date", "date"
            description = "Daily observation date; last input day in samples (target minus one day)" if column == "date" else "Prediction target day; API daily dates use " + config["timezone"]
        elif column in {"city", "split"}:
            unit, data_type = "category", "categorical"
            description = "Configured location" if column == "city" else "Train, validation, or test, assigned by target_date"
        elif column in {"history_days", "recent_days"}:
            unit = "days"
            description = "Prior-year date count" if column == "history_days" else "Recent date count"
        records.append({"column": column, "role": role, "data_type": data_type, "unit": unit, "description": description})
    return pd.DataFrame(records)


def run_pipeline(stage="plan", config_path=None, project_root=None):
    """No stage collects, prepares, trains, or analyzes data implicitly."""
    if stage not in STAGES:
        raise ValueError(f"Choose one of {STAGES}; model fitting is intentionally outside this runner")
    root = Path(project_root or PROJECT_ROOT).resolve()
    config = json.loads(Path(config_path or root / "config.json").read_text())
    boundaries = [date.fromisoformat(config[key]) for key in ("start_date", "train_end", "validation_end", "end_date")]
    if not boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3]:
        raise ValueError("Require start_date < train_end < validation_end < end_date")
    ZoneInfo(config["timezone"])
    for key in ("recent_days", "history_years", "seasonal_window_days"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if stage == "plan":
        return {
            "stage": "plan", "location": config["location"],
            "period": [config["start_date"], config["end_date"]],
            "timezone": config["timezone"],
            "features": {key: config[key] for key in ("recent_days", "history_years", "seasonal_window_days")},
            "raw_feature_count": (config["recent_days"] + config["history_years"] * (2 * config["seasonal_window_days"] + 1)) * (15 + 2) + 2,
            "target": "target_mean_temperature",
            "splits": {key: config[key] for key in ("train_end", "validation_end")},
            "interfaces": {"data": "collect, then prepare", "models": "build_models returns unfitted pipelines", "analysis": "explicit summary and plotting helpers"},
            "note": "Preview only. No API request, processing, model fitting, or analysis has run.",
        }

    folder = root / "data/processed"
    if stage == "collect":
        from .data import collect_data

        daily, quality, units = collect_data(config, root / "data/raw")
        folder.mkdir(parents=True, exist_ok=True)
        daily.to_csv(folder / "daily.csv", index=False)
        _save_json(folder / "collection_metadata.json", {
            "collection_config": {key: config[key] for key in COLLECTION_KEYS},
            "quality": quality, "units": units,
        })
        return {"stage": stage, "rows": len(daily), "output": str(folder / "daily.csv")}

    import pandas as pd
    from .features import build_samples

    metadata_path = folder / "collection_metadata.json"
    if not metadata_path.is_file() or not (folder / "daily.csv").is_file():
        raise FileNotFoundError("Run stage='collect' first; preparation never downloads data")
    metadata = json.loads(metadata_path.read_text())
    if metadata["collection_config"] != {key: config[key] for key in COLLECTION_KEYS}:
        raise ValueError("Collection settings changed; explicitly run stage='collect' again")
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"])
    samples, features = build_samples(daily, **{
        key: config[key] for key in ("recent_days", "history_years", "seasonal_window_days")
    })
    if samples.empty:
        raise ValueError("No labeled samples have complete recent and prior-year windows; check the collection period")
    samples["split"] = "test"
    samples.loc[samples.target_date.le(config["validation_end"]), "split"] = "validation"
    samples.loc[samples.target_date.le(config["train_end"]), "split"] = "train"
    samples.to_csv(folder / "samples.csv", index=False)
    _save_json(folder / "feature_columns.json", features)
    _dictionary(daily, samples, features, metadata["units"], config).to_csv(folder / "data_dictionary.csv", index=False)
    _save_json(folder / "preparation_metadata.json", {
        "config": config, "rows": len(samples), "feature_count": len(features),
        "target": "target_mean_temperature",
        "splits": samples.groupby("split").size().to_dict(),
        "evaluation": "Rolling one-day retrospective experiment; only earlier observations enter each row. Model fitting and tuning remain separate.",
    })
    return {"stage": stage, "rows": len(samples), "feature_count": len(features), "output": str(folder / "samples.csv")}
