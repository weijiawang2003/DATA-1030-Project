"""Explicit stage orchestration. The default only describes the pipeline."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("plan", "collect", "prepare", "train", "plot", "all")


def _write_json(path, values):
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n")


def _context(config_path, mode, cities, project_root):
    root = Path(project_root or PROJECT_ROOT).resolve()
    config = json.loads(Path(config_path or root / "config.json").read_text())
    if mode not in config["modes"]:
        raise ValueError(f"Unknown mode: {mode}")
    period = config["modes"][mode]
    dates = [date.fromisoformat(period[key]) for key in ("start", "train_end", "validation_end", "end")]
    if not dates[0] < dates[1] < dates[2] < dates[3]:
        raise ValueError("Require start < train_end < validation_end < end")
    if config["timezone"] != "UTC":
        raise ValueError("This starter uses reproducible UTC daily boundaries; timezone must be UTC")
    configured = config["cities"]
    if not configured or len({city["id"] for city in configured}) != len(configured):
        raise ValueError("Configure at least one city, with unique city IDs")
    selected = configured
    if cities:
        unknown = set(cities) - {city["id"] for city in configured}
        if unknown:
            raise ValueError(f"Unknown city IDs: {sorted(unknown)}")
        selected = [city for city in configured if city["id"] in cities]
    run_id = mode if len(selected) == len(configured) else mode + "_" + "-".join(city["id"] for city in selected)
    return {
        "root": root, "config": config, "period": period, "cities": selected,
        "mode": mode, "run_id": run_id,
        "raw_dir": root / "data/raw", "processed_dir": root / "data/processed" / run_id,
        "result_dir": root / "results" / run_id, "figure_dir": root / "figures" / run_id,
    }


def _plan(context, include_svm, include_range):
    run_id = context["run_id"]
    return {
        "stage": "plan", "run_id": run_id,
        "cities": [city["name"] for city in context["cities"]],
        "period": context["period"],
        "options": {"include_svm": include_svm, "include_range": include_range},
        "note": "Configuration preview only; no data collection, processing, training or plotting has run.",
        "stages": [
            {"stage": "collect", "input": "config.json and public API", "output": f"data/raw/ and data/processed/{run_id}/daily.csv"},
            {"stage": "prepare", "input": f"data/processed/{run_id}/daily.csv", "output": "supervised.csv, data dictionary and split definitions"},
            {"stage": "train", "input": "prepared samples and feature definitions", "output": f"results/{run_id}/"},
            {"stage": "plot", "input": "saved data and model outputs", "output": f"figures/{run_id}/"},
        ],
    }


def _prior_metadata(path, context, prerequisite):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}; explicitly run stage='{prerequisite}' first")
    metadata = json.loads(path.read_text())
    if metadata["config"] != context["config"] or metadata["selected_cities"] != [city["id"] for city in context["cities"]]:
        raise ValueError(f"Configuration differs from saved artifacts; explicitly rerun stage='{prerequisite}'")
    return metadata


def _metadata(context):
    return {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": context["run_id"], "mode": context["mode"],
        "config": context["config"], "period": context["period"],
        "selected_cities": [city["id"] for city in context["cities"]],
    }


def _collect(context, offline, refresh):
    import pandas as pd
    from .data import collect_city

    frames, checks, units = [], [], None
    for city in context["cities"]:
        daily, quality, current_units = collect_city(
            city, context["period"], context["config"], context["raw_dir"], offline, refresh,
        )
        if units is not None and units != current_units:
            raise ValueError("API units differ across cities")
        frames.append(daily)
        checks.append(quality)
        units = current_units
    folder = context["processed_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(folder / "daily.csv", index=False)
    pd.DataFrame(checks).to_csv(folder / "data_quality.csv", index=False)
    metadata = {**_metadata(context), "stage": "collect", "api_units": units, "daily_rows": len(daily)}
    _write_json(folder / "collection_metadata.json", metadata)
    return {"stage": "collect", "rows": len(daily), "output": f"data/processed/{context['run_id']}/daily.csv"}


def _prepare(context):
    import pandas as pd
    from .data import build_supervised, data_dictionary

    folder = context["processed_dir"]
    collected = _prior_metadata(folder / "collection_metadata.json", context, "collect")
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"])
    supervised, features = build_supervised(daily)
    period = context["period"]
    supervised["split"] = "test"
    supervised.loc[supervised["target_date"].le(period["validation_end"]), "split"] = "validation"
    supervised.loc[supervised["target_date"].le(period["train_end"]), "split"] = "train"
    supervised.to_csv(folder / "supervised.csv", index=False)
    dictionary = data_dictionary(daily, supervised, features, collected["api_units"], context["config"])
    dictionary.to_csv(folder / "data_dictionary.csv", index=False)
    split_summary = supervised.groupby(["city", "split"])["target_date"].agg(
        rows="size", target_start="min", target_end="max",
    ).reset_index()
    split_summary.to_csv(folder / "splits.csv", index=False)
    metadata = {
        **_metadata(context), "stage": "prepare", "daily_rows": len(daily),
        "supervised_rows": len(supervised), "feature_columns": features,
    }
    _write_json(folder / "preparation_metadata.json", metadata)
    return {"stage": "prepare", "rows": len(supervised), "features": features}


def _train(context, include_svm, include_range):
    import pandas as pd
    from .modeling import run_models

    folder = context["processed_dir"]
    prepared = _prior_metadata(folder / "preparation_metadata.json", context, "prepare")
    supervised = pd.read_csv(folder / "supervised.csv", parse_dates=["date", "target_date"])
    targets = ("temperature", "weather", "range") if include_range else ("temperature", "weather")
    run_models(
        supervised, prepared["feature_columns"], context["result_dir"],
        train_end=context["period"]["train_end"], validation_end=context["period"]["validation_end"],
        include_svm=include_svm, targets=targets, seed=context["config"]["seed"],
    )
    metadata = {
        **_metadata(context), "stage": "train", "include_svm": include_svm, "include_range": include_range,
        "targets": list(targets), "feature_columns": prepared["feature_columns"],
        "daily_rows": prepared["daily_rows"], "supervised_rows": prepared["supervised_rows"],
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "requests", "joblib"]},
        "experiment": "Retrospective ERA5 t to t+1 experiment; finalized inputs are not live observations",
        "selection": "Hyperparameters and learned model family selected on validation only",
    }
    _write_json(context["result_dir"] / "run_metadata.json", metadata)
    return {"stage": "train", "output": f"results/{context['run_id']}/"}


def _plot(context):
    import pandas as pd
    from .plots import plot_eda, plot_model_results

    _prior_metadata(context["processed_dir"] / "collection_metadata.json", context, "collect")
    _prior_metadata(context["result_dir"] / "run_metadata.json", context, "train")
    daily = pd.read_csv(context["processed_dir"] / "daily.csv", parse_dates=["date"])
    metrics = pd.read_csv(context["result_dir"] / "metrics.csv")
    predictions = pd.read_csv(context["result_dir"] / "predictions.csv")
    plot_eda(daily, context["figure_dir"])
    plot_model_results(metrics, predictions, context["figure_dir"])
    return {"stage": "plot", "output": f"figures/{context['run_id']}/"}


def run_pipeline(config_path=None, mode="full", cities=None, include_svm=False,
                 include_range=False, offline=False, refresh=False,
                 project_root=None, stage="plan"):
    """Preview by default; execute only an explicitly requested stage.

    collect -> prepare -> train -> plot are separate manual entry points.
    Missing prerequisites raise errors instead of launching earlier stages.
    stage='all' explicitly opts into every stage for future use.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose {STAGES}")
    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    context = _context(config_path, mode, cities, project_root)
    plan = _plan(context, include_svm, include_range)
    if stage == "plan":
        return plan
    requested = ("collect", "prepare", "train", "plot") if stage == "all" else (stage,)
    completed = []
    for current in requested:
        if current == "collect":
            completed.append(_collect(context, offline, refresh))
        elif current == "prepare":
            completed.append(_prepare(context))
        elif current == "train":
            completed.append(_train(context, include_svm, include_range))
        else:
            completed.append(_plot(context))
    return {"stage": stage, "run_id": context["run_id"], "completed": completed}
