"""Reproducible exploratory figures for the daily weather data."""

from __future__ import annotations

import calendar
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_WEATHER_ORDER = ["Clear", "Cloudy", "Precipitation"]
_WEATHER_COLORS = ["#EDBD52", "#8A9DAD", "#3877AE"]
_SOURCE = "Source: Open-Meteo / ERA5; days use UTC. Exploratory summaries; no model results shown."
_MODEL_NAMES = {
    "persistence": "Persistence",
    "climatology": "Climatology",
    "ridge": "Ridge",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "svm": "SVM (RBF)",
}
_MODEL_ORDER = list(_MODEL_NAMES)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    # Initialize tick-label layout before the tight-bounding-box save pass.
    fig.canvas.draw()
    fig.savefig(output_dir / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _next_day_pairs(daily: pd.DataFrame) -> pd.DataFrame:
    """Pair temperatures only when the following record is the next calendar day."""
    ordered = daily.sort_values(["city", "date"]).copy()
    grouped = ordered.groupby("city", sort=False)
    ordered["next_date"] = grouped["date"].shift(-1)
    ordered["next_temperature"] = grouped["temperature_2m_mean"].shift(-1)
    contiguous = (ordered["next_date"] - ordered["date"]).eq(pd.Timedelta(days=1))
    paired = ordered.loc[
        contiguous,
        ["city", "date", "temperature_2m_mean", "next_temperature"],
    ].dropna(subset=["temperature_2m_mean", "next_temperature"])
    paired = paired.loc[
        np.isfinite(paired["temperature_2m_mean"])
        & np.isfinite(paired["next_temperature"])
    ].copy()
    paired["absolute_change"] = (
        paired["next_temperature"] - paired["temperature_2m_mean"]
    ).abs()
    return paired


def plot_eda(daily: pd.DataFrame, output_dir: Path) -> None:
    """Save four data-derived EDA PNGs without requiring an interactive display.

    Monthly summaries pool the same calendar month across the available years.
    Daily-change summaries assign each next-day pair to its first day's month.
    """
    required = {"city", "date", "temperature_2m_mean", "weather_type"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"EDA input is missing columns: {sorted(missing)}")
    if daily.empty:
        raise ValueError("EDA input contains no daily records.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    cities = list(daily["city"].dropna().drop_duplicates())
    if not cities:
        raise ValueError("EDA input contains no named cities.")
    months = np.arange(1, 13)
    month_labels = [calendar.month_abbr[month] for month in months]
    pairs = _next_day_pairs(daily)

    with plt.rc_context(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.titlesize": 17,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
        }
    ):
        monthly = (
            daily.assign(month=daily["date"].dt.month)
            .groupby(["city", "month"])["temperature_2m_mean"]
            .mean()
            .unstack("month")
            .reindex(index=cities, columns=months)
        )
        fig, ax = plt.subplots(figsize=(14, 6.5))
        colors = plt.get_cmap("tab10").colors
        for index, city in enumerate(cities):
            ax.plot(
                months,
                monthly.loc[city],
                marker="o",
                markersize=4.5,
                linewidth=2,
                color=colors[index % len(colors)],
                label=city,
            )
        ax.set(
            title="Monthly mean temperature by city",
            xlabel="Calendar month (all available years pooled)",
            ylabel="Mean daily temperature (°C)",
            xticks=months,
            xticklabels=month_labels,
            xlim=(0.7, 12.3),
        )
        ax.grid(axis="y")
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=min(4, len(cities)), frameon=False,
        )
        fig.subplots_adjust(bottom=0.27, top=0.9)
        fig.text(0.02, 0.02, _SOURCE, fontsize=9, color="#555555")
        _save(fig, output_dir, "01_monthly_mean_temperature.png")

        ncols = min(4, len(cities))
        nrows = int(np.ceil(len(cities) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(15, 4.1 * nrows + 0.6),
            sharex=True, sharey=True, squeeze=False,
        )
        if pairs.empty:
            lower, upper = 0.0, 1.0
        else:
            all_temperatures = pairs[["temperature_2m_mean", "next_temperature"]].to_numpy()
            lower, upper = float(np.min(all_temperatures)), float(np.max(all_temperatures))
            pad = max((upper - lower) * 0.04, 1.0)
            lower, upper = lower - pad, upper + pad
        for index, city in enumerate(cities):
            ax = axes.flat[index]
            subset = pairs.loc[pairs["city"].eq(city)]
            original_count = len(subset)
            if original_count > 1_500:
                sample_indices = np.linspace(0, original_count - 1, 1_500, dtype=int)
                subset = subset.iloc[sample_indices]
            ax.scatter(
                subset["temperature_2m_mean"], subset["next_temperature"],
                s=8, alpha=0.25, color=colors[index % len(colors)],
                linewidths=0, rasterized=True,
            )
            ax.plot([lower, upper], [lower, upper], "--", color="#333333", linewidth=1)
            ax.set(
                title=city, xlim=(lower, upper), ylim=(lower, upper),
                aspect="equal",
            )
            count_label = f"{original_count:,} consecutive-day pairs"
            if original_count > len(subset):
                count_label += f"\n{len(subset):,} displayed"
            ax.text(
                0.04, 0.96, count_label, transform=ax.transAxes,
                va="top", fontsize=8.5,
            )
            ax.grid()
        for index in range(len(cities), nrows * ncols):
            axes.flat[index].set_visible(False)
        fig.suptitle("Today's and next day's mean temperature", y=0.98)
        fig.supxlabel("Today's mean temperature (°C)", y=0.065)
        fig.supylabel("Next day's mean temperature (°C)", x=0.025)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.14, top=0.91, hspace=0.25, wspace=0.15)
        fig.text(
            0.02, 0.015, _SOURCE + " Dashed line: next-day temperature equals today's.",
            fontsize=8.5, color="#555555",
        )
        _save(fig, output_dir, "02_today_vs_next_day_temperature.png")

        changes = (
            pairs.assign(month=pairs["date"].dt.month)
            .groupby(["city", "month"])["absolute_change"]
            .mean()
            .unstack("month")
            .reindex(index=cities, columns=months)
        )
        matrix = changes.to_numpy(dtype=float)
        finite = matrix[np.isfinite(matrix)]
        vmax = max(float(finite.max()), 0.1) if finite.size else 1.0
        fig, ax = plt.subplots(figsize=(14, max(5.2, 0.52 * len(cities) + 2)))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#E8E8E8")
        heat = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set(
            title="Mean absolute day-to-day temperature change",
            xlabel="Month of the first day in each consecutive-day pair",
            xticks=np.arange(12), xticklabels=month_labels,
            yticks=np.arange(len(cities)), yticklabels=cities,
        )
        for row in range(len(cities)):
            for column in range(12):
                value = matrix[row, column]
                label = f"{value:.2f}" if np.isfinite(value) else "—"
                text_color = "white" if np.isfinite(value) and value / vmax < 0.55 else "#151515"
                ax.text(column, row, label, ha="center", va="center", color=text_color, fontsize=10)
        fig.colorbar(heat, ax=ax, label="Mean absolute change (°C)", fraction=0.035, pad=0.025)
        fig.subplots_adjust(left=0.14, right=0.94, bottom=0.2, top=0.86)
        fig.text(0.02, 0.025, _SOURCE, fontsize=9, color="#555555")
        _save(fig, output_dir, "03_day_to_day_temperature_change.png")

        counts = (
            pd.crosstab(daily["city"], daily["weather_type"])
            .reindex(index=cities, columns=_WEATHER_ORDER, fill_value=0)
        )
        labeled_days = counts.sum(axis=1)
        proportions = counts.div(labeled_days.replace(0, np.nan), axis=0).fillna(0)
        fig, ax = plt.subplots(figsize=(14, 6.6))
        x = np.arange(len(cities))
        bottom = np.zeros(len(cities))
        for label, color in zip(_WEATHER_ORDER, _WEATHER_COLORS):
            values = proportions[label].to_numpy()
            ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.66)
            for index, value in enumerate(values):
                if value >= 0.04:
                    ax.text(
                        index, bottom[index] + value / 2, f"{value:.1%}",
                        ha="center", va="center", fontsize=10,
                        color="white" if label == "Precipitation" else "#17212B",
                    )
            bottom += values
        ax.set(
            title="Weather-type proportions by city",
            ylabel="Share of days with a valid weather label",
            xticks=x, xticklabels=cities, ylim=(0, 1.04),
        )
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="y")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        fig.subplots_adjust(bottom=0.23, top=0.9)
        fig.text(
            0.02, 0.025,
            _SOURCE + " Missing weather labels are excluded from proportions.",
            fontsize=8.5, color="#555555",
        )
        _save(fig, output_dir, "04_weather_type_proportions.png")


def plot_model_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot descriptive held-out test comparisons without selecting a winner.

    Regression figures show MAE (°C); the weather figure shows macro-F1.
    Weather confusion matrices pool test predictions across cities and display
    both counts and within-actual-class percentages.
    """
    required = {"city", "target", "model", "split", "mae", "macro_f1"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Model metrics are missing columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test = metrics.loc[metrics["split"].eq("test")].copy()
    if test.empty:
        return
    cities = list(test["city"].drop_duplicates())
    result_source = (
        "Open-Meteo / ERA5; target dates use UTC. "
        "Test comparisons are descriptive; use validation results for model selection."
    )
    target_names = {
        "temperature": "Next-day mean temperature",
        "range": "Next-day diurnal temperature range",
        "weather": "Next-day weather type",
    }
    with plt.rc_context({"font.size": 11, "axes.titlesize": 15}):
        for target in test["target"].drop_duplicates():
            subset = test.loc[test["target"].eq(target)]
            available = list(subset["model"].drop_duplicates())
            models = [model for model in _MODEL_ORDER if model in available]
            models.extend(model for model in available if model not in models)
            is_classification = target == "weather"
            metric = "macro_f1" if is_classification else "mae"
            values = subset.pivot(index="city", columns="model", values=metric).reindex(
                index=cities, columns=models,
            )
            matrix = values.to_numpy(dtype=float)
            finite = matrix[np.isfinite(matrix)]
            vmax = 1.0 if is_classification else max(float(finite.max()), 0.1) if finite.size else 1.0
            cmap = plt.get_cmap("viridis" if is_classification else "viridis_r").copy()
            cmap.set_bad("#E8E8E8")
            fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.8 + 3), max(5.5, len(cities) * 0.55 + 2)))
            heat = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
            metric_label = "Macro-F1 (higher is better)" if is_classification else "MAE (°C; lower is better)"
            ax.set(
                title=f"Exploratory test comparison: {target_names.get(target, target)}",
                xlabel="Model (no test-based model selection)",
                xticks=np.arange(len(models)),
                xticklabels=[_MODEL_NAMES.get(model, model).replace(" ", "\n", 1) for model in models],
            )
            for row in range(len(cities)):
                for column in range(len(models)):
                    value = matrix[row, column]
                    label = f"{value:.3f}" if is_classification else f"{value:.2f}"
                    if not np.isfinite(value):
                        label = "—"
                    normalized = value / vmax if np.isfinite(value) else 1.0
                    low_luminance = normalized < 0.55 if is_classification else normalized > 0.45
                    color = "white" if np.isfinite(value) and low_luminance else "#151515"
                    ax.text(column, row, label, ha="center", va="center", color=color, fontsize=11)
            fig.colorbar(heat, ax=ax, label=metric_label, fraction=0.035, pad=0.025)
            # Set labels explicitly after the colorbar modifies the axes layout.
            ax.set_yticks(np.arange(len(cities)), labels=list(cities))
            ax.tick_params(axis="y", labelleft=True, labelcolor="black")
            for tick_label in ax.get_yticklabels():
                tick_label.set_clip_on(False)
            fig.subplots_adjust(left=0.15, right=0.94, bottom=0.2, top=0.86)
            fig.text(0.02, 0.025, result_source, fontsize=9, color="#555555")
            _save(fig, output_dir, f"05_test_{target}_{metric}.png")

        prediction_required = {"target", "split", "model", "y_true", "y_pred"}
        if predictions.empty:
            return
        prediction_missing = prediction_required.difference(predictions.columns)
        if prediction_missing:
            raise ValueError(f"Model predictions are missing columns: {sorted(prediction_missing)}")
        weather = predictions.loc[
            predictions["target"].eq("weather") & predictions["split"].eq("test")
        ]
        available = list(weather["model"].drop_duplicates())
        models = [model for model in _MODEL_ORDER if model in available]
        models.extend(model for model in available if model not in models)
        if not models:
            return
        ncols = min(3, len(models))
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(15, 4 * nrows + 1.2), squeeze=False,
        )
        for index, model in enumerate(models):
            ax = axes.flat[index]
            subset = weather.loc[weather["model"].eq(model)]
            confusion = pd.crosstab(subset["y_true"], subset["y_pred"]).reindex(
                index=_WEATHER_ORDER, columns=_WEATHER_ORDER, fill_value=0,
            )
            counts = confusion.to_numpy(dtype=int)
            row_totals = counts.sum(axis=1, keepdims=True)
            shares = np.divide(
                counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals != 0,
            )
            heat = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
            ax.set(
                title=_MODEL_NAMES.get(model, model),
                xticks=np.arange(3), xticklabels=["Clear", "Cloudy", "Precip."],
                yticks=np.arange(3), yticklabels=_WEATHER_ORDER,
            )
            for row in range(3):
                for column in range(3):
                    percentage = f"{shares[row, column]:.1%}" if row_totals[row, 0] else "n/a"
                    ax.text(
                        column, row, f"{counts[row, column]:,}\n({percentage})",
                        ha="center", va="center", fontsize=10,
                        color="white" if shares[row, column] > 0.55 else "#17212B",
                    )
        for index in range(len(models), ncols * nrows):
            axes.flat[index].set_visible(False)
        fig.suptitle("Exploratory test comparison: weather confusion matrices pooled across cities", y=0.98, fontsize=16)
        fig.supxlabel("Predicted weather type", y=0.065)
        fig.supylabel("Actual weather type", x=0.015)
        fig.subplots_adjust(left=0.09, right=0.91, bottom=0.14, top=0.9, hspace=0.32, wspace=0.38)
        color_axis = fig.add_axes([0.94, 0.22, 0.012, 0.57])
        colorbar = fig.colorbar(heat, cax=color_axis, label="Share of actual class")
        colorbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        fig.text(0.02, 0.015, result_source, fontsize=9, color="#555555")
        _save(fig, output_dir, "06_test_weather_confusion_matrices.png")
