"""Small analysis helpers with no automatic plotting, fitting, or file writes."""

import pandas as pd


def data_overview(daily):
    """Summarize types, missing values, unique values, and numeric ranges."""
    overview = pd.DataFrame({
        "dtype": daily.dtypes.astype(str),
        "missing": daily.isna().sum(),
        "unique": daily.nunique(dropna=True),
    })
    numeric = daily.select_dtypes(include="number").drop(columns="weather_code", errors="ignore")
    if not numeric.empty:
        overview = overview.join(numeric.agg(["min", "mean", "max"]).T)
    return overview.rename_axis("column").reset_index()


def plot_temperature(daily):
    """Return a daily-temperature figure without saving or showing it."""
    import matplotlib.pyplot as plt

    if not {"date", "temperature_2m_mean"}.issubset(daily.columns):
        raise ValueError("daily must contain date and temperature_2m_mean")
    ordered = daily.sort_values("date")
    figure, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(pd.to_datetime(ordered["date"]), ordered["temperature_2m_mean"], linewidth=0.7)
    ax.set(title="Brown University daily mean temperature", xlabel="Date", ylabel="Temperature (°C)")
    figure.tight_layout()
    return figure


def plot_scores(scores, metric="mae"):
    """Return a comparison figure from a table with model and metric columns."""
    import matplotlib.pyplot as plt

    if not {"model", metric}.issubset(scores.columns):
        raise ValueError(f"scores must contain model and {metric}")
    figure, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(scores["model"].astype(str), scores[metric])
    ax.set(title=f"Model comparison: {metric}", xlabel="Model", ylabel=metric)
    ax.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    return figure
