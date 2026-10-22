"""Keep recent and prior-year daily observations as separate raw predictors."""

from __future__ import annotations

from calendar import monthrange
from numbers import Integral

import numpy as np
import pandas as pd

from .data import DAILY_VARIABLES


POSITION_FIELDS = (*DAILY_VARIABLES, "calendar_sin", "calendar_cos")
METADATA_COLUMNS = ("city", "date", "target_date", "recent_days", "history_days")
TARGET_COLUMN = "target_mean_temperature"


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")


def _offset_name(offset):
    if offset < 0:
        return f"m{-offset}"
    return "z0" if offset == 0 else f"p{offset}"


def _history_dates(target, years, window):
    """Order by previous year 1..years, then calendar offsets -window..window."""
    dates = []
    for years_back in range(1, years + 1):
        year = target.year - years_back
        day = min(target.day, monthrange(year, target.month)[1])
        anchor = pd.Timestamp(year=year, month=target.month, day=day)
        dates.extend(pd.date_range(
            anchor - pd.Timedelta(days=window), periods=2 * window + 1,
        ))
    return pd.DatetimeIndex(dates)


def _calendar_phase(dates):
    """Use the same month/day position in leap year 2000 for every source year."""
    reference = pd.to_datetime("2000-" + dates.dt.strftime("%m-%d"))
    return 2 * np.pi * (reference.dt.dayofyear - 1) / 366


def build_samples(daily, recent_days=28, history_years=5, seasonal_window_days=3):
    """Return raw-position X columns and the target day's mean temperature y.

    For target date d, recent lag 01 is d-1 and lag 28 is d-28. Each prior
    year contributes an inclusive +/- calendar window around the corresponding
    month/day. February 29 anchors become February 28 in non-leap years;
    windows may cross calendar-year boundaries. All context dates must exist
    within the same city and be strictly earlier than d. Incomplete windows
    skip that target, while missing predictor measurements remain missing.

    Each context position contains the 15 API fields and its own calendar
    sine/cosine. Weather codes are categorical strings, with np.nan for unknown
    codes. Other fields are numeric. Two additional predictors describe d's
    known calendar date. Defaults give (28 + 5*7)*17 + 2 = 1,073 predictors.
    No means, standard deviations, probabilities, imputation or fitting occur.

    Metadata date is d-1; this is an input-day boundary, not a statement that
    finalized reanalysis was available live then. Rows without a finite target
    mean temperature are omitted. Empty results retain the complete schema.
    """
    _positive_integer(recent_days, "recent_days")
    _positive_integer(history_years, "history_years")
    _positive_integer(seasonal_window_days, "seasonal_window_days", minimum=0)
    if not daily.columns.is_unique:
        raise ValueError("Daily data contains duplicate column names.")
    required = {"city", "date", *DAILY_VARIABLES}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing daily columns: {sorted(missing)}.")
    if daily["city"].isna().any():
        raise ValueError("Each daily row must have a city.")

    frame = daily.copy()
    try:
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        frame["date"] = dates.dt.normalize()
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("date must contain consistent calendar dates.") from exc
    if frame["date"].isna().any():
        raise ValueError("Daily data contains missing dates.")
    if frame.duplicated(["city", "date"]).any():
        raise ValueError("Dates must be unique within each city.")
    for variable in DAILY_VARIABLES:
        numeric = pd.to_numeric(frame[variable], errors="raise").astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        if variable == "weather_code":
            if numeric.dropna().mod(1).ne(0).any():
                raise ValueError("weather_code must contain integer category codes or missing values.")
            frame[variable] = numeric.map(
                lambda value: str(int(value)) if pd.notna(value) else np.nan,
            ).astype(object)
        else:
            frame[variable] = numeric
    phase = _calendar_phase(frame["date"])
    frame["calendar_sin"], frame["calendar_cos"] = np.sin(phase), np.cos(phase)

    prefixes = [f"recent_lag_{lag:02d}" for lag in range(1, recent_days + 1)]
    prefixes += [
        f"history_y{year}_offset_{_offset_name(offset)}"
        for year in range(1, history_years + 1)
        for offset in range(-seasonal_window_days, seasonal_window_days + 1)
    ]
    features = [f"{prefix}__{field}" for prefix in prefixes for field in POSITION_FIELDS]
    features += ["target_calendar_sin", "target_calendar_cos"]
    columns = [*METADATA_COLUMNS, *features, TARGET_COLUMN]
    n_history = history_years * (2 * seasonal_window_days + 1)
    records = []

    for city, group in frame.groupby("city", sort=True):
        group = group.sort_values("date").set_index("date")
        index = group.index
        values = group[list(POSITION_FIELDS)].to_numpy(dtype=object)
        for position, target in enumerate(index):
            target_mean = values[position, POSITION_FIELDS.index("temperature_2m_mean")]
            if not np.isfinite(target_mean):
                continue
            # Skip the initial historical warmup before constructing every window.
            oldest_year = target.year - history_years
            oldest_day = min(target.day, monthrange(oldest_year, target.month)[1])
            oldest_anchor = pd.Timestamp(year=oldest_year, month=target.month, day=oldest_day)
            if oldest_anchor - pd.Timedelta(days=seasonal_window_days) < index[0]:
                continue
            recent_dates = pd.DatetimeIndex([
                target - pd.Timedelta(days=lag) for lag in range(1, recent_days + 1)
            ])
            history_dates = _history_dates(target, history_years, seasonal_window_days)
            if history_dates.max() >= target:
                continue
            recent_positions = index.get_indexer(recent_dates)
            history_positions = index.get_indexer(history_dates)
            if (recent_positions < 0).any() or (history_positions < 0).any():
                continue

            context_positions = np.concatenate([recent_positions, history_positions])
            raw_features = values[context_positions].reshape(-1)
            records.append([
                city, target - pd.Timedelta(days=1), target, recent_days, n_history,
                *raw_features, values[position, -2], values[position, -1], target_mean,
            ])

    samples = pd.DataFrame(records, columns=columns)
    for column in ("date", "target_date"):
        samples[column] = pd.to_datetime(samples[column])
    # Preserve categorical semantics even if a code column is entirely missing.
    for column in features:
        if column.endswith("__weather_code"):
            samples[column] = samples[column].astype(object)
    return samples, features
