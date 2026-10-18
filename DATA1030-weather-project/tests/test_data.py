"""Focused checks for temporal alignment and weather-label semantics."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_project.data import DAILY_VARIABLES, build_supervised, weather_types


def daily_fixture():
    frames = []
    dates = pd.date_range("2019-12-24", periods=12)
    for city, offset in [("A", 0.0), ("B", 1000.0)]:
        frame = pd.DataFrame({name: np.arange(12, dtype=float) for name in DAILY_VARIABLES})
        frame["city"] = city
        frame["date"] = dates
        frame["temperature_2m_mean"] = np.arange(12, dtype=float) + offset
        frame["temperature_2m_max"] = frame["temperature_2m_mean"] + 3
        frame["temperature_2m_min"] = frame["temperature_2m_mean"] - 2
        frame["temp_range"] = frame["temperature_2m_max"] - frame["temperature_2m_min"]
        frame["precipitation_sum"] = np.arange(12) % 3
        frame["cloud_cover_mean"] = 50.0
        frame["weather_type"] = weather_types(frame)
        frames.append(frame)
    # Reversed, interleaved input must not alter within-city temporal alignment.
    return pd.concat(frames, ignore_index=True).sort_values("date", ascending=False)


class WeatherLabelTests(unittest.TestCase):
    def test_precipitation_priority_thresholds_and_unknown_inputs(self):
        frame = pd.DataFrame({
            "precipitation_sum": [1.0, 3.0, 0.9, 0.0, np.nan, 0.0, 2.0],
            "cloud_cover_mean": [0.0, 99.0, 40.0, 39.9, 0.0, np.nan, np.nan],
        }, index=[10, 20, 30, 40, 50, 60, 70])
        labels = weather_types(frame)
        self.assertEqual(list(labels.loc[[10, 20, 30, 40, 70]]),
                         ["Precipitation", "Precipitation", "Cloudy", "Clear", "Precipitation"])
        self.assertTrue(pd.isna(labels.loc[50]))
        self.assertTrue(pd.isna(labels.loc[60]))
        self.assertTrue(labels.index.equals(frame.index))


class TemporalFeatureTests(unittest.TestCase):
    def test_missing_baseline_rows_are_excluded_but_other_missing_features_remain(self):
        daily = daily_fixture()
        baseline_missing = daily.city.eq("A") & daily.date.eq(pd.Timestamp("2019-12-31"))
        predictor_missing = daily.city.eq("A") & daily.date.eq(pd.Timestamp("2020-01-01"))
        daily.loc[baseline_missing, "temperature_2m_mean"] = np.nan
        daily.loc[predictor_missing, "relative_humidity_2m_mean"] = np.nan
        supervised, _ = build_supervised(daily)
        self.assertFalse((supervised.city.eq("A") & supervised.date.eq(pd.Timestamp("2019-12-31"))).any())
        retained = supervised.loc[supervised.city.eq("A") & supervised.date.eq(pd.Timestamp("2020-01-01"))]
        self.assertEqual(len(retained), 1)
        self.assertTrue(pd.isna(retained.iloc[0].relative_humidity_2m_mean))
        self.assertFalse(supervised[["persistence_temperature", "persistence_range", "persistence_weather"]].isna().any().any())

    def test_targets_stay_within_city_and_unlabeled_final_day_is_removed(self):
        daily = daily_fixture()
        supervised, _ = build_supervised(daily)
        self.assertEqual(supervised.groupby("city").size().to_dict(), {"A": 4, "B": 4})
        lookup = daily.set_index(["city", "date"])
        for row in supervised.itertuples():
            tomorrow = lookup.loc[(row.city, row.date + pd.Timedelta(days=1))]
            self.assertEqual(row.target_date, row.date + pd.Timedelta(days=1))
            self.assertEqual(row.target_mean_temperature, tomorrow["temperature_2m_mean"])
            self.assertEqual(row.target_temperature_range, tomorrow["temp_range"])
            self.assertEqual(row.target_weather_type, tomorrow["weather_type"])
        for city, group in daily.groupby("city"):
            selected = supervised.loc[supervised.city.eq(city)]
            self.assertEqual(selected.target_date.max(), group.date.max())
            self.assertLess(selected.date.max(), group.date.max())
        first_b = supervised.loc[supervised.city.eq("B")].iloc[0]
        self.assertEqual(first_b.temperature_2m_mean_lag_7, 1000.0)
        self.assertEqual(first_b.temperature_2m_mean_lag_1, 1006.0)
        self.assertAlmostEqual(first_b.temperature_rolling_7, 1004.0)

    def test_changing_future_observations_cannot_change_current_features(self):
        daily = daily_fixture()
        original, features = build_supervised(daily)
        current = original.loc[original.city.eq("A")].iloc[0]
        changed_daily = daily.copy()
        future = changed_daily.city.eq("A") & changed_daily.date.gt(current.date)
        changed_daily.loc[future, DAILY_VARIABLES] += 5000
        changed_daily.loc[future, "temp_range"] += 50
        changed_daily.loc[future, "weather_type"] = "Clear"
        changed, changed_features = build_supervised(changed_daily)
        after = changed.loc[changed.city.eq("A") & changed.date.eq(current.date)].iloc[0]
        self.assertEqual(features, changed_features)
        np.testing.assert_allclose(current[features].to_numpy(dtype=float),
                                   after[features].to_numpy(dtype=float), equal_nan=True)
        self.assertNotEqual(current.target_mean_temperature, after.target_mean_temperature)

    def test_gaps_cannot_silently_make_next_record_mean_next_day(self):
        daily = daily_fixture()
        gap = daily.city.eq("A") & daily.date.eq(pd.Timestamp("2019-12-27"))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            build_supervised(daily.loc[~gap])


if __name__ == "__main__":
    unittest.main()
