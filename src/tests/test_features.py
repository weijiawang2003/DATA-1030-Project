"""Offline raw-window and temporal checks; no data requests or model fitting."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weather_project.data import DAILY_VARIABLES
from weather_project.features import POSITION_FIELDS, build_samples


def daily_fixture():
    dates = pd.date_range("1999-01-01", "2005-03-08")
    frame = pd.DataFrame({"city": "Brown", "date": dates})
    for number, variable in enumerate(DAILY_VARIABLES):
        frame[variable] = np.arange(len(dates), dtype=float) + 100 * number
    frame["weather_code"] = np.asarray([0, 1, 3, 61, 73])[np.arange(len(dates)) % 5]
    return frame


def row_for(samples, date, city="Brown"):
    return samples.loc[samples.city.eq(city) & samples.target_date.eq(pd.Timestamp(date))].iloc[0]


class RawFeatureWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = daily_fixture()
        cls.samples, cls.features = build_samples(cls.daily)
        cls.lookup = cls.daily.set_index("date")

    def test_all_28_recent_positions_retain_their_raw_values(self):
        row = row_for(self.samples, "2005-01-03")
        self.assertEqual(row.date, pd.Timestamp("2005-01-02"))
        for lag in range(1, 29):
            source_date = pd.Timestamp("2005-01-03") - pd.Timedelta(days=lag)
            for field in DAILY_VARIABLES:
                expected = self.lookup.loc[source_date, field]
                if field == "weather_code":
                    expected = str(int(expected))
                self.assertEqual(row[f"recent_lag_{lag:02d}__{field}"], expected)
        self.assertEqual(row["recent_lag_28__temperature_2m_mean"],
                         self.lookup.loc["2004-12-06", "temperature_2m_mean"])

    def test_new_year_history_offsets_and_position_calendar_are_exact(self):
        row = row_for(self.samples, "2005-01-03")
        self.assertEqual(row.recent_days, 28)
        self.assertEqual(row.history_days, 35)
        expected_dates = {
            "history_y1_offset_m3": "2003-12-31",
            "history_y1_offset_z0": "2004-01-03",
            "history_y1_offset_p3": "2004-01-06",
            "history_y5_offset_m3": "1999-12-31",
            "history_y5_offset_p3": "2000-01-06",
        }
        for prefix, source_date in expected_dates.items():
            self.assertEqual(row[f"{prefix}__temperature_2m_mean"],
                             self.lookup.loc[source_date, "temperature_2m_mean"])
        self.assertAlmostEqual(row["history_y1_offset_m3__calendar_sin"],
                               np.sin(2 * np.pi * 365 / 366))
        self.assertAlmostEqual(row["recent_lag_01__calendar_sin"], np.sin(2 * np.pi / 366))
        self.assertAlmostEqual(row.target_calendar_sin, np.sin(2 * np.pi * 2 / 366))

    def test_february_29_fallback_and_fixed_calendar_phase(self):
        row = row_for(self.samples, "2004-02-29")
        expected_dates = {
            "history_y1_offset_m3": "2003-02-25",
            "history_y1_offset_z0": "2003-02-28",
            "history_y1_offset_p3": "2003-03-03",
            "history_y4_offset_z0": "2000-02-29",
        }
        for prefix, source_date in expected_dates.items():
            self.assertEqual(row[f"{prefix}__temperature_2m_mean"],
                             self.lookup.loc[source_date, "temperature_2m_mean"])
        self.assertAlmostEqual(row["history_y1_offset_z0__calendar_sin"], np.sin(2 * np.pi * 58 / 366))
        self.assertAlmostEqual(row["history_y4_offset_z0__calendar_sin"], np.sin(2 * np.pi * 59 / 366))
        self.assertAlmostEqual(row.target_calendar_sin, np.sin(2 * np.pi * 59 / 366))
        leap = row_for(self.samples, "2004-03-01")
        regular = row_for(self.samples, "2005-03-01")
        self.assertEqual(leap.target_calendar_sin, regular.target_calendar_sin)
        self.assertEqual(leap.target_calendar_cos, regular.target_calendar_cos)

    def test_warmup_and_exact_1073_predictor_schema(self):
        self.assertEqual(self.samples.target_date.min(), pd.Timestamp("2004-01-04"))
        self.assertEqual(self.samples.target_date.max(), self.daily.date.max())
        self.assertEqual(len(POSITION_FIELDS), 17)
        self.assertEqual(len(self.features), 1073)
        self.assertEqual(len(set(self.features)), 1073)
        self.assertEqual(self.features[0], "recent_lag_01__temperature_2m_mean")
        self.assertEqual(self.features[-3], "history_y5_offset_p3__calendar_cos")
        self.assertEqual(self.features[-2:], ["target_calendar_sin", "target_calendar_cos"])
        self.assertEqual(len([name for name in self.features if name.endswith("__weather_code")]), 63)
        self.assertTrue(set(self.features).isdisjoint({
            "city", "date", "target_date", "recent_days", "history_days", "target_mean_temperature",
        }))
        self.assertNotIn("target_temperature_range", self.samples)
        self.assertNotIn("target_weather_type", self.samples)
        self.assertTrue(self.samples.target_date.eq(self.samples.date + pd.Timedelta(days=1)).all())
        too_early, features = build_samples(self.daily.loc[self.daily.date.lt("2004-01-01")])
        self.assertTrue(too_early.empty)
        self.assertEqual(features, self.features)

    def test_target_and_future_weather_cannot_change_inputs(self):
        target = pd.Timestamp("2005-01-03")
        changed = self.daily.copy()
        future = changed.date.ge(target)
        weather_fields = [name for name in DAILY_VARIABLES if name != "weather_code"]
        changed.loc[future, weather_fields] += 10000
        changed.loc[future, "weather_code"] = 73
        samples, _ = build_samples(changed)
        before, after = row_for(self.samples, target), row_for(samples, target)
        pd.testing.assert_series_equal(before[self.features], after[self.features], check_names=False)
        self.assertNotEqual(before.target_mean_temperature, after.target_mean_temperature)

    def test_missing_dates_skip_affected_recent_and_historical_windows(self):
        missing = pd.to_datetime(["1999-02-25", "2004-06-15"])
        samples, _ = build_samples(self.daily.loc[~self.daily.date.isin(missing)])
        available = set(samples.target_date)
        self.assertNotIn(pd.Timestamp("2004-02-29"), available)
        self.assertNotIn(pd.Timestamp("2004-06-16"), available)
        self.assertNotIn(pd.Timestamp("2004-07-13"), available)
        self.assertIn(pd.Timestamp("2004-07-14"), available)

    def test_cities_are_isolated_and_within_city_duplicates_fail(self):
        other = self.daily.copy()
        other["city"] = "Other"
        weather_fields = [name for name in DAILY_VARIABLES if name != "weather_code"]
        other[weather_fields] += 10000
        combined = pd.concat([other, self.daily], ignore_index=True).iloc[::-1]
        samples, _ = build_samples(combined)
        original = row_for(self.samples, "2005-01-03")
        brown = row_for(samples, "2005-01-03")
        second = row_for(samples, "2005-01-03", city="Other")
        pd.testing.assert_series_equal(original[self.features], brown[self.features], check_names=False)
        for name in ("recent_lag_01__temperature_2m_mean", "history_y5_offset_m3__temperature_2m_mean"):
            self.assertEqual(second[name] - brown[name], 10000)
        duplicate = pd.concat([self.daily, self.daily.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique within each city"):
            build_samples(duplicate)

    def test_categorical_codes_missing_predictors_and_unlabeled_target(self):
        original = row_for(self.samples, "2005-01-03")
        self.assertIsInstance(original["recent_lag_01__weather_code"], str)
        changed = self.daily.copy()
        changed["weather_code"] = np.nan
        changed.loc[changed.date.eq("2005-01-02"), "relative_humidity_2m_mean"] = np.nan
        changed.loc[changed.date.eq("2005-01-03"), "temperature_2m_max"] = np.nan
        changed.loc[changed.date.eq("2005-01-04"), "temperature_2m_mean"] = np.nan
        samples, _ = build_samples(changed)
        row = row_for(samples, "2005-01-03")
        self.assertTrue(pd.isna(row["recent_lag_01__relative_humidity_2m_mean"]))
        self.assertTrue(pd.isna(row["recent_lag_01__weather_code"]))
        self.assertIsInstance(row["recent_lag_01__weather_code"], float)
        code_columns = [name for name in self.features if name.endswith("__weather_code")]
        self.assertTrue(samples[code_columns].isna().all().all())
        self.assertTrue(all(samples[name].dtype == object for name in code_columns))
        self.assertNotIn(pd.Timestamp("2005-01-04"), set(samples.target_date))


if __name__ == "__main__":
    unittest.main()
