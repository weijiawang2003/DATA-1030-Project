"""Small offline checks for split boundaries and holdout independence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather_project.modeling import _scores, run_models


def supervised_fixture():
    target_dates = pd.date_range("2019-12-30", periods=6)
    return pd.DataFrame({
        "city": "A",
        "date": target_dates - pd.Timedelta(days=1),
        "target_date": target_dates,
        "x": np.arange(6, dtype=float),
        "target_mean_temperature": [0.0, 2.0, 100.0, 200.0, 3.0, 4.0],
        "persistence_temperature": [0.0, 0.0, 2.0, 100.0, 200.0, 3.0],
    })


def run_small(frame, destination):
    # Exercise production orchestration with a cheap deterministic family.
    with patch("weather_project.modeling._families", return_value=[
        ("ridge", Ridge, [{"alpha": 1.0}], True),
    ]):
        return run_models(frame, ["x"], destination, train_end="2019-12-31",
                          validation_end="2020-01-02", targets=("temperature",))


class ModelingGuardTests(unittest.TestCase):
    def test_outcome_cannot_be_a_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "target columns cannot be features"):
                run_models(supervised_fixture(), ["x", "target_mean_temperature"], directory,
                           targets=("temperature",))

    def test_non_next_day_pairs_are_rejected(self):
        frame = supervised_fixture()
        frame.loc[0, "target_date"] += pd.Timedelta(days=1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly one day"):
                run_models(frame, ["x"], directory, targets=("temperature",))

    def test_missing_holdout_class_still_counts_in_three_class_macro_f1(self):
        score = _scores(["Clear", "Clear"], ["Clear", "Clear"], "classification")
        self.assertAlmostEqual(score["macro_f1"], 1.0 / 3.0)
        self.assertEqual(score["recall_clear"], 1.0)
        self.assertEqual(score["recall_precipitation"], 0.0)

    def test_single_class_training_is_rejected_before_model_fitting(self):
        frame = supervised_fixture()
        frame["target_weather_type"] = ["Clear", "Clear", "Cloudy", "Cloudy", "Precipitation", "Precipitation"]
        frame["persistence_weather"] = "Clear"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least two weather classes"):
                run_models(frame, ["x"], directory, train_end="2019-12-31",
                           validation_end="2020-01-02", targets=("weather",))


class HoldoutIndependenceTests(unittest.TestCase):
    def test_target_date_boundaries_and_train_only_monthly_climatology(self):
        frame = supervised_fixture()
        changed = frame.copy()
        changed.loc[2:3, "target_mean_temperature"] = [1000.0, 2000.0]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            metrics = run_small(frame, first)
            run_small(changed, second)
            predictions = pd.read_csv(Path(first) / "predictions.csv")
            changed_predictions = pd.read_csv(Path(second) / "predictions.csv")
            selection = json.loads((Path(first) / "model_selection.json").read_text())

        split_info = selection["cities"][0]["splits"]
        self.assertEqual({name: info["n_samples"] for name, info in split_info.items()},
                         {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(split_info["validation"]["first_target_date"], "2020-01-01")
        # January 1 has a December 31 feature date, but belongs in validation.
        self.assertEqual(set(predictions.loc[predictions.split.eq("validation"), "target_date"]),
                         {"2020-01-01", "2020-01-02"})
        self.assertEqual(set(metrics["split"]), {"validation", "test"})

        def climate(values, split):
            return values.loc[values.model.eq("climatology") & values.split.eq(split), "y_pred"].to_numpy()

        # January was unseen in training: validation uses the training global mean.
        np.testing.assert_allclose(climate(predictions, "validation"), [1.0, 1.0])
        np.testing.assert_allclose(climate(changed_predictions, "validation"), [1.0, 1.0])
        # Test baseline is refitted on development: its January mean uses validation,
        # never the held-out January test outcomes (3 and 4).
        np.testing.assert_allclose(climate(predictions, "test"), [150.0, 150.0])
        np.testing.assert_allclose(climate(changed_predictions, "test"), [1500.0, 1500.0])


if __name__ == "__main__":
    unittest.main()
