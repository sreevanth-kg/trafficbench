"""Product-boundary and calibration invariants, without training another model."""

import unittest

import numpy as np

from trafficbench.discovery import calibrate_intervals, interval_metrics, paired_day_bootstrap
from trafficbench.inference import preprocess_request


class InferenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"sensor_ids": ["a", "b"], "mean": 50, "std": 10,
                         "fallback_speeds_mph": [40, 60]}
        timestamps = np.datetime64("2012-06-01T08:00") + np.arange(12) * np.timedelta64(5, "m")
        self.request = {"sensor_ids": ["b", "a"], "speeds_mph": [[0, 20]] + [[60, 30]] * 11,
                        "timestamps": [str(value) for value in timestamps]}

    def test_sensor_order_and_leading_missing_history_are_handled_causally(self):
        features, _origin, fraction = preprocess_request(self.request, self.metadata)
        np.testing.assert_array_equal(features[0, :, 0], [-3, 1])
        self.assertLess(fraction, 1)
        self.request["previous_speeds_mph"] = [70, 20]
        features, _origin, _fraction = preprocess_request(self.request, self.metadata)
        np.testing.assert_array_equal(features[0, :, 0], [-3, 2])

    def test_unknown_network_and_untrained_time_gaps_are_rejected(self):
        wrong = {**self.request, "sensor_ids": ["b", "c"]}
        with self.assertRaises(ValueError):
            preprocess_request(wrong, self.metadata)
        self.request["timestamps"][5] = "2012-06-01T09:30"
        with self.assertRaises(ValueError):
            preprocess_request(self.request, self.metadata)


class UncertaintyAndComparisonTests(unittest.TestCase):
    def test_calibration_ignores_missing_targets(self):
        quantiles = np.zeros((2, 12, 1, 3))
        quantiles[..., 0] = 40
        quantiles[..., 1] = 45
        quantiles[..., 2] = 50
        truth = np.full((2, 12, 1), 55.0)
        truth[0] = 0
        self.assertEqual(calibrate_intervals(quantiles, truth), [5.0] * 12)

    def test_interval_coverage_is_reported_by_fixed_regime(self):
        quantiles = np.zeros((1, 12, 3, 3), dtype=float)
        quantiles[..., 0] = 40
        quantiles[..., 1] = 45
        quantiles[..., 2] = 50
        truth = np.full((1, 12, 3), 45.0)
        latest = np.full((1, 3), 45.0)
        rows = interval_metrics(quantiles, truth, [0.0] * 12, latest)
        self.assertEqual(rows[0]["coverage_by_regime"]["free_flow"]["count"], 3)
        self.assertEqual(rows[0]["coverage_by_regime"]["congested"]["count"], 0)
        self.assertEqual(rows[0]["coverage_by_regime"]["transition"]["count"], 0)
        self.assertEqual(rows[0]["coverage_by_regime"]["free_flow"]["empirical_coverage"], 1.0)

    def test_day_block_comparison_preserves_known_paired_gain(self):
        truth = np.full((4, 12, 1), 20.0)
        baseline = truth + 4
        candidate = truth + 2
        timestamps = np.array(["2012-06-01T08:00", "2012-06-01T09:00", "2012-06-02T08:00", "2012-06-02T09:00"], dtype="datetime64[ns]")
        rows = paired_day_bootstrap(baseline, candidate, truth, timestamps, repetitions=20)
        for row in rows:
            self.assertEqual(row["mae_improvement_mph"], 2)
            self.assertEqual(row["day_block_bootstrap_95_percent_interval_mph"], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
