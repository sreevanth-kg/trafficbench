"""Regression checks for leakage, masking, graph propagation and time-aware replay."""

import unittest

import networkx as nx
import numpy as np
import torch

from trafficbench.data import causal_fill, split_origins
from trafficbench.metrics import speed_metrics
from trafficbench.models import DiffusionConvolution, GraphWaveNet
from trafficbench.routing import MPH_TO_MPS, select_path, simulate_path


class TemporalProtocolTests(unittest.TestCase):
    def test_targets_do_not_overlap_splits_and_gaps_are_excluded(self):
        timestamps = np.datetime64("2012-03-01") + np.arange(200) * np.timedelta64(5, "m")
        timestamps[55:] += np.timedelta64(1, "D")
        splits, train_end, val_end = split_origins(timestamps)
        train_targets = {t + h for t in splits["train"] for h in range(1, 13)}
        val_targets = {t + h for t in splits["val"] for h in range(1, 13)}
        test_targets = {t + h for t in splits["test"] for h in range(1, 13)}
        self.assertLess(max(train_targets), train_end)
        self.assertGreaterEqual(min(val_targets), train_end)
        self.assertLess(max(val_targets), val_end)
        self.assertGreaterEqual(min(test_targets), val_end)
        self.assertFalse(train_targets & val_targets or val_targets & test_targets)
        for values in splits.values():
            for t in values:
                self.assertTrue(np.all(np.diff(timestamps[t - 11:t + 13]) == np.timedelta64(5, "m")))

    def test_future_values_do_not_change_inputs_or_training_statistics(self):
        first = np.array([[10.0], [20.0], [0.0], [0.0], [90.0]], dtype=np.float32)
        second = first.copy()
        second[-1] = 10000
        a = causal_fill(first, train_end=2)
        b = causal_fill(second, train_end=2)
        np.testing.assert_array_equal(a[0][:4], b[0][:4])
        self.assertEqual(a[0][3, 0], 20)
        self.assertEqual(a[2:4], b[2:4])
        np.testing.assert_array_equal(a[4], b[4])

    def test_missing_values_and_congestion_share_correct_horizon_masks(self):
        truth = np.full((1, 12, 3), [0.0, 20.0, 60.0], dtype=np.float32)
        prediction = np.full_like(truth, [999.0, 24.0, 62.0])
        rows = speed_metrics(prediction, truth, 40)
        for row in rows:
            self.assertEqual(row["count"], 2 if row["period"] == "all" else 1)
            self.assertEqual(row["mae_mph"], 3 if row["period"] == "all" else 4)
            self.assertAlmostEqual(row["rmse_mph"], np.sqrt(10) if row["period"] == "all" else 4)


class GraphModelTests(unittest.TestCase):
    def test_directed_diffusion_reaches_a_neighbor(self):
        convolution = DiffusionConvolution(1, support_count=1, dropout=0)
        with torch.no_grad():
            convolution.projection.weight.zero_()
            convolution.projection.bias.zero_()
            convolution.projection.weight[0, 1, 0, 0] = 1
        values = torch.tensor([[[[7.0], [0.0]]]])
        support = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        output = convolution(values, [support])
        self.assertEqual(output[0, 0, 1, 0].item(), 7)
        self.assertEqual(output[0, 0, 0, 0].item(), 0)

    def test_learned_adjacency_receives_gradient(self):
        torch.manual_seed(7)
        model = GraphWaveNet(np.eye(4, dtype=np.float32), channels=4)
        output = model(torch.randn(2, 12, 4, 2))
        self.assertEqual(tuple(output.shape), (2, 12, 4))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.nodevec1.grad).all())
        self.assertGreater(model.nodevec1.grad.abs().sum().item(), 0)


class RouteSimulationTests(unittest.TestCase):
    def test_traversal_integrates_speed_changes_and_units(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, length_m=10 * MPH_TO_MPS * 300 + 20 * MPH_TO_MPS * 150)
        truth = np.array([[10.0, 10.0], [20.0, 20.0]])
        result = simulate_path(graph, [0, 1], truth, np.ones_like(truth, dtype=bool), departure_step=0)
        self.assertAlmostEqual(result, 450)
        valid = np.ones_like(truth, dtype=bool)
        valid[1, 1] = False
        self.assertIsNone(simulate_path(graph, [0, 1], truth, valid, departure_step=0))

    def test_route_uses_forecast_but_is_scored_on_observed_speeds(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, length_m=100)
        graph.add_edge(1, 3, length_m=100)
        graph.add_edge(0, 2, length_m=150)
        graph.add_edge(2, 3, length_m=150)
        static = select_path(graph, 0, 3, np.array([60, 60, 60, 60]))
        predicted = select_path(graph, 0, 3, np.array([60, 5, 60, 60]))
        self.assertEqual(static, [0, 1, 3])
        self.assertEqual(predicted, [0, 2, 3])
        truth = np.array([[60.0, 5.0, 60.0, 60.0]])
        valid = np.ones_like(truth, dtype=bool)
        self.assertLess(simulate_path(graph, predicted, truth, valid, 0),
                        simulate_path(graph, static, truth, valid, 0))


if __name__ == "__main__":
    unittest.main()
