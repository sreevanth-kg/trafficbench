"""Checks for real liquid behavior, ordered mixtures and history-only inference."""

import unittest

import numpy as np
from ncps.torch import CfC
import torch

from trafficbench.regime import RegimeLiquidGraph, RoutedMultiScaleLiquidGraph


class RegimeArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = RegimeLiquidGraph(np.eye(4, dtype=np.float32), 55, 15).eval()
        self.inputs = torch.randn(2, 12, 4, 2)
        self.inputs[..., 1] = 0.4

    def test_quantile_mixtures_are_nonnegative_and_ordered(self):
        details = self.model.forward_details(self.inputs)
        quantiles = details["quantiles_mph"]
        self.assertEqual(tuple(quantiles.shape), (2, 12, 4, 3))
        self.assertTrue((quantiles >= 0).all())
        self.assertTrue((quantiles[..., 0] <= quantiles[..., 1]).all())
        self.assertTrue((quantiles[..., 1] <= quantiles[..., 2]).all())
        torch.testing.assert_close(details["gates"].sum(-1), torch.ones(2, 12, 4))

    def test_liquid_encoder_is_genuine_and_elapsed_time_changes_output(self):
        self.assertIsInstance(self.model.temporal, CfC)
        slow = self.model(self.inputs, timespans=torch.ones(2, 12) * 0.5)
        regular = self.model(self.inputs, timespans=torch.ones(2, 12))
        self.assertGreater((slow - regular).abs().sum().item(), 0)

    def test_regime_prior_responds_to_observed_congestion(self):
        fast = self.inputs.clone()
        slow = self.inputs.clone()
        fast[..., 0] = (65 - 55) / 15
        slow[..., 0] = (20 - 55) / 15
        self.assertGreater(self.model.observed_prior(slow)[..., 1].mean().item(),
                           self.model.observed_prior(fast)[..., 1].mean().item())

    def test_training_targets_cannot_change_forward_predictions(self):
        before = self.model(self.inputs).detach()
        truth = torch.full((2, 12, 4), 20.0)
        _prediction, auxiliary = self.model.training_outputs(self.inputs, truth, torch.ones_like(truth, dtype=torch.bool))
        self.assertTrue(torch.isfinite(auxiliary))
        torch.testing.assert_close(before, self.model(self.inputs).detach())

    def test_all_ablations_train_and_router_receives_gradient(self):
        for name in ("regimeliquid", "liquidgraph", "regimegru", "regimeliquid_noprior"):
            model = RegimeLiquidGraph(np.eye(4, dtype=np.float32), 55, 15, variant=name)
            truth = torch.full((2, 12, 4), 35.0)
            prediction, extra = model.training_outputs(self.inputs, truth, torch.ones_like(truth, dtype=torch.bool))
            (prediction.square().mean() + extra).backward()
            if model.use_experts:
                self.assertGreater(model.router[-1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(prediction).all())

    def test_multiscale_router_is_history_only_and_blends_three_domains(self):
        model = RoutedMultiScaleLiquidGraph(np.eye(4, dtype=np.float32), 55, 15).eval()
        details = model.forward_details(self.inputs)
        self.assertEqual(tuple(details["domain_gates"].shape), (2, 4, 3))
        torch.testing.assert_close(details["domain_gates"].sum(-1), torch.ones(2, 4))
        self.assertTrue(torch.isfinite(details["point"]).all())
        truth = torch.full((2, 12, 4), 35.0)
        prediction, auxiliary = model.training_outputs(
            self.inputs, truth, torch.ones_like(truth, dtype=torch.bool)
        )
        self.assertEqual(tuple(prediction.shape), (2, 12, 4))
        self.assertTrue(torch.isfinite(auxiliary))
        (prediction.square().mean() + auxiliary).backward()
        self.assertGreater(model.domain_router[-1].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
