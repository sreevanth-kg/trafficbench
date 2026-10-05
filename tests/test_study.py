"""Multi-seed selection/statistics and exact interrupted-training recovery."""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trafficbench.data import TrafficData
from trafficbench.discovery import sha256
from trafficbench.experiment import fit, make_model, seed_everything, validation_mae
from trafficbench.regime import VARIANTS
from trafficbench.study import (aggregate_forecasts, evaluate, freeze_selection,
                                paired_seed_day_bootstrap, paired_seeds, select_architecture)


class MultiSeedProtocolTests(unittest.TestCase):
    def test_selection_uses_all_seeds_and_rejects_incomplete_results(self):
        records = {"7": {"a": {"best_validation_mae_mph": 1}, "b": {"best_validation_mae_mph": 2}},
                   "42": {"a": {"best_validation_mae_mph": 5}, "b": {"best_validation_mae_mph": 2}}}
        winner, _scores = select_architecture(records, ["a", "b"], [7, 42])
        self.assertEqual(winner, "b")
        del records["42"]["b"]
        with self.assertRaises(ValueError):
            select_architecture(records, ["a", "b"], [7, 42])

    def test_aggregation_reports_sample_sd_and_requires_identical_masks(self):
        def record(mae, count=10):
            return {"metrics": [{"horizon_minutes": 15, "period": "all", "count": count, "mae_mph": mae, "rmse_mph": mae + 1}]}
        records = {"7": {"a": record(2)}, "42": {"a": record(4)}}
        rows = aggregate_forecasts(records, ["a"], [7, 42])
        self.assertEqual(rows[0]["mae_mph"]["mean"], 3)
        self.assertAlmostEqual(rows[0]["mae_mph"]["sample_standard_deviation"], np.sqrt(2))
        records["42"]["a"] = record(4, 11)
        with self.assertRaises(ValueError):
            aggregate_forecasts(records, ["a"], [7, 42])

    def test_seed_day_bootstrap_preserves_known_gain_with_unequal_day_counts(self):
        counts = np.array([10, 20, 5])
        result = paired_seed_day_bootstrap(np.tile(counts * 2, (5, 1)), counts, repetitions=100)
        self.assertEqual(result["mean_improvement"], 2)
        self.assertEqual(result["seed_and_day_bootstrap_95_percent_interval"], [2, 2])

    def test_partial_seed_aggregation_and_pairing_do_not_require_unfinished_runs(self):
        row = {"horizon_minutes": 15, "period": "all", "count": 10, "mae_mph": 2, "rmse_mph": 3}
        records = {"7": {"graphwavenet": {"metrics": [row]}, "regimeliquid": {"metrics": [row]}},
                   "42": {"graphwavenet": {"metrics": [row]}}, "87": {}}
        self.assertEqual(paired_seeds(records, "regimeliquid", [7, 42, 87]), [7])
        rows = aggregate_forecasts(records, ["graphwavenet", "regimeliquid", "lstm"], [7, 42, 87])
        self.assertEqual(rows[0]["seeds_used"], [7, 42])
        self.assertEqual(rows[1]["seeds_used"], [7])
        self.assertIsNone(rows[1]["mae_mph"]["sample_standard_deviation"])

    def test_frozen_selection_excludes_later_seed_scores_and_is_stable_as_they_finish(self):
        plan = {"config": {"seeds": [7, 21, 42], "candidates": ["a", "b"]}, "selection_seeds": [7, 21]}
        records = {str(seed): {"a": {"best_validation_mae_mph": 1}, "b": {"best_validation_mae_mph": 2}} for seed in (7, 21)}
        records["42"] = {"a": {"best_validation_mae_mph": 1000}, "b": {"best_validation_mae_mph": 0}}
        with tempfile.TemporaryDirectory() as temporary:
            first = freeze_selection(Path(temporary), plan, records)
            self.assertEqual(first["selected"], "a")
            self.assertEqual(first["validation_scores"]["a"]["values"], [1, 1])
            records["42"]["a"]["best_validation_mae_mph"] = 2000
            self.assertEqual(freeze_selection(Path(temporary), plan, records), first)

    def test_partial_evaluation_separates_confirmation_and_reuses_cached_predictions(self):
        raw = np.full((30, 3), 30, dtype=np.float32)
        timestamps = np.datetime64("2012-06-01") + np.arange(30) * np.timedelta64(5, "m")
        origins = {"train": np.array([11]), "val": np.array([12]), "test": np.array([13, 14])}
        data = TrafficData(raw, raw.copy(), np.ones_like(raw, dtype=bool), np.zeros((30, 3, 2), dtype=np.float32),
                           timestamps, ["a", "b", "c"], np.eye(3, dtype=np.float32), origins, 12, 13, 30, 10,
                           np.full(3, 30, dtype=np.float32))
        args = SimpleNamespace(seeds=[7, 21, 42, 87, 123], baselines=["lstm", "graphwavenet"], candidates=list(VARIANTS),
                               epochs=50, batch_size=2, data="unused", pairs=2, departures=2)
        plan = {"scope": "smoke_subset", "config": {"seeds": args.seeds, "baselines": args.baselines,
                                                   "candidates": args.candidates, "deployment_seed": 42},
                "dataset_manifest": {}, "protocol": "fixture", "selection_seeds": [7, 21],
                "confirmation_seeds": [42, 87, 123], "confirmation_models": ["regimeliquid"]}

        def fake_route(route_args):
            directory = Path(route_args.forecasts)
            forecast = json.loads((directory / "forecast_report.json").read_text())
            trips, summary = [], {}
            for index, label in enumerate(["static", *forecast["models"]]):
                seconds = 100 + index
                summary[label] = {"trips": 2, "mean_simulated_travel_seconds": seconds,
                                  "mean_paired_improvement_vs_static_percent": -index}
                for origin in (13, 14):
                    trips.append({"policy": label, "forecast_origin_index": origin, "source_sensor": "a", "target_sensor": "b",
                                  "departure_timestamp": "2012-06-01T01:20", "simulated_travel_seconds": seconds})
            (directory / "routing_report.json").write_text(json.dumps({"trips": trips, "summary": summary, "complete_trips": 2, "attempted_trips": 4}))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = {str(seed): {} for seed in args.seeds}
            for seed in args.seeds:
                methods = [*args.baselines, *args.candidates] if seed in (7, 21) else ["graphwavenet", "regimeliquid"] if seed == 42 else ["graphwavenet"] if seed == 87 else []
                seed_dir = directory / "seeds" / f"seed-{seed}"
                seed_dir.mkdir(parents=True)
                for method in methods:
                    model = make_model(method, data)
                    checkpoint = seed_dir / f"{method}.pt"
                    torch.save(model.state_dict(), checkpoint)
                    records[str(seed)][method] = {"checkpoint_sha256": sha256(checkpoint), "parameters": 1, "training_seconds": 1,
                                                 "epochs_run": 1, "best_epoch": 1, "reached_epoch_cap": False,
                                                 "best_validation_mae_mph": 1 if method == "liquidgraph" else 2, "history": []}
            selection = freeze_selection(directory, plan, records)
            with patch("trafficbench.routing.run", side_effect=fake_route):
                evaluate(args, directory, selection, data, origins, torch.device("cpu"), records)
                with patch("trafficbench.study.predict", side_effect=AssertionError("Cached predictions should be reused")):
                    evaluate(args, directory, selection, data, origins, torch.device("cpu"), records)
            report = json.loads((directory / "comparison_report.json").read_text())
            self.assertFalse(report["schedule_complete"])
            confirmation = [row for row in report["paired_comparison_with_graphwavenet"] if row["seed_group"] == "held_out_confirmation"]
            self.assertEqual(len(confirmation), 6)
            self.assertTrue(all(row["seeds_used"] == [42] and row["model"] == "regimeliquid" for row in confirmation))
            markdown = (directory / "comparison.md").read_text()
            self.assertIn("do not confirm the selected architecture", markdown)
            self.assertIn("one seed; SD unavailable", markdown)

    def test_resume_restores_optimizer_scheduler_shuffle_and_dropout_rng(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                self.check_resume(torch.device(device))

    def check_resume(self, device):
        args = SimpleNamespace(epochs=6, patience=10, learning_rate=0.01)
        data = SimpleNamespace(mean=0, std=1)
        generator = torch.Generator().manual_seed(4)
        x = torch.randn(9, 2, generator=generator)
        y = x.sum(dim=1, keepdim=True)
        dataset = TensorDataset(x, y, torch.ones_like(y, dtype=torch.bool))

        def setup(seed):
            seed_everything(seed)
            model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)).to(device)
            train = DataLoader(dataset, batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(73))
            val = DataLoader(dataset, batch_size=3, shuffle=False)
            return model, train, val

        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            first.mkdir()
            second.mkdir()
            uninterrupted, train, val = setup(73)
            def plateau(model, loader, data, device):
                validation_mae(model, loader, data, device)
                return 1.0

            with patch("trafficbench.experiment.validation_mae", side_effect=plateau):
                expected = fit(uninterrupted, train, val, data, device, args, "tiny", first)
            calls = 0

            def interrupt(model, loader, data, device):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("Simulated interruption")
                return plateau(model, loader, data, device)

            interrupted, train, val = setup(73)
            with patch("trafficbench.experiment.validation_mae", side_effect=interrupt):
                with self.assertRaises(RuntimeError):
                    fit(interrupted, train, val, data, device, args, "tiny", second, progress=True)
            resumed, train, val = setup(9999)
            with patch("trafficbench.experiment.validation_mae", side_effect=plateau):
                actual = fit(resumed, train, val, data, device, args, "tiny", second, progress=True, resume=True)
            self.assertEqual(actual["best_epoch"], expected["best_epoch"])
            for field in ("train_mae_mph", "val_mae_mph"):
                self.assertEqual([row[field] for row in actual["history"]], [row[field] for row in expected["history"]])
            for key, value in uninterrupted.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
