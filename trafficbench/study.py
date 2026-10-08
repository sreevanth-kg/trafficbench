"""Resumable multi-seed training, validation-only selection and frozen comparison."""

import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Windows, limit_origins, load_data
from .discovery import (calibrate_intervals, configure_device, distribution_predictions,
                        export_bundle, interval_metrics, model_code_hash, sha256)
from .experiment import fit, make_model, predict, seed_everything
from .metrics import regime_frequencies, speed_metrics
from .regime import VARIANTS


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def implementation_hashes():
    directory = Path(__file__).parent
    return {name: sha256(directory / name) for name in
            ("study.py", "experiment.py", "data.py", "metrics.py", "discovery.py", "models.py", "regime.py", "routing.py")}


def seed_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
            "values": values.tolist()}


def select_architecture(records, candidates, seeds):
    scores = {}
    for name in candidates:
        values = []
        for seed in seeds:
            record = records.get(str(seed), {}).get(name)
            if record is None:
                raise ValueError(f"Incomplete validation study: {name}, seed {seed}")
            values.append(record["best_validation_mae_mph"])
        scores[name] = seed_statistics(values)
    return min(scores, key=lambda name: scores[name]["mean"]), scores


def paired_seed_day_bootstrap(daily_differences, daily_counts, repetitions=2000):
    """Resample seeds and common days; positive differences favor the candidate."""
    differences = np.asarray(daily_differences, dtype=np.float64)
    counts = np.asarray(daily_counts, dtype=np.float64)
    if differences.ndim != 2 or differences.shape[1:] != counts.shape or not (counts.sum() > 0):
        raise ValueError("Expected [seeds, days] differences and observed per-day counts")
    seeds, days = differences.shape
    rng = np.random.default_rng(2026)
    sampled_seeds = rng.integers(seeds, size=(repetitions, seeds))
    sampled_days = rng.integers(days, size=(repetitions, days))
    totals = differences[sampled_seeds[:, :, None], sampled_days[:, None, :]].sum(axis=(1, 2))
    denominators = seeds * counts[sampled_days].sum(axis=1)
    samples = totals[denominators > 0] / denominators[denominators > 0]
    return {"mean_improvement": float(differences.sum() / (counts.sum() * seeds)),
            "seed_and_day_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "seeds": seeds, "days": days, "bootstrap_repetitions": repetitions}


def aggregate_forecasts(records, methods, seeds):
    rows = []
    for name in methods:
        available = [seed for seed in seeds if name in records.get(str(seed), {})]
        if not available:
            continue
        indexed = [{(row["horizon_minutes"], row["period"]): row for row in records[str(seed)][name]["metrics"]}
                   for seed in available]
        for key in sorted(indexed[0]):
            if any(set(table) != set(indexed[0]) or table[key]["count"] != indexed[0][key]["count"] for table in indexed):
                raise ValueError("Seed comparison requires identical forecast masks")
            rows.append({"model": name, "seeds_used": available, "horizon_minutes": key[0], "period": key[1], "count_per_seed": indexed[0][key]["count"],
                         "mae_mph": seed_statistics([table[key]["mae_mph"] for table in indexed]),
                         "rmse_mph": seed_statistics([table[key]["rmse_mph"] for table in indexed])})
    return rows


def print_status(directory):
    progress_file = directory / "progress.json"
    if not progress_file.exists():
        print("Study has not started:", directory)
        return
    progress = json.loads(progress_file.read_text())
    pid = progress.get("runner_pid")
    try:
        os.kill(pid, 0)
        progress["runner_process_present"] = True
    except (ProcessLookupError, TypeError):
        progress["runner_process_present"] = False
    active = progress.get("active_run")
    if active:
        path = directory / "seeds" / f"seed-{active['seed']}" / f"{active['model']}_progress.json"
        if path.exists():
            progress["active_epoch_progress"] = json.loads(path.read_text())
    print(json.dumps(progress, indent=2))


def read_records(directory, seeds):
    records = {}
    for seed in seeds:
        path = directory / "seeds" / f"seed-{seed}" / "training_records.json"
        records[str(seed)] = json.loads(path.read_text()) if path.exists() else {}
    return records


def training_schedule(plan):
    config = plan["config"]
    return plan.get("run_schedule", [{"seed": seed, "model": name} for seed in config["seeds"]
                                     for name in [*config["baselines"], *config["candidates"]]])


def freeze_selection(directory, plan, records):
    seeds = plan.get("selection_seeds", plan["config"]["seeds"])
    candidates = plan["config"]["candidates"]
    if any(name not in records.get(str(seed), {}) for seed in seeds for name in candidates):
        return None
    selected, scores = select_architecture(records, candidates, seeds)
    # Later seed records must not affect selection or its frozen provenance.
    selection_records = {str(seed): {name: records[str(seed)][name] for name in candidates} for seed in seeds}
    path = directory / "selection.json"
    if path.exists():
        selection = json.loads(path.read_text())
        if selection["selected"] != selected or selection["selection_records"] != selection_records:
            raise ValueError("Previously frozen selection no longer matches its selection-seed records")
        return selection
    selection = {**plan, "selected": selected, "validation_scores": scores, "selection_seeds": seeds,
                 "selection_records": selection_records, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(path, selection)
    print("Frozen validation selection:", selected, "using seeds", seeds, scores[selected], flush=True)
    return selection


def paired_seeds(records, candidate, seeds, baseline="graphwavenet"):
    return [seed for seed in seeds if candidate in records.get(str(seed), {}) and baseline in records[str(seed)]]


def run(args):
    directory = Path(args.output)
    if args.stage == "status":
        print_status(directory)
        return
    directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "evaluate":
        # Evaluation reads only atomically completed runs; it can run while the
        # training process continues. A separate lock protects its report files.
        with (directory / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _run(args)
        return
    with (directory / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This study is already running; use --stage status to inspect it") from None
        _run(args)


def _run(args):
    directory = Path(args.output)
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        raise ValueError("Training settings must be positive")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("Supply at least two distinct nonnegative seeds")
    if len(set(args.candidates)) != len(args.candidates) or len(set(args.baselines)) != len(args.baselines):
        raise ValueError("Candidate and baseline names must be unique")
    if "graphwavenet" not in args.baselines:
        raise ValueError("Include graphwavenet as the paired comparison baseline")
    if args.pairs < 1 or args.departures < 1:
        raise ValueError("Routing pair and departure counts must be positive")
    if any(limit < 0 for limit in (args.max_train, args.max_val, args.max_test)):
        raise ValueError("Sample limits must be nonnegative")
    if 42 not in args.seeds and args.deployment_seed is None:
        raise ValueError("Specify deployment-seed when the predefined seed 42 is absent")
    deployment_seed = 42 if args.deployment_seed is None else args.deployment_seed
    if deployment_seed not in args.seeds:
        raise ValueError("Deployment seed must be in the study seed list")
    device = configure_device(args)
    data = load_data(args.data)
    origins = {split: limit_origins(data.origins[split], getattr(args, f"max_{split}")) for split in ("train", "val", "test")}
    directory.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items() if key not in ("command", "stage", "resume")}
    config["deployment_seed"] = deployment_seed
    config["data"] = str(Path(args.data).resolve())
    config["output"] = str(directory.resolve())
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    # Match actual artifacts, rather than just trusting a stale manifest.
    if any(sha256(Path(args.data) / item["file"]) != item["sha256"] for item in manifest["artifacts"]):
        raise ValueError("Dataset artifact differs from its manifest")
    plan_path = directory / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if args.stage != "evaluate" and not args.resume:
            raise ValueError("Existing study: use --resume or a fresh output directory")
        if plan["config"] != config or plan["implementation_hashes"] != implementation_hashes() or plan["dataset_manifest"] != manifest:
            raise ValueError("Settings, implementation or data changed; use a fresh study directory")
    else:
        if args.stage == "evaluate":
            raise ValueError("Train the study before evaluating")
        plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "implementation_hashes": implementation_hashes(), "model_code_hashes": model_code_hash(),
                "dataset": data.dataset, "dataset_manifest": manifest,
                "scope": "full_dataset" if all(len(origins[k]) == len(data.origins[k]) for k in origins) else "smoke_subset",
                "samples": {k: {"used": len(origins[k]), "available": len(data.origins[k])} for k in origins},
                "selection_rule": "Lowest mean best-checkpoint validation MAE over all predefined seeds and twelve horizons; no test-model forecasts during selection",
                "comparison_rule": "Independent models per seed; report mean and sample SD, not an ensemble or best test seed",
                "protocol": "70/10/20 raw-time split, history/targets 12/12, training-only normalization, causal input fill, original missing labels masked, congestion below 40 mph",
                "optimizer": {"name": "Adam", "weight_decay": 1e-4, "gradient_clip_norm": 5,
                              "scheduler": "ReduceLROnPlateau, patience 3, factor 0.5"},
                "software": {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
        write_json(plan_path, plan)
    progress_path = directory / "progress.json"
    schedule = training_schedule(plan)
    records = read_records(directory, args.seeds)
    for seed in args.seeds:
        for name, record in records[str(seed)].items():
            if sha256(directory / "seeds" / f"seed-{seed}" / f"{name}.pt") != record["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed: {name}, seed {seed}")
    selection = freeze_selection(directory, plan, records)
    if args.stage == "evaluate":
        if selection is None:
            raise ValueError("Finish all selection-seed candidate runs before test evaluation")
        evaluate(args, directory, selection, data, origins, device, records)
        return
    progress = {"runner_pid": os.getpid(), "stage": "training",
                "completed_runs": sum(item["model"] in records[str(item["seed"])] for item in schedule),
                "total_runs": len(schedule), "active_run": None,
                "selection_seeds": plan.get("selection_seeds", args.seeds),
                "confirmation_seeds": plan.get("confirmation_seeds", []), "run_schedule": schedule}
    write_json(progress_path, progress)
    last_evaluated_runs = None
    for item in schedule:
        seed, name = item["seed"], item["model"]
        if name in records[str(seed)]:
            continue
        seed_directory = directory / "seeds" / f"seed-{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)
        record_path = seed_directory / "training_records.json"
        progress["active_run"] = item
        write_json(progress_path, progress)
        print(f"Run {progress['completed_runs'] + 1}/{progress['total_runs']}: {name}, seed {seed}", flush=True)
        seed_everything(seed)
        model = make_model(name, data).to(device)
        train_loader = DataLoader(Windows(data, origins["train"]), batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(Windows(data, origins["val"]), batch_size=args.batch_size, shuffle=False)
        record = fit(model, train_loader, val_loader, data, device, args, name, seed_directory, progress=True, resume=args.resume)
        record["checkpoint_sha256"] = sha256(seed_directory / f"{name}.pt")
        record["epochs_run"] = len(record["history"])
        record["reached_epoch_cap"] = record["epochs_run"] == args.epochs
        records[str(seed)][name] = record
        write_json(record_path, records[str(seed)])
        progress["completed_runs"] += 1
        progress["active_run"] = None
        write_json(progress_path, progress)
        del model
        selection = freeze_selection(directory, plan, records)
        # Revised studies publish a snapshot after selection, and after each
        # completed confirmation seed. Reports do not require the full queue.
        end_of_seed = all(run["model"] in records[str(seed)] for run in schedule if run["seed"] == seed)
        if args.stage == "all" and selection is not None and end_of_seed:
            progress["stage"] = "evaluation"
            progress["selected"] = selection["selected"]
            write_json(progress_path, progress)
            with (directory / ".evaluation.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                evaluate(args, directory, selection, data, origins, device, records)
            last_evaluated_runs = progress["completed_runs"]
            progress["stage"] = "training"
            write_json(progress_path, progress)
    selection = freeze_selection(directory, plan, records)
    if selection is None:
        raise ValueError("Schedule ended without completing the selection seeds")
    if args.stage != "train" and last_evaluated_runs != progress["completed_runs"]:
        with (directory / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            evaluate(args, directory, selection, data, origins, device, records)
    progress["stage"] = "completed"
    progress["active_run"] = None
    progress["selected"] = selection["selected"]
    write_json(progress_path, progress)


def evaluate(args, directory, selection, data, origins, device, records=None):
    name = selection["selected"]
    records = records or read_records(directory, args.seeds)
    confirmation_models = selection.get("confirmation_models", [name])
    candidates = list(dict.fromkeys([name, *confirmation_models]))
    methods = list(dict.fromkeys([*args.baselines, *candidates]))
    evaluation = directory / "evaluation"
    evaluation.mkdir(exist_ok=True)
    test_loader = DataLoader(Windows(data, origins["test"]), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(Windows(data, origins["val"]), batch_size=args.batch_size, shuffle=False)
    truth = np.stack([data.raw[t + 1:t + 13] for t in origins["test"]])
    val_truth = np.stack([data.raw[t + 1:t + 13] for t in origins["val"]])
    forecast_report = {"dataset": data.dataset, "scope": selection["scope"], "dataset_manifest": selection["dataset_manifest"],
                       "models": {}, "forecast_sources": {}, "protocol": selection["protocol"],
                       "regime_frequencies": regime_frequencies(data, origins),
                       "seeds": args.seeds, "selected": name,
                       "selection_seeds": selection["selection_seeds"],
                       "confirmation_seeds": selection.get("confirmation_seeds", []),
                       "proposed_model_selection": str((directory / "selection.json").resolve())}
    persistence = np.broadcast_to(data.filled[origins["test"], None, :], truth.shape).copy()
    persist_path = evaluation / "persistence_forecasts.npz"
    np.savez_compressed(persist_path, prediction=persistence, origins=origins["test"], sensor_ids=np.asarray(data.sensor_ids))
    forecast_report["models"]["persistence"] = {"metrics": speed_metrics(persistence, truth), "parameters": 0, "training_seconds": 0}
    forecast_report["forecast_sources"]["persistence"] = str(persist_path.resolve())
    del persistence
    tested = {str(seed): {} for seed in args.seeds}
    days, day_index = np.unique(data.timestamps[origins["test"]].astype("datetime64[D]"), return_inverse=True)
    paired = {method: {(minutes, period): {} for minutes in (15, 30, 60) for period in ("all", "congested")} for method in candidates}
    day_counts = {}
    for seed in args.seeds:
        seed_directory = directory / "seeds" / f"seed-{seed}"
        baseline_predictions = None
        for method in methods:
            if method not in records.get(str(seed), {}):
                continue
            label = f"{method}_seed{seed}"
            path = evaluation / f"{label}_forecasts.npz"
            cache = evaluation / f"{label}_record.json"
            model = make_model(method, data).to(device)
            checkpoint = seed_directory / f"{method}.pt"
            if sha256(checkpoint) != records[str(seed)][method]["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after freezing selection")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            if cache.exists() and path.exists():
                record = json.loads(cache.read_text())
                if record["checkpoint_sha256"] != records[str(seed)][method]["checkpoint_sha256"] or sha256(path) != record["forecasts_sha256"]:
                    raise ValueError("Cached evaluation artifact changed")
                with np.load(path, allow_pickle=False) as saved:
                    if not np.array_equal(saved["origins"], origins["test"]) or saved["sensor_ids"].tolist() != data.sensor_ids:
                        raise ValueError("Cached forecast origins or sensor ordering differ")
                    predictions = saved["prediction"]
                metrics = record["metrics"]
            else:
                predictions, seconds = predict(model, test_loader, data, device)
                metrics = speed_metrics(predictions, truth)
                record = {**records[str(seed)][method], "metrics": metrics, "inference_seconds": seconds, "seed": seed}
            if method == "graphwavenet":
                baseline_predictions = predictions
            if method in candidates:
                if "uncertainty_metrics" not in record:
                    quantiles, gates = distribution_predictions(model, val_loader, device)
                    corrections = calibrate_intervals(quantiles, val_truth)
                    del quantiles, gates
                    quantiles, gates = distribution_predictions(model, test_loader, device)
                    record["uncertainty_metrics"] = interval_metrics(
                        quantiles, truth, corrections, data.filled[origins["test"]])
                    record["validation_interval_corrections_mph"] = corrections
                    record["mean_expert_weights"] = gates.mean(axis=(0, 1, 2), dtype=np.float64).tolist()
                    del quantiles, gates
                corrections = record["validation_interval_corrections_mph"]
                if method == name and seed == selection["config"]["deployment_seed"]:
                    if selection["scope"] != "full_dataset":
                        print("Smoke subset: no deployable bundle exported", flush=True)
                    else:
                        export_bundle(directory / "bundle", checkpoint, method, data, corrections,
                                      f"Single {data.dataset} network; {args.epochs}-epoch cap, {len(args.seeds)}-seed study; predefined deployment seed {seed}; cross-network transfer not established",
                                      evaluation_forecast_filename=f"{method}_seed{seed}_forecasts.npz")
                for minutes, step in ((15, 2), (30, 5), (60, 11)) if baseline_predictions is not None else ():
                    valid = np.isfinite(truth[:, step]) & (truth[:, step] > 0)
                    for period, mask in (("all", valid), ("congested", valid & (truth[:, step] < 40))):
                        difference = np.where(mask, np.abs(baseline_predictions[:, step] - truth[:, step]) - np.abs(predictions[:, step] - truth[:, step]), 0).sum(axis=1, dtype=np.float64)
                        key = (minutes, period)
                        paired[method][key][seed] = np.bincount(day_index, weights=difference, minlength=len(days))
                        day_counts[key] = np.bincount(day_index, weights=mask.sum(axis=1), minlength=len(days))
            tested[str(seed)][method] = record
            if "forecasts_sha256" not in record:
                np.savez_compressed(path, prediction=predictions, origins=origins["test"], sensor_ids=np.asarray(data.sensor_ids))
                record["forecasts_sha256"] = sha256(path)
                write_json(cache, record)
            forecast_report["models"][label] = record
            forecast_report["forecast_sources"][label] = str(path.resolve())
            write_json(evaluation / "forecast_report.json", forecast_report)
            print(f"Test {label}: " + "; ".join(f"{row['horizon_minutes']}m MAE={row['mae_mph']:.3f}" for row in metrics if row["period"] == "all"), flush=True)
            del model, predictions
    aggregated = aggregate_forecasts(tested, methods, args.seeds)
    comparisons = []
    groups = {"all_completed": args.seeds, "selection": selection["selection_seeds"],
              "held_out_confirmation": selection.get("confirmation_seeds", [])}
    for candidate in candidates:
        for group, group_seeds in groups.items():
            available = paired_seeds(tested, candidate, group_seeds)
            if not available:
                continue
            group_metrics = aggregate_forecasts(tested, ["graphwavenet", candidate], available)
            reference = {(row["horizon_minutes"], row["period"]): row for row in group_metrics if row["model"] == "graphwavenet"}
            for row in group_metrics:
                if row["model"] != candidate:
                    continue
                key = row["horizon_minutes"], row["period"]
                baseline = reference[key]
                differences = np.asarray(baseline["mae_mph"]["values"]) - row["mae_mph"]["values"]
                comparisons.append({"model": candidate, "seed_group": group, "seeds_used": available,
                                    "horizon_minutes": key[0], "period": key[1],
                                    "mae_reduction_percent_from_seed_means": 100 * (baseline["mae_mph"]["mean"] - row["mae_mph"]["mean"]) / baseline["mae_mph"]["mean"],
                                    "paired_seed_mae_improvement_mph": seed_statistics(differences),
                                    "seeds_with_lower_mae": int((differences > 0).sum()),
                                    "seed_and_day_bootstrap": paired_seed_day_bootstrap([paired[candidate][key][seed] for seed in available], day_counts[key])})
    write_json(evaluation / "per_seed_test_records.json", tested)
    routing_args = SimpleNamespace(data=args.data, forecasts=str(evaluation), pairs=args.pairs, departures=args.departures,
                                   departure_minutes=15, speed_floor_mph=5.0, seed=42)
    from .routing import run as route
    route(routing_args)
    routing_report = json.loads((evaluation / "routing_report.json").read_text())
    routing_aggregate = []
    for method in methods:
        available = [seed for seed in args.seeds if method in tested[str(seed)]]
        if not available:
            continue
        summaries = [routing_report["summary"][f"{method}_seed{seed}"] for seed in available]
        routing_aggregate.append({"model": method, "seeds_used": available, "trips_per_seed": summaries[0]["trips"],
                                  "mean_simulated_travel_seconds": seed_statistics([item["mean_simulated_travel_seconds"] for item in summaries]),
                                  "mean_paired_improvement_vs_static_percent": seed_statistics([item["mean_paired_improvement_vs_static_percent"] for item in summaries])})
    routing_comparisons = []
    for candidate in candidates:
        for group, group_seeds in groups.items():
            available = paired_seeds(tested, candidate, group_seeds)
            if available:
                routing_comparisons.append({"model": candidate, "seed_group": group, "seeds_used": available,
                                            **route_seed_comparison(routing_report, candidate, available)})
    summary = {"dataset": data.dataset, "selection": str((directory / "selection.json").resolve()), "scope": selection["scope"],
               "selected": name, "seeds": args.seeds, "maximum_epochs": args.epochs,
               "selection_seeds": selection["selection_seeds"], "confirmation_seeds": selection.get("confirmation_seeds", []),
               "confirmation_models": confirmation_models,
               "complete_training_runs": {seed: list(models) for seed, models in records.items()},
               "schedule_complete": all(item["model"] in records[str(item["seed"])] for item in training_schedule(selection)),
               "validation_scores": selection["validation_scores"], "forecast_metrics": aggregated,
               "persistence_metrics": forecast_report["models"]["persistence"]["metrics"],
               "paired_comparison_with_graphwavenet": comparisons, "routing_metrics": routing_aggregate,
               "routing_common_complete_trips": routing_report["complete_trips"],
               "routing_attempted_trips": routing_report["attempted_trips"],
               "routing_paired_comparisons": routing_comparisons,
               "total_training_seconds": sum(record["training_seconds"] for seed in records.values() for record in seed.values()),
                "budget_diagnostics": {method: {"seeds_used": [seed for seed in args.seeds if method in records[str(seed)]],
                                              "best_epochs": [records[str(seed)][method]["best_epoch"] for seed in args.seeds if method in records[str(seed)]],
                                              "epochs_run": [records[str(seed)][method]["epochs_run"] for seed in args.seeds if method in records[str(seed)]],
                                              "parameter_counts": [records[str(seed)][method]["parameters"] for seed in args.seeds if method in records[str(seed)]],
                                              "inference_seconds": [tested[str(seed)][method]["inference_seconds"] for seed in args.seeds if method in tested[str(seed)]],
                                              "runs_reaching_cap": sum(records[str(seed)][method]["reached_epoch_cap"] for seed in args.seeds if method in records[str(seed)])}
                                       for method in [*args.baselines, *args.candidates]},
                "interval_coverage": {method: {str(seed): tested[str(seed)][method].get("uncertainty_metrics", [])
                                                for seed in args.seeds if method in tested[str(seed)]}
                                      for method in candidates},
               "interpretation": "Mean and sample SD across independent trained seeds, not an ensemble. The public test period was also used in the earlier 10-epoch study. Seed/day intervals are conditional on this network and period."}
    write_json(directory / "comparison_report.json", summary)
    with (directory / "forecast_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "seeds_used", "horizon_minutes", "period", "count_per_seed", "mae_mean_mph", "mae_sd_mph", "rmse_mean_mph", "rmse_sd_mph"])
        writer.writeheader()
        for row in aggregated:
            writer.writerow({key: row[key] for key in ("model", "seeds_used", "horizon_minutes", "period", "count_per_seed")} |
                            {f"{field}_mean_mph": row[f"{field}_mph"]["mean"] for field in ("mae", "rmse")} |
                            {f"{field}_sd_mph": row[f"{field}_mph"]["sample_standard_deviation"] for field in ("mae", "rmse")})
    write_markdown(directory, summary)
    print("Completed multi-seed comparison:", directory / "comparison.md", flush=True)


def route_seed_comparison(report, candidate, seeds):
    differences, percentages = [], []
    for seed in seeds:
        policies = {}
        for method in ("graphwavenet", candidate):
            policies[method] = {(row["forecast_origin_index"], row["source_sensor"], row["target_sensor"]): row
                                for row in report["trips"] if row["policy"] == f"{method}_seed{seed}"}
        keys = sorted(policies["graphwavenet"])
        if not keys or set(keys) != set(policies[candidate]):
            raise ValueError("Routing comparison requires identical nonempty paired trips")
        dates = [policies[candidate][key]["departure_timestamp"][:10] for key in keys]
        days, index = np.unique(dates, return_inverse=True)
        first = np.asarray([policies["graphwavenet"][key]["simulated_travel_seconds"] for key in keys])
        second = np.asarray([policies[candidate][key]["simulated_travel_seconds"] for key in keys])
        differences.append(np.bincount(index, weights=first - second, minlength=len(days)))
        percentages.append(np.bincount(index, weights=100 * (first - second) / first, minlength=len(days)))
        counts = np.bincount(index, minlength=len(days))
    return {"improvement_seconds": paired_seed_day_bootstrap(differences, counts),
            "paired_improvement_percent": paired_seed_day_bootstrap(percentages, counts)}


def format_statistics(values, decimals=3):
    mean = f"{values['mean']:.{decimals}f}"
    sd = values["sample_standard_deviation"]
    return mean + (f" ± {sd:.{decimals}f}" if sd is not None else " (one seed; SD unavailable)")


def write_markdown(directory, summary):
    dataset_name = summary.get("dataset", "traffic dataset")
    lines = [f"# Longer-training, multi-seed {dataset_name} comparison", "",
             f"Scope: **{summary['scope']}**. Maximum epochs: **{summary['maximum_epochs']}**. Seeds: **{summary['seeds']}**.",
             f"Selected using mean validation MAE from seeds **{summary['selection_seeds']} only**: **{summary['selected']}**.",
             f"Held-out training seeds: **{summary['confirmation_seeds']}**. Scheduled training complete: **{summary['schedule_complete']}**.",
             "Cells show means and sample standard deviations of completed runs; each row lists its actual seeds.", "",
             "## Validation architecture comparison", "", "| Variant | Mean validation MAE | Sample SD |", "| --- | ---: | ---: |"]
    for name, values in summary["validation_scores"].items():
        lines.append(f"| {name} | {values['mean']:.4f} | {values['sample_standard_deviation']:.4f} |")
    if summary["confirmation_seeds"]:
        lines += ["", "Held-out confirmation is predefined for **regimeliquid vs Graph WaveNet**. Later-seed validation scores never change architecture selection."]
        if summary["selected"] not in summary["confirmation_models"]:
            lines += ["", f"The selected architecture is **{summary['selected']}**, not regimeliquid. The later seeds confirm the predefined regimeliquid model only; they do not confirm the selected architecture."]
    for period in ("all", "congested"):
        lines += ["", f"## Test speeds: {period}", "", "| Model | Completed seeds | Horizon | MAE | RMSE |", "| --- | --- | ---: | ---: | ---: |"]
        for row in summary["forecast_metrics"]:
            if row["period"] == period:
                mae, rmse = row["mae_mph"], row["rmse_mph"]
                lines.append(f"| {row['model']} | {row['seeds_used']} | {row['horizon_minutes']} min | {format_statistics(mae)} | {format_statistics(rmse)} |")
        lines += ["", "Persistence (no seed variability):"]
        for row in summary["persistence_metrics"]:
            if row["period"] == period:
                lines.append(f"- {row['horizon_minutes']} min: MAE {row['mae_mph']:.3f}, RMSE {row['rmse_mph']:.3f}.")
    lines += ["", "## Paired MAE comparison against Graph WaveNet", "",
              "Positive differences favor the named candidate. Comparisons use only seed intersections, not unmatched model means. Bootstrap resamples paired seeds and common forecast-origin days.", "",
              "| Candidate | Seed group | Paired seeds | Horizon | Traffic group | MAE reduction | Seeds improved | Gain ± seed SD, mph | 95% seed/day interval, mph |",
              "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in summary["paired_comparison_with_graphwavenet"]:
        bootstrap = row["seed_and_day_bootstrap"]
        low, high = bootstrap["seed_and_day_bootstrap_95_percent_interval"]
        lines.append(f"| {row['model']} | {row['seed_group']} | {row['seeds_used']} | {row['horizon_minutes']} min | {row['period']} | {row['mae_reduction_percent_from_seed_means']:.2f}% | {row['seeds_with_lower_mae']}/{len(row['seeds_used'])} | {format_statistics(row['paired_seed_mae_improvement_mph'])} | [{low:.3f}, {high:.3f}] |")
    if summary["confirmation_seeds"] and not any(row["seed_group"] == "held_out_confirmation" for row in summary["paired_comparison_with_graphwavenet"]):
        lines += ["", "Held-out confirmation is pending: no later seed has both paired runs completed yet."]
    lines += ["", "## Routing simulation", "",
              f"All model seeds share **{summary['routing_common_complete_trips']}/{summary['routing_attempted_trips']}** complete simulated trips. Fixed routing seed 42; 15-minute departure lead.", "",
              "| Speed source | Completed seeds | Mean trip seconds ± seed SD | Mean paired improvement vs static ± seed SD |", "| --- | --- | ---: | ---: |"]
    for row in summary["routing_metrics"]:
        times, gains = row["mean_simulated_travel_seconds"], row["mean_paired_improvement_vs_static_percent"]
        lines.append(f"| {row['model']} | {row['seeds_used']} | {format_statistics(times, 2)} | {format_statistics(gains)}% |")
    for row in summary["routing_paired_comparisons"]:
        for measure, units in (("improvement_seconds", "seconds"), ("paired_improvement_percent", "%")):
            result = row[measure]
            low, high = result["seed_and_day_bootstrap_95_percent_interval"]
            lines += ["", f"{row['model']} vs Graph WaveNet, {row['seed_group']}, paired seeds {row['seeds_used']}: mean improvement {result['mean_improvement']:.3f} {units}, 95% paired seed/day interval [{low:.3f}, {high:.3f}] {units}."]
    lines += ["", "## Parameter counts and inference latency", "",
              "Parameter counts are per model checkpoint. Inference time is the measured configured-device test prediction loop.", "",
              "| Model | Completed seeds | Parameters | Test inference seconds |", "| --- | --- | --- | ---: |"]
    for method, values in summary["budget_diagnostics"].items():
        lines.append(f"| {method} | {values['seeds_used']} | {values['parameter_counts']} | {values['inference_seconds']} |")
    lines += ["", "## Training budget diagnostics", "", "| Model | Completed seeds | Best epochs | Epochs run | Runs reaching cap |", "| --- | --- | --- | --- | ---: |"]
    for method, values in summary["budget_diagnostics"].items():
        lines.append(f"| {method} | {values['seeds_used']} | {values['best_epochs']} | {values['epochs_run']} | {values['runs_reaching_cap']} |")
    lines += ["", "## Interval coverage by horizon and regime", "",
              "Intervals are calibrated on validation data only. Regime labels use the fixed 40 mph congestion and 8 mph change thresholds.", "",
              "| Model | Seed | Horizon | Regime | Count | Coverage | Mean width (mph) |", "| --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for method, seed_records in summary["interval_coverage"].items():
        for seed, metrics in seed_records.items():
            for row in metrics:
                for regime, values in row["coverage_by_regime"].items():
                    coverage = "n/a" if values["empirical_coverage"] is None else f"{values['empirical_coverage']:.4f}"
                    width = "n/a" if values["mean_width_mph"] is None else f"{values['mean_width_mph']:.3f}"
                    lines.append(f"| {method} | {seed} | {row['horizon_minutes']} min | {regime} | {values['count']} | {coverage} | {width} |")
    lines += ["", f"Total recorded training time: {summary['total_training_seconds'] / 3600:.2f} hours, including validation/checkpointing; excluding evaluation and routing.", "",
              "These are independent model runs, not averaged ensemble predictions or the best test seed. Maximum epochs with early stopping do not guarantee convergence.",
              "The candidate family has additional pinball/router objectives; baseline optimizer settings and per-run budgets are matched. Architecture-search cost is additional.",
              "The original public test period was already examined in the 10-epoch study. This is repeated-seed evidence on that period, not a fresh external holdout.",
              "Held-out confirmation refers to unseen training seeds, not a separate city or observation period. A single completed seed does not estimate between-seed variability.",
              "Routing uses an abstract sensor graph and observed-speed replay, conditional on a common complete-trip subset; results are not measured customer savings.", ""]
    (directory / "comparison.md").write_text("\n".join(lines))
