"""Architecture selection on validation only, followed by one frozen test evaluation."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Windows, limit_origins, load_data
from .experiment import fit, make_model, predict, seed_everything
from .metrics import regime_frequencies, speed_metrics
from .regime import VARIANTS


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def model_code_hash():
    directory = Path(__file__).parent
    return {name: sha256(directory / name) for name in ("regime.py", "models.py")}


def configure_device(args):
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable.")
    return device


def discover(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.threads < 1:
        raise ValueError("Training settings must be positive.")
    if len(set(args.candidates)) != len(args.candidates) or args.max_train < 0 or args.max_val < 0:
        raise ValueError("Candidates must be unique and sample limits nonnegative.")
    data = load_data(args.data)
    train_origins = limit_origins(data.origins["train"], args.max_train)
    val_origins = limit_origins(data.origins["val"], args.max_val)
    device = configure_device(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data.dataset,
        "scope": "full_dataset" if len(train_origins) == len(data.origins["train"]) and len(val_origins) == len(data.origins["val"]) else "smoke_subset",
        "selection_rule": "Lowest validation masked MAE over all twelve horizons; checkpoints selected by the same rule; test forecasts are not generated during search",
        "arguments": vars(args), "train_windows": len(train_origins), "val_windows": len(val_origins),
        "model_code_hashes": model_code_hash(),
        "dataset_manifest": json.loads((Path(args.data) / "manifest.json").read_text()),
        "regime_frequencies": regime_frequencies(data, {"train": train_origins, "val": val_origins}),
        "auxiliary_objectives": "Masked point MAE + 0.05 normalized pinball loss + 0.02 regime cross entropy (expert variants); rglflow_msr additionally uses 0.01 history-only domain-routing cross entropy",
        "candidates": {}, "selected": None,
    }
    for name in args.candidates:
        seed_everything(args.seed)
        model = make_model(name, data).to(device)
        train_loader = DataLoader(Windows(data, train_origins), batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(Windows(data, val_origins), batch_size=args.batch_size, shuffle=False)
        record = fit(model, train_loader, val_loader, data, device, args, name, output)
        val_predictions, _ = predict(model, val_loader, data, device)
        val_truth = np.stack([data.raw[t + 1:t + 13] for t in val_origins])
        record["validation_horizon_metrics"] = speed_metrics(val_predictions, val_truth)
        record["checkpoint_sha256"] = sha256(output / f"{name}.pt")
        report["candidates"][name] = record
        (output / "validation_search.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del model, val_predictions, val_truth
    name = min(report["candidates"], key=lambda candidate: report["candidates"][candidate]["best_validation_mae_mph"])
    report["selected"] = name
    # Freeze the choice before reading/generating any new test-model forecasts.
    (output / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Selected on validation only:", name, report["candidates"][name]["best_validation_mae_mph"], flush=True)


@torch.inference_mode()
def distribution_predictions(model, loader, device):
    quantiles, gates = [], []
    for x, _y, _mask in loader:
        details = model.forward_details(x.to(device))
        quantiles.append(details["quantiles_mph"].cpu().numpy())
        gates.append(details["gates"].cpu().numpy())
    return np.concatenate(quantiles), np.concatenate(gates)


def calibrate_intervals(quantiles, truth, coverage=0.8):
    """Validation CQR-style corrections; coverage is empirical, not an IID guarantee."""
    scores = np.maximum(np.maximum(quantiles[..., 0] - truth, truth - quantiles[..., 2]), 0)
    valid = np.isfinite(truth) & (truth > 0)
    corrections = []
    for step in range(12):
        values = scores[:, step][valid[:, step]]
        if not len(values):
            raise ValueError("Cannot calibrate a horizon without observed validation speeds.")
        rank_level = min(1.0, np.ceil((len(values) + 1) * coverage) / len(values))
        corrections.append(float(np.quantile(values, rank_level, method="higher")))
    return corrections


def interval_metrics(quantiles, truth, corrections, latest=None):
    valid = np.isfinite(truth) & (truth > 0)
    lower = np.maximum(0, quantiles[..., 0] - np.asarray(corrections)[None, :, None])
    upper = quantiles[..., 2] + np.asarray(corrections)[None, :, None]
    rows = []
    if latest is not None:
        change = np.abs(truth - latest[:, None])
        regimes = {"free_flow": valid & (truth >= 40) & (change < 8),
                   "congested": valid & (truth < 40),
                   "transition": valid & (truth >= 40) & (change >= 8)}
    else:
        regimes = {"all": valid}
    for minutes, step in ((15, 2), (30, 5), (60, 11)):
        mask = valid[:, step]
        covered = (truth[:, step] >= lower[:, step]) & (truth[:, step] <= upper[:, step])
        row = {"horizon_minutes": minutes,
               "empirical_coverage": float(covered[mask].mean()),
               "mean_width_mph": float((upper[:, step] - lower[:, step])[mask].mean()),
               "coverage_by_regime": {}}
        for name, regime_mask in regimes.items():
            selected = regime_mask[:, step]
            count = int(selected.sum())
            row["coverage_by_regime"][name] = {
                "count": count,
                "empirical_coverage": float(covered[selected].mean()) if count else None,
                "mean_width_mph": float((upper[:, step] - lower[:, step])[selected].mean()) if count else None,
            }
        rows.append(row)
    return rows


def paired_day_bootstrap(baseline, candidate, truth, origin_timestamps, repetitions=1000):
    """Resample forecast-origin days, respecting within-day error dependence."""
    days, day_index = np.unique(origin_timestamps.astype("datetime64[D]"), return_inverse=True)
    rng = np.random.default_rng(2026)
    sampled_days = rng.integers(0, len(days), size=(repetitions, len(days)))
    rows = []
    for minutes, step in ((15, 2), (30, 5), (60, 11)):
        valid = np.isfinite(truth[:, step]) & (truth[:, step] > 0)
        for period, mask in (("all", valid), ("congested", valid & (truth[:, step] < 40))):
            difference = np.where(mask, np.abs(baseline[:, step] - truth[:, step]) - np.abs(candidate[:, step] - truth[:, step]), 0).sum(axis=1)
            counts = mask.sum(axis=1)
            daily_difference = np.bincount(day_index, weights=difference, minlength=len(days))
            daily_count = np.bincount(day_index, weights=counts, minlength=len(days))
            denominator = daily_count[sampled_days].sum(axis=1)
            bootstrap = daily_difference[sampled_days].sum(axis=1) / np.maximum(denominator, 1)
            rows.append({"horizon_minutes": minutes, "period": period, "days": len(days),
                         "mae_improvement_mph": float(difference.sum() / counts.sum()),
                         "day_block_bootstrap_95_percent_interval_mph": np.quantile(bootstrap, [0.025, 0.975]).tolist()})
    return rows


def evaluate_selected(args):
    output = Path(args.experiment)
    selection = json.loads((output / "selection.json").read_text())
    if selection["scope"] != "full_dataset":
        raise ValueError("Final evaluation requires a full-data validation search.")
    if selection["model_code_hashes"] != model_code_hash():
        raise ValueError("Model implementation changed after architecture selection.")
    name = selection["selected"]
    checkpoint = output / f"{name}.pt"
    if sha256(checkpoint) != selection["candidates"][name]["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after selection.")
    data = load_data(args.data)
    baseline_dir = Path(args.baseline).resolve()
    baseline_report = json.loads((baseline_dir / "forecast_report.json").read_text())
    hashes = lambda manifest: {item["file"]: item["sha256"] for item in manifest["artifacts"]}
    current_manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    if any(hashes(current_manifest) != hashes(report["dataset_manifest"]) for report in (baseline_report, selection)):
        raise ValueError("Baseline, selected model and evaluation data must match.")
    if baseline_report["scope"] != "full_dataset" or baseline_report["arguments"]["epochs"] != selection["arguments"]["epochs"]:
        raise ValueError("Use a full-data baseline with the same epoch budget.")
    device = configure_device(args)
    model = make_model(name, data).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    val_origins = data.origins["val"]
    val_loader = DataLoader(Windows(data, val_origins), batch_size=args.batch_size, shuffle=False)
    val_quantiles, _ = distribution_predictions(model, val_loader, device)
    val_truth = np.stack([data.raw[t + 1:t + 13] for t in val_origins])
    corrections = calibrate_intervals(val_quantiles, val_truth)
    del val_quantiles, val_truth
    origins = data.origins["test"]
    test_loader = DataLoader(Windows(data, origins), batch_size=args.batch_size, shuffle=False)
    predictions, inference_seconds = predict(model, test_loader, data, device)
    quantiles, gates = distribution_predictions(model, test_loader, device)
    truth = np.stack([data.raw[t + 1:t + 13] for t in origins])
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    forecast_sources = {}
    for baseline_name in baseline_report["models"]:
        filename = baseline_dir / f"{baseline_name}_forecasts.npz"
        with np.load(filename, allow_pickle=False) as saved:
            if not np.array_equal(saved["origins"], origins) or saved["sensor_ids"].tolist() != data.sensor_ids:
                raise ValueError("Baseline forecast origins or sensor ordering differ.")
        forecast_sources[baseline_name] = str(filename)
    np.savez_compressed(evaluation / f"{name}_forecasts.npz", prediction=predictions, origins=origins,
                        sensor_ids=np.asarray(data.sensor_ids))
    forecast_sources[name] = str((evaluation / f"{name}_forecasts.npz").resolve())
    record = {**selection["candidates"][name], "metrics": speed_metrics(predictions, truth),
              "inference_seconds": inference_seconds, "uncertainty_metrics": interval_metrics(
                  quantiles, truth, corrections, data.filled[origins]),
              "validation_interval_corrections_mph": corrections,
               "mean_expert_weights": gates.mean(axis=(0, 1, 2), dtype=np.float64).tolist(),
              "single_seed": selection["arguments"]["seed"], "epochs": selection["arguments"]["epochs"]}
    with np.load(forecast_sources["graphwavenet"], allow_pickle=False) as saved:
        record["paired_comparison_with_graphwavenet"] = paired_day_bootstrap(saved["prediction"], predictions, truth, data.timestamps[origins])
    report = {**baseline_report, "forecast_sources": forecast_sources,
              "proposed_model_selection": str((output / "selection.json").resolve()),
              "model_code_hashes": model_code_hash(), "models": {**baseline_report["models"], name: record}}
    (evaluation / "forecast_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = [{"model": method, **row} for method, details in report["models"].items() for row in details["metrics"]]
    with (evaluation / "forecast_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    export_bundle(output / "bundle", checkpoint, name, data, corrections)
    print("Selected model:", name)
    print(json.dumps(record["metrics"], indent=2))
    print("Saved comparison to", evaluation, "and deployable inference bundle to", output / "bundle")


def export_bundle(bundle, checkpoint, name, data, corrections, source=None, evaluation_forecast_filename=None):
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, bundle / "model.pt")
    # Supports live inference without loading historical training/test files.
    np.savez_compressed(bundle / "graph.npz", adjacency=data.adjacency)
    metadata = {"dataset": data.dataset, "architecture": name, "mean": data.mean, "std": data.std, "sensor_ids": data.sensor_ids,
                "fallback_speeds_mph": data.static_speeds.tolist(), "history_steps": 12,
                "forecast_minutes": list(range(5, 65, 5)), "interval_corrections_mph": corrections,
                "interval_target_coverage": 0.8, "model_code_hashes": model_code_hash(),
                "checkpoint_sha256": sha256(bundle / "model.pt"),
                 "graph_sha256": sha256(bundle / "graph.npz"),
                 "evaluation_forecast_filename": evaluation_forecast_filename or f"{name}_forecasts.npz",
                 "source": source or f"Single {data.dataset} network, fixed-budget research prototype; cross-network transfer not established"}
    (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    # History only: the example request contains no future observations.
    origins = data.origins["test"]
    origin = int(origins[len(origins) // 2])
    request = {"sensor_ids": data.sensor_ids,
               "timestamps": [str(value) for value in data.timestamps[origin - 11:origin + 1]],
               "speeds_mph": data.filled[origin - 11:origin + 1].tolist()}
    (bundle / "example_request.json").write_text(json.dumps(request, indent=2) + "\n")
