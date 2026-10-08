"""Train and evaluate all baselines on identical, leakage-free traffic windows."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Windows, limit_origins, load_data
from .metrics import regime_frequencies, speed_metrics
from .models import GraphWaveNet, SensorLSTM


def make_model(name, data):
    if name == "lstm":
        return SensorLSTM()
    if name == "graphwavenet":
        return GraphWaveNet(data.adjacency)
    from .regime import build_proposed
    return build_proposed(data, name)


def masked_mae(prediction, truth, mask):
    error = torch.where(mask, torch.abs(prediction - truth), 0)
    return error.sum() / mask.sum().clamp_min(1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def validation_mae(model, loader, data, device):
    model.eval()
    total, count = 0.0, 0
    for x, y, mask in loader:
        prediction = model(x.to(device)).cpu() * data.std + data.mean
        total += torch.where(mask, (prediction - y).abs(), 0).double().sum().item()
        count += mask.sum().item()
    if count == 0:
        raise ValueError("Validation split has no observed targets.")
    return total / count


def fit(model, train_loader, val_loader, data, device, args, name, output, progress=False, resume=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    checkpoint = output / f"{name}.pt"
    state_file = output / f"{name}_training_state.pt"
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    previous_seconds = 0.0
    best_weights = None
    if resume and state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, best_epoch, stale = state["best"], state["best_epoch"], state["stale"]
        history, previous_seconds = state["history"], state["training_seconds"]
        best_weights = state["best_model"]
        # Restore the matching best checkpoint, even if a crash occurred between
        # saving a newer best and committing its corresponding epoch state.
        torch.save(best_weights, checkpoint)
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        train_loader.generator.set_state(state["loader_rng"])
        random.setstate(state["python_rng"])
        numpy_rng = state["numpy_rng"]
        np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
        print(f"Resuming {name} after epoch {len(history)}", flush=True)
    started = time.perf_counter()
    for epoch in range(len(history) + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        epoch_started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "training_outputs"):
                prediction, auxiliary = model.training_outputs(x, y, mask)
            else:
                prediction, auxiliary = model(x), 0
            point_loss = masked_mae(prediction, (y - data.mean) / data.std, mask)
            loss = point_loss + auxiliary
            if not torch.isfinite(loss):
                raise ValueError(f"{name} produced a non-finite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            observed = mask.sum().item()
            total += point_loss.detach().item() * observed * data.std
            count += observed
        val_mae = validation_mae(model, val_loader, data, device)
        scheduler.step(val_mae)
        history.append({"epoch": epoch, "train_mae_mph": total / max(count, 1),
                        "val_mae_mph": val_mae, "seconds": time.perf_counter() - epoch_started})
        print(f"{name} epoch {epoch}/{args.epochs}: train MAE {history[-1]['train_mae_mph']:.3f}, val MAE {val_mae:.3f}", flush=True)
        if val_mae < best:
            best, best_epoch, stale = val_mae, epoch, 0
            torch.save(model.state_dict(), checkpoint)
            if progress:
                best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if progress:
            numpy_rng = np.random.get_state()
            state = {"model": model.state_dict(), "best_model": best_weights,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
                     "training_seconds": previous_seconds + time.perf_counter() - started,
                     "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                     "loader_rng": train_loader.generator.get_state(), "python_rng": random.getstate(),
                     "numpy_rng": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:])}
            temporary = state_file.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(state_file)
            status = {"epoch": epoch, "maximum_epochs": args.epochs, "best_epoch": best_epoch,
                      "best_validation_mae_mph": best, "last_validation_mae_mph": val_mae,
                      "stale_epochs": stale, "training_seconds": state["training_seconds"],
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "finished": stale >= args.patience or epoch == args.epochs}
            temporary = output / f"{name}_progress.tmp"
            temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / f"{name}_progress.json")
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return {"training_seconds": previous_seconds + time.perf_counter() - started, "best_epoch": best_epoch,
            "best_validation_mae_mph": best, "history": history,
            "parameters": sum(parameter.numel() for parameter in model.parameters())}


@torch.inference_mode()
def predict(model, loader, data, device):
    model.eval()
    # Warm up CUDA kernels outside the measured test loop.
    first = next(iter(loader))[0].to(device)
    model(first)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = []
    for x, _y, _mask in loader:
        outputs.append((model(x.to(device)) * data.std + data.mean).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    predictions = np.concatenate(outputs).astype(np.float32)
    return predictions, time.perf_counter() - started


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.congestion_mph <= 0 or args.learning_rate <= 0 or args.threads < 1:
        raise ValueError("epochs, batch size, patience, learning rate, threads and congestion threshold must be positive.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("Select each baseline model at most once per run.")
    if any(limit < 0 for limit in (args.max_train, args.max_val, args.max_test)):
        raise ValueError("Sample limits must be nonnegative (0 means all windows).")
    torch.set_num_threads(args.threads)
    data = load_data(args.data)
    origins = {split: limit_origins(values, getattr(args, f"max_{split}")) for split, values in data.origins.items()}
    datasets = {split: Windows(data, values) for split, values in origins.items()}
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    truth = np.stack([data.raw[t + 1:t + 13] for t in origins["test"]])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data.dataset, "speed_units": "mph", "sampling_minutes": 5,
        "history_steps": 12, "forecast_steps": 12,
        "input_features": ["training-normalized causally filled speed", "time of day fraction"],
        "scope": "smoke_subset" if any(len(origins[k]) < len(data.origins[k]) for k in origins) else "full_dataset",
        "protocol": "70/10/20 raw-time split; all labels stay within split; windows crossing time gaps are excluded; training-only global scaler; causal input fill; original missing labels masked",
        "congestion_definition": f"Observed target speed > 0 and < {args.congestion_mph} mph; same mask for all models",
        "normalization": {"mean": data.mean, "std": data.std},
        "sensor_ids": data.sensor_ids,
        "split_boundaries": {"train_end_exclusive": str(data.timestamps[data.train_end]),
                             "val_end_exclusive": str(data.timestamps[data.val_end])},
        "samples": {k: {"used": len(origins[k]), "available": len(data.origins[k])} for k in origins},
        "regime_frequencies": regime_frequencies(data, origins),
        "dataset_manifest": json.loads((Path(args.data) / "manifest.json").read_text()),
        "software": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
                     "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "arguments": vars(args),
        "model_code_sha256": hashlib.sha256((Path(__file__).parent / "models.py").read_bytes()).hexdigest(),
        "models": {},
    }
    rows = []
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False)
    for name in args.models:
        seed_everything(args.seed)
        if name == "persistence":
            started = time.perf_counter()
            predictions = np.broadcast_to(data.filled[origins["test"], None, :], truth.shape).copy()
            inference_seconds = time.perf_counter() - started
            details = {"training_seconds": 0.0, "parameters": 0, "history": []}
        else:
            model = make_model(name, data).to(device)
            train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                                      generator=torch.Generator().manual_seed(args.seed))
            val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
            details = fit(model, train_loader, val_loader, data, device, args, name, output)
            predictions, inference_seconds = predict(model, test_loader, data, device)
            del model
        metrics = speed_metrics(predictions, truth, args.congestion_mph)
        details.update({"inference_seconds": inference_seconds, "metrics": metrics,
                        "inference_scope": "All selected test windows; CPU persistence, configured neural device; excludes metrics and serialization"})
        report["models"][name] = details
        rows.extend({"model": name, **row} for row in metrics)
        np.savez_compressed(output / f"{name}_forecasts.npz", prediction=predictions,
                            origins=origins["test"], sensor_ids=np.asarray(data.sensor_ids))
        # Keep completed-model progress readable if a later model fails or training is interrupted.
        (output / "forecast_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{name}: " + "; ".join(f"{r['horizon_minutes']}m MAE={r['mae_mph']:.3f} RMSE={r['rmse_mph']:.3f}" for r in metrics if r["period"] == "all"), flush=True)
        del predictions
    with (output / "forecast_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved forecast benchmark to {output}", flush=True)
