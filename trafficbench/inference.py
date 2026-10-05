"""Product-facing history-only inference using an exported model bundle."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .discovery import model_code_hash, sha256
from .regime import REGIMES, RegimeLiquidGraph


def preprocess_request(request, metadata):
    ids = [str(value) for value in request.get("sensor_ids", [])]
    expected = metadata["sensor_ids"]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("sensor_ids must contain each sensor from this network bundle exactly once.")
    raw = np.asarray(request.get("speeds_mph"), dtype=np.float32)
    if raw.shape != (12, len(ids)):
        raise ValueError(f"speeds_mph must be a 12-by-{len(ids)} history matrix.")
    timestamps = pd.DatetimeIndex(pd.to_datetime(request.get("timestamps", [])))
    if len(timestamps) != 12 or timestamps.hasnans or timestamps.tz is not None:
        raise ValueError("Supply 12 naive timestamps in the same network clock convention as training.")
    if not np.all(np.diff(timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)) == 300_000_000_000):
        raise ValueError("This evaluated bundle requires consecutive five-minute history readings.")
    ordering = [ids.index(sensor) for sensor in expected]
    raw = raw[:, ordering]
    valid = np.isfinite(raw) & (raw > 0)
    fallback = np.asarray(metadata["fallback_speeds_mph"], dtype=np.float32)
    if "previous_speeds_mph" in request:
        previous = np.asarray(request["previous_speeds_mph"], dtype=np.float32)
        if previous.shape != (len(ids),):
            raise ValueError("previous_speeds_mph must have one value per input sensor.")
        previous = previous[ordering]
        fallback = np.where(np.isfinite(previous) & (previous > 0), previous, fallback)
    filled = np.empty_like(raw)
    for step in range(12):
        fallback = np.where(valid[step], raw[step], fallback)
        filled[step] = fallback
    fraction = (timestamps.hour * 60 + timestamps.minute).to_numpy(dtype=np.float32) / 1440
    features = np.stack([(filled - metadata["mean"]) / metadata["std"],
                         np.broadcast_to(fraction[:, None], raw.shape)], axis=-1)
    return features.astype(np.float32), timestamps[-1], float(valid.mean())


class ForecastEngine:
    def __init__(self, bundle, device="cpu", threads=4):
        torch.set_num_threads(threads)
        bundle = Path(bundle)
        self.metadata = json.loads((bundle / "metadata.json").read_text())
        if self.metadata["model_code_hashes"] != model_code_hash():
            raise ValueError("Model code does not match this exported bundle.")
        if sha256(bundle / "model.pt") != self.metadata["checkpoint_sha256"] or sha256(bundle / "graph.npz") != self.metadata["graph_sha256"]:
            raise ValueError("Exported model/graph checksum mismatch.")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        with np.load(bundle / "graph.npz", allow_pickle=False) as graph:
            self.model = RegimeLiquidGraph(graph["adjacency"], self.metadata["mean"], self.metadata["std"],
                                           variant=self.metadata["architecture"]).to(self.device)
        self.model.load_state_dict(torch.load(bundle / "model.pt", map_location=self.device, weights_only=True))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, request):
        started = time.perf_counter()
        features, origin, observed_fraction = preprocess_request(request, self.metadata)
        details = self.model.forward_details(torch.from_numpy(features[None]).to(self.device))
        quantiles = details["quantiles_mph"][0].cpu().numpy()
        corrections = np.asarray(self.metadata["interval_corrections_mph"])[:, None]
        lower = np.maximum(0, quantiles[..., 0] - corrections)
        upper = quantiles[..., 2] + corrections
        result = {
            "architecture": self.metadata["architecture"], "forecast_origin": str(origin),
            "sensor_ids": self.metadata["sensor_ids"], "forecast_minutes": self.metadata["forecast_minutes"],
            "speed_units": "mph", "median_speeds_mph": quantiles[..., 1].tolist(),
            "lower_speeds_mph": lower.tolist(), "upper_speeds_mph": upper.tolist(),
            "interval_target_coverage": self.metadata["interval_target_coverage"],
            "interval_description": "Validation-calibrated empirical intervals; future/cross-city coverage is not guaranteed",
            "regime_labels": list(REGIMES) if self.model.use_experts else ["global"],
            "regime_weights": details["gates"][0].cpu().numpy().tolist(),
            "observed_history_fraction": observed_fraction,
            "network_scope": self.metadata["source"],
        }
        result["prediction_latency_ms"] = (time.perf_counter() - started) * 1000
        return result


def run(args):
    engine = ForecastEngine(args.bundle, args.device, args.threads)
    request = json.loads(Path(args.input).read_text())
    result = engine.predict(request)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Predicted {len(result['sensor_ids'])} sensors across 12 horizons in {result['prediction_latency_ms']:.2f} ms; saved {output}")
