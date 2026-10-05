#!/usr/bin/env python3
"""Same-hardware, batch-one CPU comparison against Graph WaveNet."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from trafficbench.inference import ForecastEngine, preprocess_request
from trafficbench.models import GraphWaveNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="results/metr-la/proposed-10ep/bundle")
    parser.add_argument("--baseline", default="results/metr-la/baseline-10ep")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1 or args.threads < 1:
        parser.error("samples and threads must be positive")
    engine = ForecastEngine(args.bundle, "cpu", args.threads)
    bundle = Path(args.bundle)
    request = json.loads((bundle / "example_request.json").read_text())
    features, _time, _fraction = preprocess_request(request, engine.metadata)
    inputs = torch.from_numpy(features[None])
    with np.load(bundle / "graph.npz", allow_pickle=False) as graph:
        baseline = GraphWaveNet(graph["adjacency"])
    baseline.load_state_dict(torch.load(Path(args.baseline) / "graphwavenet.pt", map_location="cpu", weights_only=True))
    baseline.eval()
    summary = {}
    with torch.inference_mode():
        for name, model in (("graphwavenet", baseline), (engine.metadata["architecture"], engine.model)):
            for _ in range(10):
                model(inputs)
            samples = []
            for _ in range(args.samples):
                start = time.perf_counter_ns()
                model(inputs)
                samples.append((time.perf_counter_ns() - start) / 1_000_000)
            summary[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                             "p50_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, 0.95)),
                             "samples": len(samples)}
    report = {"scope": "CPU model forward only; batch 1, all 207 sensors, 12 horizons; excludes HTTP, JSON and preprocessing",
              "threads": args.threads, "summary": summary}
    (bundle.parent / "cpu_inference_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
