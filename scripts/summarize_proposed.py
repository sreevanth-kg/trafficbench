#!/usr/bin/env python3
"""Summarize frozen forecasts and uncertainty in paired simulated route gains."""

import argparse
import json
from pathlib import Path

import numpy as np


def route_comparison(report, candidate, baseline="graphwavenet"):
    def keyed(policy):
        rows = [row for row in report["trips"] if row["policy"] == policy]
        result = {(row["forecast_origin_index"], row["source_sensor"], row["target_sensor"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired route records")
        return result

    reference, ours = keyed(baseline), keyed(candidate)
    if not reference or reference.keys() != ours.keys():
        raise ValueError("Routing comparison requires identical, nonempty paired trips")
    keys = sorted(reference)
    first = np.asarray([reference[key]["simulated_travel_seconds"] for key in keys])
    second = np.asarray([ours[key]["simulated_travel_seconds"] for key in keys])
    dates = [reference[key]["departure_timestamp"][:10] for key in keys]
    days, index = np.unique(dates, return_inverse=True)
    counts = np.bincount(index, minlength=len(days))
    rng = np.random.default_rng(2026)
    sampled_days = rng.integers(len(days), size=(1000, len(days)))

    def compare(values):
        totals = np.bincount(index, weights=values, minlength=len(days))
        samples = totals[sampled_days].sum(axis=1) / counts[sampled_days].sum(axis=1)
        return {"mean": float(values.mean()), "day_block_bootstrap_95_percent_interval": np.quantile(samples, [0.025, 0.975]).tolist()}

    return {"baseline": baseline, "candidate": candidate, "trips": len(keys), "departure_days": len(days),
            "interpretation": "Positive is better; conditional on the fixed OD pairs and common complete-trip subset; resamples departure days, not independent trips",
            "improvement_seconds": compare(first - second),
            "paired_improvement_percent": compare(100 * (first - second) / first),
            "fraction_faster_than_baseline": float(((first - second) > 1e-8).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="results/metr-la/proposed-10ep")
    args = parser.parse_args()
    directory = Path(args.experiment)
    selection = json.loads((directory / "selection.json").read_text())
    forecast = json.loads((directory / "evaluation" / "forecast_report.json").read_text())
    routes = json.loads((directory / "evaluation" / "routing_report.json").read_text())
    cpu = json.loads((directory / "cpu_inference_benchmark.json").read_text())
    name = selection["selected"]
    ours = forecast["models"][name]
    baseline = forecast["models"]["graphwavenet"]
    reference = {(row["horizon_minutes"], row["period"]): row for row in baseline["metrics"]}
    metrics = []
    for row in ours["metrics"]:
        other = reference[row["horizon_minutes"], row["period"]]
        if row["count"] != other["count"]:
            raise ValueError("Forecast comparison masks differ")
        metrics.append({**row, "mae_reduction_vs_graphwavenet_percent": 100 * (other["mae_mph"] - row["mae_mph"]) / other["mae_mph"],
                        "rmse_reduction_vs_graphwavenet_percent": 100 * (other["rmse_mph"] - row["rmse_mph"]) / other["rmse_mph"]})
    summary = {"selected": name, "seed": selection["arguments"]["seed"], "epochs": selection["arguments"]["epochs"],
               "validation_ablations": {key: {field: record[field] for field in ("best_validation_mae_mph", "parameters", "training_seconds", "best_epoch")}
                                        for key, record in selection["candidates"].items()},
               "total_search_training_seconds": sum(record["training_seconds"] for record in selection["candidates"].values()),
               "forecast_comparison": metrics, "forecast_day_bootstrap": ours["paired_comparison_with_graphwavenet"],
               "uncertainty": ours["uncertainty_metrics"], "validation_interval_corrections_mph": ours["validation_interval_corrections_mph"],
               "mean_expert_weights": ours["mean_expert_weights"], "routing": routes["summary"],
               "routing_comparison": route_comparison(routes, name), "cpu_inference": cpu}
    (directory / "comparison_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
