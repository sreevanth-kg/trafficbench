"""Snapshot route selection and time-varying replay on an abstract sensor graph.

Distances are DCRNN road-network distances, not learned adjacency weights. Edges
are sensor connections; this graph does not reconstruct legal street-level routes.
"""

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .data import load_data


MPH_TO_MPS = 0.44704
INTERVAL_SECONDS = 300


def build_graph(data, distance_file):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(data.sensor_ids)))
    lookup = {sensor: index for index, sensor in enumerate(data.sensor_ids)}
    # METR-LA includes a header; the original PEMS-BAY distance artifact does not.
    with Path(distance_file).open() as stream:
        first = next(csv.reader(stream))
    has_header = first[:2] == ["from", "to"]
    distances = pd.read_csv(distance_file, header=0 if has_header else None,
                            names=["from", "to", "distance"],
                            dtype={"from": str, "to": str})
    for row in distances.itertuples(index=False, name=None):
        source, target, length = row
        if source not in lookup or target not in lookup or not np.isfinite(length) or length <= 0:
            continue
        u, v = lookup[source], lookup[target]
        if u == v or data.adjacency[u, v] <= 0:
            continue
        if not graph.has_edge(u, v) or graph[u][v]["length_m"] > length:
            graph.add_edge(u, v, length_m=float(length))
    if not graph.number_of_edges():
        raise ValueError("No sensor connections have positive, finite road distances.")
    return graph


def edge_speed_mps(speeds, u, v, floor_mph):
    first, second = max(float(speeds[u]), floor_mph), max(float(speeds[v]), floor_mph)
    # Equal-length endpoint speed interpolation: harmonic mean, converted from mph.
    return (2 / (1 / first + 1 / second)) * MPH_TO_MPS


def select_path(graph, source, target, speeds, floor_mph=5.0):
    return nx.shortest_path(
        graph, source, target,
        weight=lambda u, v, edge: edge["length_m"] / edge_speed_mps(speeds, u, v, floor_mph),
    )


def simulate_path(graph, path, truth, valid, departure_step=2, floor_mph=5.0):
    """Integrate distance as speeds change; incomplete truth returns None, never imputed."""
    initial = departure_step * INTERVAL_SECONDS
    elapsed = float(initial)
    for u, v in zip(path, path[1:]):
        remaining = graph[u][v]["length_m"]
        while remaining > 1e-8:
            bucket = int(elapsed // INTERVAL_SECONDS)
            if bucket >= len(truth) or not (valid[bucket, u] and valid[bucket, v]):
                return None
            speed = edge_speed_mps(truth[bucket], u, v, floor_mph)
            available = (bucket + 1) * INTERVAL_SECONDS - elapsed
            covered = speed * available
            if remaining <= covered:
                elapsed += remaining / speed
                remaining = 0
            else:
                remaining -= covered
                elapsed += available
    return elapsed - initial


def choose_pairs(graph, count, seed):
    # Structural selection only: no test traffic or model predictions influence OD pairs.
    candidates = [(source, target) for source, lengths in nx.all_pairs_shortest_path_length(graph)
                  for target, hops in lengths.items() if hops >= 2]
    if not candidates:
        raise ValueError("Sensor graph has no reachable multi-edge OD pairs.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return [candidates[index] for index in indices]


def run(args):
    if args.pairs < 1 or args.departures < 1 or args.speed_floor_mph <= 0:
        raise ValueError("Pair/departure counts and speed floor must be positive.")
    output = Path(args.forecasts)
    forecast_report = json.loads((output / "forecast_report.json").read_text())
    data = load_data(args.data)
    current_manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    hashes = lambda manifest: {item["file"]: item["sha256"] for item in manifest["artifacts"]}
    if hashes(current_manifest) != hashes(forecast_report["dataset_manifest"]):
        raise ValueError("Routing data differs from the data used for forecasting.")
    predictions, origins = {}, None
    for name in forecast_report["models"]:
        filename = forecast_report.get("forecast_sources", {}).get(name, output / f"{name}_forecasts.npz")
        with np.load(filename, allow_pickle=False) as saved:
            if saved["sensor_ids"].tolist() != data.sensor_ids:
                raise ValueError(f"Sensor ordering mismatch in {name} forecasts.")
            if origins is not None and not np.array_equal(origins, saved["origins"]):
                raise ValueError("All route policies must use identical forecast origins.")
            origins = saved["origins"].copy()
            prediction = saved["prediction"]
            if prediction.shape != (len(origins), 12, len(data.sensor_ids)) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid forecast shape or values for {name}.")
            predictions[name] = prediction.copy()
    if origins is None or not set(origins).issubset(set(data.origins["test"])):
        raise ValueError("Forecast origins must be valid test origins.")
    distance_file = current_manifest.get("distance_file", "distances_la_2012.csv")
    graph = build_graph(data, Path(args.data) / distance_file)
    pairs = choose_pairs(graph, args.pairs, args.seed)
    selected = np.linspace(0, len(origins) - 1, min(args.departures, len(origins)), dtype=int)
    step = args.departure_minutes // 5 - 1
    records = []
    skipped = 0
    for sample in selected:
        origin = int(origins[sample])
        truth = data.raw[origin + 1:origin + 13]
        valid = data.valid[origin + 1:origin + 13]
        policies = {"static": data.static_speeds, **{name: forecast[sample, step] for name, forecast in predictions.items()}}
        for source, target in pairs:
            paths = {name: select_path(graph, source, target, speed, args.speed_floor_mph)
                     for name, speed in policies.items()}
            times = {name: simulate_path(graph, path, truth, valid, step, args.speed_floor_mph)
                     for name, path in paths.items()}
            # Score every policy on the same complete-ground-truth trips.
            if any(value is None for value in times.values()):
                skipped += 1
                continue
            static_time = times["static"]
            for name, path in paths.items():
                gain = (static_time - times[name]) / static_time * 100
                records.append({
                    "forecast_origin_index": origin,
                    "departure_timestamp": str(data.timestamps[origin + step + 1]),
                    "source_sensor": data.sensor_ids[source], "target_sensor": data.sensor_ids[target],
                    "policy": name, "simulated_travel_seconds": times[name],
                    "improvement_vs_static_percent": gain,
                    "path_changed_vs_static": path != paths["static"],
                    "path_sensor_ids": [data.sensor_ids[node] for node in path],
                })
    if not records:
        raise ValueError("No common trips have complete truth within the forecast replay horizon.")
    summary = {}
    for name in ("static", *predictions):
        group = [record for record in records if record["policy"] == name]
        summary[name] = {
            "trips": len(group),
            "mean_simulated_travel_seconds": float(np.mean([record["simulated_travel_seconds"] for record in group])),
            "mean_paired_improvement_vs_static_percent": float(np.mean([record["improvement_vs_static_percent"] for record in group])),
            "fraction_faster_than_static": float(np.mean([record["improvement_vs_static_percent"] > 1e-8 for record in group])),
            "fraction_routes_changed": float(np.mean([record["path_changed_vs_static"] for record in group])),
        }
    report = {
        "dataset": data.dataset,
        "scope": f"Abstract {data.dataset} sensor-graph simulation; not legal street-level navigation or measured trip time",
        "forecast_scope": forecast_report["scope"],
        "routing_algorithm": "NetworkX Dijkstra with fixed snapshot edge costs; route fixed before replay",
        "topology": "Positive directed DCRNN physical adjacency intersected with finite positive published road distances; learned adjacency is not used as road connectivity",
        "edge_lengths": f"Published DCRNN sensor-to-sensor road distances in meters ({distance_file})",
        "edge_speed": "Harmonic mean of endpoint sensor speeds; speeds are surrogates for the entire connection",
        "static_speed": "Per-sensor mean of valid training observations only",
        "replay": "Actual held-out five-minute sensor speeds, piecewise constant; integrate across interval boundaries; missing truth or horizon overflow discards the trip for all policies",
        "speed_floor_mph": args.speed_floor_mph,
        "departure_forecast_minutes": args.departure_minutes,
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "pairs": [[data.sensor_ids[u], data.sensor_ids[v]] for u, v in pairs],
        "departure_origin_indices": [int(origins[sample]) for sample in selected],
        "attempted_trips": len(pairs) * len(selected), "skipped_incomplete_trips": skipped,
        "complete_trips": summary["static"]["trips"], "summary": summary,
        "trips": records,
    }
    (output / "routing_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "routing_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["policy", *next(iter(summary.values()))])
        writer.writeheader()
        writer.writerows({"policy": name, **values} for name, values in summary.items())
    print(json.dumps(summary, indent=2))
    print(f"Scored {report['complete_trips']}/{report['attempted_trips']} common trips; saved simulation to {output}")
