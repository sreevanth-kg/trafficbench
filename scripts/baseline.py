#!/usr/bin/env python3
"""Install, run, and measure the real GraphHopper engine (Python stdlib only)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
VERSION = "11.1"
JAR_URL = (
    "https://repo.maven.apache.org/maven2/com/graphhopper/graphhopper-web/"
    f"{VERSION}/graphhopper-web-{VERSION}.jar"
)
MAP_URL = (
    "https://raw.githubusercontent.com/graphhopper/graphhopper/"
    f"{VERSION}/core/files/monaco.osm.gz"
)
JRE_VERSION = "21.0.12.1+1"
JRE_URL = (
    "https://github.com/adoptium/temurin21-binaries/releases/download/"
    "jdk-21.0.12.1%2B1/OpenJDK21U-jre_x64_linux_hotspot_21.0.12.1_1.tar.gz"
)
JRE_SHA256 = "2413149700df0f7d440500a84a8f764c535f21e5a5e87d38328b64eec2c5b500"
RUNTIME = ROOT / ".runtime"
MANIFEST = RUNTIME / "manifest.json"
CONFIG = ROOT / "configs/graphhopper.yml"
# CHStorage rounds shortcut weights to 0.001; several shortcuts can accumulate rounding.
# A 0.01 absolute tolerance is deliberately small for this short-route smoke check.
COST_ABS_TOLERANCE = 0.01
COST_REL_TOLERANCE = 1e-6


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, destination, expected=None, algorithm="sha256"):
    if destination.exists() and (expected is None or digest(destination, algorithm) == expected):
        print(f"Using {destination.relative_to(ROOT)}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}", flush=True)
    request = Request(url, headers={"User-Agent": "self-optim-baseline/0.1"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if expected is not None and digest(partial, algorithm) != expected:
            raise ValueError(f"Checksum mismatch for {url}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def setup(_args):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("This bootstrap downloads a Linux x86-64 runtime. Use Java 17+ on other platforms.")
    RUNTIME.mkdir(exist_ok=True)
    jar = RUNTIME / f"graphhopper-web-{VERSION}.jar"
    with urlopen(JAR_URL + ".sha1", timeout=30) as response:
        expected_jar = response.read().decode().strip().split()[0]
    download(JAR_URL, jar, expected_jar, "sha1")
    osm = ROOT / "data/monaco.osm.gz"
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    expected_map = previous.get("map", {}).get("sha256")
    download(MAP_URL, osm, expected_map)
    archive = RUNTIME / "temurin-jre.tar.gz"
    download(JRE_URL, archive, JRE_SHA256)
    with tarfile.open(archive) as package:
        top_level = package.getmembers()[0].name.split("/")[0]
        java = RUNTIME / top_level / "bin/java"
        if not java.exists():
            package.extractall(RUNTIME, filter="data")
    java_version = subprocess.run([str(java), "-version"], capture_output=True, text=True, check=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "GraphHopper",
        "engine_version": VERSION,
        "jar": {"path": str(jar.relative_to(ROOT)), "url": JAR_URL, "sha256": digest(jar)},
        "map": {"path": str(osm.relative_to(ROOT)), "url": MAP_URL, "sha256": digest(osm)},
        "java": {
            "path": str(java.relative_to(ROOT)),
            "version": JRE_VERSION,
            "url": JRE_URL,
            "archive_sha256": JRE_SHA256,
            "version_output": java_version.stderr.strip(),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: GraphHopper {VERSION}, local Java {JRE_VERSION}, Monaco OSM fixture.")


def serve(args):
    if not MANIFEST.exists():
        raise ValueError("Run `python3 scripts/baseline.py setup` first.")
    manifest = json.loads(MANIFEST.read_text())
    command = [
        str(ROOT / manifest["java"]["path"]),
        f"-Xmx{args.heap}",
        "-jar",
        str(ROOT / manifest["jar"]["path"]),
        "server",
        str(CONFIG),
    ]
    print("Starting GraphHopper at http://127.0.0.1:8989 (Ctrl+C to stop).", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def get_json(base_url, endpoint, parameters=None):
    url = base_url.rstrip("/") + endpoint
    if parameters:
        url += "?" + urlencode(parameters)
    request = Request(url, headers={"Accept": "application/json"})
    started = time.perf_counter_ns()
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return json.loads(body), elapsed
    except HTTPError as error:
        raise ValueError(f"GraphHopper HTTP {error.code}: {error.read().decode()}") from error
    except URLError as error:
        raise ValueError(f"Cannot reach {base_url}. Start the server with the `serve` command.") from error


def validate_queries(queries):
    if not isinstance(queries, list) or not queries:
        raise ValueError("Queries must be a non-empty JSON list.")
    seen = set()
    for query in queries:
        if not isinstance(query, dict) or not isinstance(query.get("id"), str) or not query["id"]:
            raise ValueError("Each query needs a non-empty string id.")
        if query["id"] in seen:
            raise ValueError(f"Duplicate query id: {query['id']}")
        seen.add(query["id"])
        for field in ("origin", "destination"):
            point = query.get(field)
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{query['id']}: {field} must be [latitude, longitude].")
            for value, bound in zip(point, (90, 180)):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > bound:
                    raise ValueError(f"{query['id']}: invalid coordinates in {field}.")


def route(base_url, query, mode):
    parameters = [
        ("point", ",".join(map(str, query["origin"]))),
        ("point", ",".join(map(str, query["destination"]))),
        ("profile", "car"),
        ("points_encoded", "false"),
        ("instructions", "false"),
        ("details", "osm_way_id"),
        ("details", "car_average_speed"),
    ]
    if mode == "flexible_dijkstra":
        parameters.extend([("ch.disable", "true"), ("lm.disable", "true"), ("algorithm", "dijkstrabi")])
    response, latency = get_json(base_url, "/route", parameters)
    if not response.get("paths"):
        raise ValueError(f"{query['id']}: route response contains no paths: {response}")
    path = response["paths"][0]
    result = {
        "query_id": query["id"],
        "method": "graphhopper_" + mode,
        "distance_m": path["distance"],
        "estimated_duration_s": path["time"] / 1000,
        "routing_cost": path["weight"],
        "http_round_trip_ms": latency,
        "geometry": path["points"],
        "snapped_waypoints": path.get("snapped_waypoints"),
        "path_details": path.get("details", {}),
    }
    return result, response


def percentile(samples, fraction):
    values = sorted(samples)
    index = (len(values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def benchmark(args):
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative.")
    queries_path = Path(args.queries)
    queries = json.loads(queries_path.read_text())
    validate_queries(queries)
    engine_info, _ = get_json(args.url, "/info")
    if engine_info.get("version") != VERSION:
        raise ValueError(f"Expected GraphHopper {VERSION}, server reports {engine_info.get('version')!r}.")
    modes = ("ch", "flexible_dijkstra")
    for _ in range(args.warmup):
        for query in queries:
            for mode in modes:
                route(args.url, query, mode)
    samples = {mode: [] for mode in modes}
    records = []
    raw_responses = []
    maximum_gap = 0.0
    matching_geometries = 0
    rng = random.Random(42)
    for repetition in range(args.repeats):
        ordering = list(queries)
        rng.shuffle(ordering)
        for query in ordering:
            mode_order = list(modes)
            rng.shuffle(mode_order)
            results = {}
            for mode in mode_order:
                result, response = route(args.url, query, mode)
                result["repetition"] = repetition
                samples[mode].append(result["http_round_trip_ms"])
                records.append(result)
                raw_responses.append({"query_id": query["id"], "mode": mode, "repetition": repetition, "response": response})
                results[mode] = result
            reference = results["flexible_dijkstra"]["routing_cost"]
            gap = abs(results["ch"]["routing_cost"] - reference)
            maximum_gap = max(maximum_gap, gap)
            matching_geometries += results["ch"]["geometry"] == results["flexible_dijkstra"]["geometry"]
            if not math.isclose(results["ch"]["routing_cost"], reference, rel_tol=COST_REL_TOLERANCE, abs_tol=COST_ABS_TOLERANCE):
                raise ValueError(
                    f"{query['id']}: CH cost {results['ch']['routing_cost']} disagrees with "
                    f"Dijkstra cost {reference}; absolute difference {gap}."
                )
    summary = {
        mode: {
            "requests": len(values),
            "http_round_trip_ms": {
                "mean": statistics.mean(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
            },
        }
        for mode, values in samples.items()
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Small real-engine correctness and latency smoke benchmark; not a city-scale evaluation.",
        "engine_info": engine_info,
        "artifacts": json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None,
        "config_sha256": digest(CONFIG),
        "queries_sha256": digest(queries_path),
        "queries": queries,
        "profile": "stock car.json, motorcar/motor_vehicle turn restrictions, 60-second U-turn penalty",
        "objective": "GraphHopper car custom weighting, including stock distance_influence=90 s/km; not pure shortest distance or pure travel time.",
        "traffic": "Static OSM-derived speed profile; no live traffic observations or trained ETA predictor.",
        "latency_scope": "Client HTTP round trip, including response transfer and connection setup; excludes client JSON decoding, import and CH preprocessing.",
        "warmup_rounds": args.warmup,
        "repeats": args.repeats,
        "seed": 42,
        "summary": summary,
        "cost_check_tolerance": {"absolute": COST_ABS_TOLERANCE, "relative": COST_REL_TOLERANCE},
        "max_absolute_ch_dijkstra_cost_difference": maximum_gap,
        "pairs_with_identical_route_geometry": matching_geometries,
        "verified_route_pairs": len(queries) * args.repeats,
        "routes": records,
        "raw_responses": raw_responses,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": record["geometry"],
                "properties": {key: record[key] for key in ("query_id", "method", "distance_m", "estimated_duration_s", "routing_cost")},
            }
            for record in records if record["repetition"] == 0
        ],
    }
    output.with_suffix(".geojson").write_text(json.dumps(geojson, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Verified CH against Dijkstra for {len(queries)} queries; maximum cost difference {maximum_gap:.8f}.")
    print(f"Saved {output} and {output.with_suffix('.geojson')}.")


def check(args):
    """Run the real-engine integration check and shut down our server afterward."""
    if not MANIFEST.exists():
        raise ValueError("Run `python3 scripts/baseline.py setup` first.")
    for port in (8989, 8990):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise ValueError(f"Port {port} is in use. Use `benchmark` against the running server or stop it first.") from error
    logfile = RUNTIME / "server-check.log"
    with logfile.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "serve", "--heap", args.heap],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError(f"Server exited with {process.returncode}. See {logfile}.")
                try:
                    with urlopen(args.url + "/info", timeout=1) as response:
                        json.load(response)
                    break
                except (OSError, ValueError):
                    time.sleep(0.5)
            else:
                raise ValueError(f"Server did not become ready. See {logfile}.")
            benchmark(args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Download pinned GraphHopper, Java, and a small OSM fixture.").set_defaults(run=setup)
    server = commands.add_parser("serve", help="Import OSM, prepare CH, and run the local server.")
    server.add_argument("--heap", default="1g")
    server.set_defaults(run=serve)
    for name, action in (("benchmark", benchmark), ("check", check)):
        bench = commands.add_parser(name, help="Measure the running engine." if name == "benchmark" else "Start, verify and stop the engine automatically.")
        if name == "benchmark":
            bench.add_argument("--url", default="http://127.0.0.1:8989")
        else:
            bench.set_defaults(url="http://127.0.0.1:8989")
            bench.add_argument("--heap", default="1g")
        bench.add_argument("--queries", default=str(ROOT / "examples/monaco_queries.json"))
        bench.add_argument("--output", default=str(ROOT / "results/graphhopper_baseline.json"))
        bench.add_argument("--warmup", type=int, default=2)
        bench.add_argument("--repeats", type=int, default=5)
        bench.set_defaults(run=action)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped.\n")


if __name__ == "__main__":
    main()
