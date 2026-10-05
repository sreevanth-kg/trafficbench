# Existing GraphHopper routing baseline

The first experiment runs **GraphHopper 11.1**, an existing routing engine, on a
real OpenStreetMap road network. GraphHopper's open-source engine underpins its
commercial Directions API and route planning on openstreetmap.org, as documented
on its [open-source page](https://www.graphhopper.com/open-source/).

## What is used in deployed systems?

| Component | Documented deployed system | What we can reproduce |
| --- | --- | --- |
| Road routing | GraphHopper's commercial Directions API and OpenStreetMap route planning | The released GraphHopper engine, its car profile, and Contraction Hierarchies (CH) |

The local GraphHopper baseline is an actual released engine. Its profile and OSM
fixture are reproducible, but are not claimed to be the complete configuration or
traffic data of any commercial service.

## Run the existing engine

Requires Python 3.12+ on Linux x86-64, an internet connection for the initial
downloads (about 100 MB), and approximately 1 GB of available RAM for this small
fixture. The script installs Java locally in `.runtime/`; Docker is unnecessary.

```bash
python3 scripts/baseline.py setup
python3 scripts/baseline.py check
```

`check` starts the engine, runs the real-engine comparison and shuts down its
server automatically. To leave the server running for interactive requests:

```bash
python3 scripts/baseline.py serve
```

The first server start imports Monaco's OSM fixture and prepares CH. Later starts
reuse the graph. The service binds to `http://127.0.0.1:8989`.

In a second terminal:

```bash
python3 scripts/baseline.py benchmark
```

This sends six fixed origin/destination queries to **both CH and flexible
bidirectional Dijkstra inside GraphHopper**, warms up both modes, then records
five repetitions per query/mode in a deterministic randomized order. It verifies
that the routing costs agree within a 0.01 absolute / 1e-6 relative tolerance.
GraphHopper's [CHStorage](https://github.com/graphhopper/graphhopper/blob/11.1/core/src/main/java/com/graphhopper/storage/CHStorage.java)
rounds shortcut weights to three decimals, so identical routes can have slightly
different reported weights. The report also counts identical route geometries.
This is a correctness
and latency smoke benchmark, not evidence of city-scale performance or predictive
accuracy.

Outputs:

- `results/graphhopper_baseline.json`: routes, raw responses, latency summaries,
  engine information, artifact hashes, configuration hash and fixed queries.
- `results/graphhopper_baseline.geojson`: route geometries for visualization.
- `.runtime/manifest.json`: exact engine, Java and OSM artifact provenance.

Adjust sample count:

```bash
python3 scripts/baseline.py benchmark --warmup 5 --repeats 20
```

Coordinates in the query file are `[latitude, longitude]`. Output GeoJSON uses
`[longitude, latitude]`. The engine snaps query coordinates onto accessible roads;
the snapped waypoints are retained in the report. Landmark names in query IDs are
descriptive, not geocoded addresses.

## Baseline configuration and objective

`configs/graphhopper.yml` enables:

- The bundled `car.json` profile, respecting motorcar access and one-way streets.
- Motorcar/motor-vehicle turn restrictions and a 60-second U-turn penalty.
- CH preprocessing for fast queries, with flexible routing available for checks.
- OSM way IDs and speed details in route output for later analysis.

The stock car profile has `distance_influence=90` seconds per kilometer. Its
routing objective includes this distance preference plus travel time and turn
costs. **The returned routing cost, estimated duration, and distance are distinct
quantities.** Neither CH nor Dijkstra is optimizing live traffic in this setup.
Changing the profile requires rebuilding the graph cache and rerunning the
baseline; changing the OSM input also requires a separate/rebuilt cache.

The map is the small fixture shipped with GraphHopper 11.1. It is real OSM data,
but not a current city extract. OSM data attribution: © OpenStreetMap contributors,
[ODbL](https://www.openstreetmap.org/copyright). GraphHopper is Apache-2.0;
downloaded runtime artifacts retain their original licenses.

## How our later method will be compared

### If we improve route search

Use the same directed road graph, access/turn restrictions, snapped endpoints,
profile objective and query set. Compare:

1. Routing cost against the existing engine/exact reference.
2. Query latency on the same hardware, including any neural inference.
3. Preprocessing/training time and memory separately from query latency.
4. Route feasibility, including one-way and turn restrictions.

CH already preserves the optimum for a fixed graph and weighting. A neural search
method can target speed, adaptability or memory, with its approximation gap made
explicit; it cannot improve the same exact objective below its true optimum.

### If we improve traffic prediction

The active forecasting benchmark uses METR-LA with persistence, LSTM and Graph
WaveNet. See the root README. Google Maps' published GNN is related work only.

First measure ETA/speed MAE, including congested periods. Then hold the route
selection procedure and candidate routes fixed while replacing the predictor, and
measure realized travel-time cost on held-out ground truth. GraphHopper's static
speed estimates are an additional reference, not an adequate sole competitor for
a new traffic predictor.

For predicted weights, prepared CH data must correspond to those weights; reusing
static CH shortcuts with changed edge costs would invalidate the comparison. A
first traffic experiment can rerank a fixed candidate-route set or use a router
that supports the weight updates, with both predictors using the same procedure.

### Data and extension boundary

The shared route record stores `query_id`, `method`, `distance_m`,
`estimated_duration_s`, `routing_cost`, `http_round_trip_ms`, `geometry`,
`snapped_waypoints` and `path_details`. Our later method should produce compatible
records on the same queries. Estimated times from a router are predictions, not
measured trip durations or training ground truth.

## Sources checked on 2026-10-04

- [GraphHopper production use](https://www.graphhopper.com/open-source/).
- [GraphHopper 11.x implementation and algorithm overview](https://github.com/graphhopper/graphhopper/tree/11.x).
- [Stock car profile](https://github.com/graphhopper/graphhopper/blob/11.1/core/src/main/resources/com/graphhopper/custom_models/car.json).
- [Google Maps deployed GNN paper](https://arxiv.org/abs/2108.11482).
- [Google DeepMind's deployment explanation](https://deepmind.google/blog/traffic-prediction-with-advanced-graph-neural-networks/).
- [Current Google Routes traffic options](https://developers.google.com/maps/documentation/routes/config_trade_offs).
- [OSRM implementation](https://github.com/Project-OSRM/osrm-backend): another
  mature engine, documenting CH and Multi-Level Dijkstra. GraphHopper is the
  selected executable baseline here.
