# Scripts and reproducible commands

All commands use the existing project environment through `uv`; they do not
create a second virtual environment. Run them from the repository root.

## Setup and data

```bash
uv sync
uv run python scripts/download_metr_la.py --dataset metr-la
uv run python scripts/download_metr_la.py --dataset pems-bay
uv run python -m unittest discover -s tests -v
```

The downloader stores the pinned observations, graph, distance and location
artifacts under `data/`. Do not rerun it unnecessarily while resuming a study;
the manifest timestamp is part of the current study provenance.

## PEMS-BAY three-seed study

The exact completed comparison is Graph WaveNet versus RGL-Flow on seeds 7, 21
and 42, with six total 50-epoch fits:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_pems_bay_3seed.sh
```

The launcher requires CUDA and uses the existing `.venv` through `uv`. It writes
checkpoints, forecasts and aggregate reports to
`results/pems-bay/study-50ep-3seed/`. It is resumable after interruption:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_pems_bay_3seed.sh --status
```

For a different location:

```bash
TRAFFICBENCH_DATA=data/pems-bay \
TRAFFICBENCH_OUTPUT=results/pems-bay/my-study \
CUDA_VISIBLE_DEVICES=0 scripts/run_pems_bay_3seed.sh
```

The underlying general-purpose study command is:

```bash
uv run trafficbench study --data data/pems-bay \
  --output results/pems-bay/study-50ep-3seed \
  --epochs 50 --patience 10 --seeds 7 21 42 \
  --deployment-seed 42 --baselines graphwavenet \
  --candidates regimeliquid --device cuda --threads 4 \
  --pairs 20 --departures 24 --resume
```

## Forecasting and routing commands

```bash
uv run trafficbench forecast --data data/pems-bay \
  --output results/pems-bay/full --epochs 50 --patience 10 --device cuda

uv run trafficbench route --data data/pems-bay \
  --forecasts results/pems-bay/full --pairs 20 --departures 24
```

The `forecast` command supports `persistence`, `lstm` and `graphwavenet` via
`--models`. The `route` command replays selected routes on held-out observed
speeds using the abstract dataset sensor graph.

## Proposed-model search and product inference

```bash
uv run trafficbench discover --data data/pems-bay \
  --output results/pems-bay/proposed-10ep --epochs 10 --device cuda

uv run trafficbench evaluate-selected --data data/pems-bay \
  --experiment results/pems-bay/proposed-10ep \
  --baseline results/pems-bay/full --device cuda

uv run trafficbench predict --bundle results/pems-bay/proposed-10ep/bundle \
  --input results/pems-bay/proposed-10ep/bundle/example_request.json \
  --output results/pems-bay/prediction.json

uv run trafficbench serve-forecast \
  --bundle results/pems-bay/proposed-10ep/bundle
```

## Utility scripts

| Script | Purpose |
| --- | --- |
| `download_metr_la.py` | Download and checksum METR-LA or PEMS-BAY artifacts |
| `run_pems_bay_3seed.sh` | Launch or inspect the exact six-fit Bay comparison |
| `benchmark_inference.py` | Compare batch-one CPU model-forward latency |
| `check_forecast_service.py` | Check the local forecast HTTP API and bundle agreement |
| `summarize_proposed.py` | Summarize the single-seed proposed-model experiment |
| `baseline.py` | Set up/check/benchmark the independent GraphHopper baseline |
| `pause_study_boundary.py` | Pause an older study after a completed run boundary |
| `focus_study_comparison.py` | Preserve work while narrowing an unfinished study |

GraphHopper-specific setup and commands are documented in
`docs/graphhopper-baseline.md`.
