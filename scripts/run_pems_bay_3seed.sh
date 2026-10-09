#!/usr/bin/env bash
set -euo pipefail

# Reproduce the completed PEMS-BAY Graph WaveNet versus RGL-Flow study using
# the project's existing uv environment. The explicit CUDA device prevents an
# accidental CPU fallback. Set TRAFFICBENCH_DATA or TRAFFICBENCH_OUTPUT to
# use alternate locations.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${TRAFFICBENCH_DATA:-data/pems-bay}"
OUTPUT="${TRAFFICBENCH_OUTPUT:-results/pems-bay/study-50ep-3seed}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${1:-}" == "--status" ]]; then
  exec uv run trafficbench study \
    --data "$DATA" \
    --output "$OUTPUT" \
    --stage status
fi

exec env CUDA_VISIBLE_DEVICES="$GPU" uv run trafficbench study \
  --data "$DATA" \
  --output "$OUTPUT" \
  --epochs 50 \
  --patience 10 \
  --seeds 7 21 42 \
  --deployment-seed 42 \
  --baselines graphwavenet \
  --candidates regimeliquid \
  --device cuda \
  --threads 4 \
  --pairs 20 \
  --departures 24 \
  --resume
