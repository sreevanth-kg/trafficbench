# Project review and PEMS-BAY three-seed run

## Confirmed experiment

The user confirmed **Graph WaveNet versus RGL-Flow (`regimeliquid`)** on
**PEMS-BAY**, using the existing `.venv` and the local RTX 4060 Laptop GPU.
Graph WaveNet is the implementation adapted from Zonghan Wu's research code;
Google's published ETA GNN is related work and is not implemented here.

**Completed:** all six 50-epoch fits, final evaluation and bundle export finished.
See [the three-seed results](pems-bay-3seed-results.md). Overall MAE improved
2.75% / 5.35% / 9.56% at 15/30/60 minutes in all three seeds. Routing gains
were inconclusive; transition-regime interval coverage remains poor.

| Setting | Value |
| --- | --- |
| Training seeds | 7, 21, 42 for both models |
| Required fits | 6 |
| Maximum epochs / early-stopping patience | 50 / 10 |
| Batch size / initial learning rate | 64 / 0.001 |
| Optimizer | Adam; weight decay 0.0001; gradient clipping 5 |
| Checkpoint selection | Lowest validation MAE over all 12 horizons |
| Deployment seed | 42, fixed before test evaluation |
| Device | Explicit CUDA, GPU 0 |
| Data | All 325 sensors; 36,435 train / 5,200 validation / 10,413 test windows |
| Output | `results/pems-bay/study-50ep-3seed/` |
| Log | `.runtime/pems-bay-study-3seed.log` |

Only these two architectures are trained. The study runner additionally computes
its built-in untrained persistence and static-routing references. Model training
and forward inference use CUDA; data loading, metric aggregation and NetworkX
routing use the CPU.

## Findings

### 1. Concurrent studies competed for the same GPU — resolved operationally

A separate METR-LA three-seed study was running concurrently with PEMS-BAY.
The first PEMS-BAY Graph WaveNet epochs took approximately 350 seconds each
while sharing the GPU. After the user approved stopping METR-LA, its process was
terminated and PEMS-BAY settled near 140 seconds per epoch. METR-LA's on-disk
epoch checkpoints remain available.

Training wall times for PEMS-BAY seed 7 include this contention, so these timings
are not a controlled model-speed comparison.

### 2. Cached downloads change dataset identity — reproducibility bug

`scripts/download_metr_la.py:76-88` always generates a new `downloaded_at_utc`
and writes the manifest, even when every artifact was already present. This was
observed for both datasets during setup. `trafficbench/study.py:206` compares the
entire manifest when resuming, so rerunning setup can prevent recovery despite
identical artifact checksums.

A correction should preserve the manifest on a verified cache hit, or compare a
canonical dataset identity including artifact hashes and interpretation metadata
without the download timestamp. The active study already records the current
manifest; no further downloader invocation is needed to continue it.

### 3. MSR inference bundle uses the wrong model class — optional variant bug

`trafficbench/inference.py:58-61` always instantiates `RegimeLiquidGraph`.
However, `trafficbench/regime.py:305-308` builds `RoutedMultiScaleLiquidGraph` for
`rglflow_msr`, which has different encoder parameters. Exported MSR checkpoints
therefore cannot load through the current inference engine. The engine should
use the same variant-aware factory as training.

The selected `regimeliquid` architecture uses the class currently constructed by
the engine; the MSR variant is outside this run.

### 4. MSR training performs two full forwards — optional variant inefficiency

`RoutedMultiScaleLiquidGraph.training_outputs` calls the base training method,
which already invokes `self.forward_details`, then invokes `forward_details`
again (`trafficbench/regime.py:291-301`). The second pass repeats all three CfC
encoders to compute its domain-router objective. Returning/reusing the first
pass's details would avoid the duplicate computation and dropout draws.

### 5. GPU utilization does not mean the entire pipeline is GPU-only

Validation copies predictions to CPU before reducing errors
(`trafficbench/experiment.py:45-54`), and training extracts scalar values every
batch (`trafficbench/experiment.py:101-108`). These transfers/synchronizations
are potential throughput costs. A future performance change could accumulate
statistics on-device, verified against the existing masked metrics.

### 6. Existing evidence has limits

The checked-in longer METR-LA study used two seeds, despite a `5seed` directory
name; the new Bay directory and plan explicitly specify three seeds. Existing
METR-LA results do not establish Bay accuracy. Several earlier fits reached the
50-epoch cap, so a matched epoch budget should not be described as guaranteed
convergence. Architecture/checkpoint selection in this run remains validation-only.

## Verification before and after full training

- Existing environment: Python 3.13, PyTorch `2.14.1+cu130`; CUDA available on
  NVIDIA GeForce RTX 4060 Laptop GPU.
- All 26 existing tests passed, including CPU/CUDA interrupted-training recovery.
- PEMS-BAY loaded with 52,116 timestamps and 325 sensors. The study checked actual
  artifact SHA-256 values against its manifest.
- A separate tiny two-seed CUDA smoke study completed all four fits, evaluation
  and routing at `/tmp/omnirush/pems-bay-smoke/`. These subset scores are pipeline
  checks, not the full experiment's accuracy results.
- All six full fits completed. Checkpoint/forecast hashes, forecast origins,
  recomputed metrics, aggregate means/SD and the seed-42 CUDA bundle prediction
  were verified after completion.

## Run and monitor using the existing environment

The full study was launched with the settings below. Resume only after the
current runner has exited; the study lock also prevents duplicate writers.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m trafficbench.cli study \
  --data data/pems-bay --output results/pems-bay/study-50ep-3seed \
  --epochs 50 --patience 10 --seeds 7 21 42 --deployment-seed 42 \
  --baselines graphwavenet --candidates regimeliquid --device cuda \
  --threads 4 --pairs 20 --departures 24 --resume
```

```bash
.venv/bin/python -m trafficbench.cli study \
  --output results/pems-bay/study-50ep-3seed --stage status
```

The runner writes epoch recovery states and progress JSON during training. After
all six fits, it automatically writes `forecast_summary.csv`,
`comparison_report.json`, `comparison.md`, per-seed forecasts, routing results,
and the seed-42 inference bundle. These final outputs are not available while
the six-run schedule is still training.
