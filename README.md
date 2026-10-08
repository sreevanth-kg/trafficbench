# METR-LA and PEMS-BAY forecasting and routing benchmark

Compare three established forecasting baselines with an evaluated liquid-graph model
on both the 207-sensor METR-LA and 325-sensor PEMS-BAY benchmarks:

1. **Persistence:** repeat the latest available sensor speed at every horizon.
2. **LSTM:** a two-layer, 64-hidden-unit temporal model shared across sensors.
3. **Graph WaveNet:** gated dilated temporal convolutions, directed graph diffusion
   and learned adaptive adjacency, following the original IJCAI 2019 implementation.
4. **RGL-Flow:** a proposed CfC liquid/graph encoder with horizon-specific specialist
   decoders and uncertainty outputs, selected using validation-only ablations.

Report **MAE and RMSE at 15, 30 and 60 minutes**, for all observed speeds and
congested observations. Then compare routes selected with predicted speeds against
static training-average speeds in a NetworkX simulation.

## Setup

Linux x86-64, Python 3.12/3.13 and `uv`. The pinned Linux PyTorch build uses CUDA
13.0; `--device cpu` is also available. `uv.lock` pins the resolved environment.

```bash
uv sync
python3 scripts/download_metr_la.py --dataset metr-la
python3 scripts/download_metr_la.py --dataset pems-bay
uv run python -m unittest discover -s tests -v
```

The downloader retrieves raw HDF5 observations from a pinned public mirror and the
original DCRNN adjacency, sensor coordinates and road-distance artifacts. It
checks the observation file's published SHA-256 and records artifact provenance
in `data/metr-la/manifest.json` and `data/pems-bay/manifest.json`. These are the
207-sensor METR-LA and 325-sensor PEMS-BAY datasets, not synthetic substitutes or
pre-windowed mirrors with unknown splits. The PEMS-BAY observation mirror is pinned
by revision and SHA-256; its original DCRNN release is hosted on Google Drive,
while the DCRNN graph and distance artifacts are pinned to the source commit.

### PEMS-BAY run

The forecasting, proposed-model, study, inference, and abstract routing commands
are dataset-agnostic. Point them at the Bay directory and use a Bay-specific output
directory so artifacts cannot be mixed with METR-LA:

```bash
uv run trafficbench forecast --data data/pems-bay \
  --output results/pems-bay/full --epochs 50 --patience 10
uv run trafficbench route --data data/pems-bay \
  --forecasts results/pems-bay/full
```

PEMS-BAY uses the same 5-minute, 12-step history/12-step forecast, raw-time
70/10/20 split, training-only normalization, causal input filling, missing-label
masking, congestion threshold, and routing replay protocol. Results remain
dataset-specific; model weights and sensor ordering are never shared between the
two networks.

## Run

### Full-data baseline training

```bash
uv run trafficbench forecast --output results/metr-la/full --epochs 50 --patience 10
uv run trafficbench route --forecasts results/metr-la/full
```

Each neural model saves its checkpoint with the lowest validation MAE across all
12 horizons. Test results do not determine checkpoints or early stopping. Set
`--device cuda` to require the GPU rather than allowing automatic CPU fallback.

### Small end-to-end verification

```bash
uv run trafficbench forecast --output results/metr-la/smoke \
  --epochs 2 --max-train 1024 --max-val 256 --max-test 512
uv run trafficbench route --forecasts results/metr-la/smoke --pairs 10 --departures 12
```

Positive sample limits spread a subset across each split and mark the report
`smoke_subset`. These runs verify the experiment pipeline; they do not establish
converged baseline accuracy. Zero sample limits, the default, use all windows.

## Forecasting protocol

- Raw observations are sampled every **five minutes**, in **mph**.
- Input is the previous 12 readings (one hour); output is the next 12 readings.
- The baseline neural models receive normalized speed and time-of-day fraction. The LSTM
  processes each sensor independently with shared weights; Graph WaveNet also
  uses neighboring sensors and learned connectivity.
- Split the raw timeline **70% train / 10% validation / 20% test**. Every target
  in a window must belong to the same split. Input history can use observations
  already available before the forecast origin, including the previous split.
- Reject windows crossing timestamp gaps. Fit the global normalization and
  per-sensor fallback/static means on valid **training observations only**.
- Treat non-finite and non-positive readings as missing, following the standard
  zero-missing convention for this dataset. Forward-fill inputs causally, with
  training-only fallback means; never fill labels used for metrics.
- Use masked MAE for the point objective, Adam and gradient clipping. The proposed
  family adds recorded auxiliary objectives. Every model uses identical forecast
  origins, targets and evaluation masks.
- Congestion is predefined as **observed target speed below 40 mph**, excluding
  missing values. `--congestion-mph` changes this threshold before the experiment.
  The report includes valid/congested counts for every horizon.
- The proposed model keeps the same fixed 40 mph congestion and 8 mph change
  thresholds on both datasets; reports also include free-flow/congested/transition
  frequencies by split and horizon so cross-dataset regime imbalance is visible.
- The 15/30/60-minute metrics use forecast steps **3/6/12**, respectively.

Graph WaveNet retains the original default 32 residual/dilation channels,
256 skip channels, 512 end channels, four blocks with two temporal layers, and
two directed fixed diffusion supports plus adaptive adjacency. Its implementation
uses modern `Conv2d` operations and registered buffers; see
`THIRD_PARTY_NOTICES.md` for source and license.

Our stricter raw-time target separation and causal missing-input handling differ
from some published pre-windowed protocols. Results must be compared within this
experiment, not represented as exact reproductions of paper tables.

## Routing simulation

Use **NetworkX on an abstract sensor graph**, rather than inventing a mapping
from METR-LA sensors to the unrelated Monaco GraphHopper fixture.

- Directed connections are the dataset's positive physical distance-based
  adjacency entries with finite, positive published road distances.
- Edge lengths come from the dataset-specific DCRNN distance CSV in meters
  (`distances_la_2012.csv` for METR-LA or `distances_bay_2017.csv` for PEMS-BAY).
  Gaussian adjacency weights and learned adaptive adjacency are **not** used as
  road lengths.
- Edge speed is the harmonic mean of its endpoint sensor speeds. Static speeds
  are per-sensor means of valid training readings.
- Pick fixed reachable multi-edge OD pairs from topology only, using a fixed
  seed. Every policy uses the same OD pairs and selected test forecast origins.
- Default departure is **15 minutes after the forecast origin**. Choose a route
  once with Dijkstra using static speeds or the model's 15-minute speed snapshot.
  The route stays fixed during evaluation.
- Replay the chosen path under **held-out observed speeds**, piecewise constant
  over five-minute intervals. Integrate traversal across time-interval changes;
  do not score a route using its own forecast.
- All policies share a documented 5 mph simulation speed floor. Forecast metrics
  remain unclipped. Missing ground-truth speeds or travel beyond the available
  replay horizon discard a trip for **every** policy, preserving paired comparison.
- Report simulated mean travel time, paired percentage improvement versus static,
  fraction of faster trips, route-change fraction, and skipped/complete trip counts.

Example:

```bash
uv run trafficbench route --forecasts results/metr-la/full \
  --pairs 20 --departures 24 --departure-minutes 15
```

This graph is not a complete navigation network. A sensor connection can span
multiple road segments; access/turn rules are not reconstructed, and endpoint
speeds are approximations for the whole connection. Travel times are **simulated**,
not measured trips. The common complete-trip subset also depends on ground-truth
availability along the selected paths. These assumptions are included in reports.

## Results

[Longer-training, two-seed study](docs/multiseed-study.md) uses a 50-epoch cap and
validation early stopping. Its runner automatically writes aggregate forecasting/
routing results and paired seed/day intervals when all runs finish; use the study
status command below to inspect the current phase.

[Initial full-data, 10-epoch results](docs/initial-results.md) include forecasting,
congestion metrics and the paired routing simulation.

[Proposed-model results](docs/proposed-results.md): the selected model reduced
Graph WaveNet's test MAE by **4.18% / 7.44% / 14.28%** at 15/30/60 minutes in the
same 10-epoch comparison. The report includes uncertainty, validation ablations,
CPU latency and the paired routing outcome.

[MSR candidate experiment](docs/msr-results.md) records a separate multi-scale
domain-routed liquid graph architecture inspired by DD-RNO and related time-series
work. It improved Graph WaveNet but did not surpass the existing RGL-Flow in the
10-epoch gate, so it was not promoted to a 50-epoch run.

Each forecast run writes:

- `forecast_metrics.csv`: per-model MAE/RMSE and counts for all/congested periods.
- `forecast_report.json`: splits, settings, model sizes, epoch curves, selected
  checkpoint, training/inference times, software and dataset provenance.
- `lstm.pt`, `graphwavenet.pt`: best-validation checkpoints.
- `*_forecasts.npz`: predictions, forecast-origin indices and sensor ordering.

Routing adds `routing_summary.csv` and `routing_report.json`, including fixed
OD pairs, departure indices and per-trip paths/costs. Inference timing includes
the selected test prediction loop, with CPU persistence and the configured neural
device; it is not a GPU-versus-CPU kernel speed comparison.

## Proposed model and inference

The proposed **RGL-Flow** family adds an actual CfC liquid encoder, directed graph
context, specialist speed-regime decoders and ordered uncertainty outputs. Four
variants are selected on validation only, including GRU/global-head/no-prior
ablations. [Architecture, prior work, commands and inference API](docs/proposed-architecture.md).

```bash
uv run trafficbench forecast --output results/metr-la/baseline-10ep --epochs 10 --patience 5
uv run trafficbench discover --output results/metr-la/proposed-10ep --epochs 10
uv run trafficbench evaluate-selected --experiment results/metr-la/proposed-10ep \
  --baseline results/metr-la/baseline-10ep
uv run trafficbench route --forecasts results/metr-la/proposed-10ep/evaluation
```

Dataset, forecast origins, input information, targets, congestion threshold and
routing evaluation remain identical. Auxiliary training objectives and total
architecture-search cost are reported separately from the fixed per-model budget.

### Longer training and multiple seeds

```bash
uv run trafficbench study --output results/metr-la/study-50ep-5seed \
  --epochs 50 --patience 10 --seeds 7 21 --deployment-seed 21 \
  --baselines graphwavenet --candidates regimeliquid --device cuda --resume
uv run trafficbench study --output results/metr-la/study-50ep-5seed --stage status
```

The revised study retains seeds 7 and 21 only (the original directory name is
preserved), comparing **Graph WaveNet and regimeliquid only**. The four required
50-epoch fits and test comparison are complete. Earlier completed artifacts are
preserved. Epoch-level recovery is available with `--resume` after
an interruption. [Protocol, runtime and output details](docs/multiseed-study.md).

## Related work

- [DCRNN, ICLR 2018](https://arxiv.org/abs/1707.01926): dataset and directed sensor graph.
- [Graph WaveNet, IJCAI 2019](https://arxiv.org/abs/1906.00121): selected graph baseline;
  [original implementation](https://github.com/nnzhan/Graph-WaveNet).
- [ETA Prediction with Graph Neural Networks in Google Maps, CIKM 2021](https://arxiv.org/abs/2108.11482):
  production context and related work. This experiment does not rebuild Google's model.

The previously verified GraphHopper engine remains available as an independent
road-routing baseline: [commands and provenance](docs/graphhopper-baseline.md).
