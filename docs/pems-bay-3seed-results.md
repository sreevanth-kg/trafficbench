# PEMS-BAY: Graph WaveNet versus RGL-Flow, three seeds

Completed on **2026-10-09** using the existing `.venv`, PyTorch
`2.14.1+cu130`, and the **RTX 4060 Laptop GPU**. All six fits finished 50 epochs;
the runner completed test evaluation, paired routing replay and bundle export.
Graph WaveNet is the research baseline implemented in this repository, not
Google's production ETA model.

## Protocol

- Models: `graphwavenet` and `regimeliquid` (RGL-Flow), fixed before this run.
- Seeds: **7, 21, 42** for each model; independent models, not an ensemble.
- Full PEMS-BAY: 325 sensors, 52,116 timestamps; **36,435 / 5,200 / 10,413**
  train / validation / test windows.
- One-hour history and forecast; five-minute sampling; raw-time 70/10/20 split,
  training-only normalization, causal input filling and masked missing targets.
- Batch size 64, Adam at 0.001, weight decay 0.0001, gradient clipping 5,
  validation-based learning-rate reduction, 50-epoch cap and patience 10.
- Each checkpoint was selected by validation MAE over all 12 horizons.
- Training and model-forward inference used explicit CUDA. Data handling,
  statistical summaries and NetworkX routing used CPU.

## Test forecast accuracy

Values are **mean ± sample SD across three seeds**, in mph. Lower is better.

### All observed speeds

| Horizon | Graph WaveNet MAE | RGL-Flow MAE | MAE reduction | Graph WaveNet RMSE | RGL-Flow RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 min | 1.360 ± 0.003 | **1.323 ± 0.002** | **2.75%** | 2.858 ± 0.005 | **2.772 ± 0.003** |
| 30 min | 1.742 ± 0.003 | **1.649 ± 0.004** | **5.35%** | 3.871 ± 0.002 | **3.679 ± 0.007** |
| 60 min | 2.160 ± 0.001 | **1.954 ± 0.007** | **9.56%** | 4.850 ± 0.018 | **4.385 ± 0.014** |

Every horizon used the same **3,384,008 valid observations per seed** for both
models. RGL-Flow had lower MAE in **3/3 seeds** at each horizon.

### Congested speeds (observed target below 40 mph)

| Horizon | Graph WaveNet MAE | RGL-Flow MAE | MAE reduction | Graph WaveNet RMSE | RGL-Flow RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 min | 4.811 ± 0.015 | **4.609 ± 0.060** | **4.20%** | 7.880 ± 0.016 | **7.584 ± 0.079** |
| 30 min | 7.666 ± 0.110 | **6.972 ± 0.129** | **9.05%** | 11.901 ± 0.131 | **11.122 ± 0.139** |
| 60 min | 10.578 ± 0.207 | **9.008 ± 0.163** | **14.84%** | 15.537 ± 0.276 | **13.727 ± 0.201** |

Congested counts are **173,028 / 172,972 / 172,945** at 15/30/60 minutes.
RGL-Flow had lower congested MAE in **3/3 seeds** at every horizon.

### Paired uncertainty in the MAE gains

The runner resampled paired training seeds and common forecast-origin days
with 2,000 bootstrap repetitions. Positive differences favor RGL-Flow.

| Horizon | All-speed MAE gain, mph (95% interval) | Congested MAE gain, mph (95% interval) |
| --- | ---: | ---: |
| 15 min | 0.037 [0.030, 0.045] | 0.202 [0.119, 0.272] |
| 30 min | 0.093 [0.075, 0.111] | 0.694 [0.438, 0.898] |
| 60 min | 0.207 [0.166, 0.244] | 1.570 [1.160, 1.930] |

These intervals are conditional on this network/test period and only three
training seeds. All six fits reached the epoch cap; matched budgets do not
establish convergence or reproduce published-paper benchmark protocols exactly.

## Routing result

The same **479/480 common complete trips** were scored for all policies.

| Forecast source | Mean simulated trip time, seconds ± seed SD |
| --- | ---: |
| Graph WaveNet | 941.61 ± 0.10 |
| RGL-Flow | 940.47 ± 0.86 |

RGL-Flow's paired mean improvement was **1.139 seconds**, with a 95% seed/day
interval of **[-0.454, 4.228] seconds**. The paired percentage improvement was
**0.082%**, interval **[-0.053%, 0.304%]**. Both intervals include zero:
**a reliable routing advantage is not established**. This is abstract
sensor-graph simulation, not measured street-level trips.

## Issue revealed by the results: regime-dependent interval coverage

The model's uncertainty intervals target 80% coverage, using validation-only
calibration. Test coverage differs substantially across regimes:

| Regime | 15 min coverage across seeds | 30 min | 60 min |
| --- | ---: | ---: | ---: |
| Free flow | 82.65–83.14% | 82.98–83.66% | 83.15–84.06% |
| Congested | 89.59–89.95% | 81.22–82.59% | **73.32–76.33%** |
| Transition | **39.86–41.14%** | **48.54–49.44%** | **59.84–61.11%** |

Transition means a non-congested target with an absolute speed change of at
least 8 mph from the latest causally filled observation. Aggregate coverage
does not establish calibrated uncertainty for these difficult transitions.
Future calibration changes should be chosen on validation data rather than
tuned to these test outcomes.

## Size, runtime and verification

| Model | Parameters | Best epochs (seeds 7, 21, 42) | Full test inference time, seconds |
| --- | ---: | --- | --- |
| Graph WaveNet | 303,312 | 48, 47, 48 | 13.74, 13.21, 12.90 |
| RGL-Flow | 80,336 | 45, 50, 49 | 2.72, 2.62, 2.65 |

Total recorded training time was **7.41 hours**, including validation and
checkpointing. Seed-7 Graph WaveNet timing includes initial contention with a
separate METR-LA study, which was stopped with the user's approval. Training
wall times should not be treated as a controlled speed benchmark. Test
inference timing includes data transfer and the prediction loop.

- All **26 existing tests passed** before training.
- A separate tiny CUDA smoke study completed both architectures and routing.
- After completion, all six checkpoint and forecast hashes were verified;
  forecast origins, sensor ordering, metrics and aggregate means/SD were checked.
- The seed-42 exported bundle was loaded on CUDA, and its example prediction
  matched the independently saved batched test prediction within tolerance.
  Quantiles were ordered and nonnegative; expert weights summed to one.

## Artifacts and reproduction

Full artifacts: `results/pems-bay/study-50ep-3seed/`, including `plan.json`,
`comparison.md`, `comparison_report.json`, `forecast_summary.csv`,
`seeds/`, `evaluation/` and `bundle/`.

Training log: `.runtime/pems-bay-study-3seed.log`.
Code/operational review: [project-review-2026-10-09.md](project-review-2026-10-09.md).

The study can be inspected with the existing environment:

```bash
.venv/bin/python -m trafficbench.cli study \
  --output results/pems-bay/study-50ep-3seed --stage status
```
