# RGL-Flow: measured METR-LA results

Executed on **2026-10-04**, using the full METR-LA dataset and the same protocol as
the [initial baselines](initial-results.md). All neural models used a maximum
**10 epochs**, seed **42**, batch size 64, Adam at 0.001, weight decay 0.0001 and
validation-selected checkpoints. Training used an RTX 4060 Laptop GPU.

**Finding:** the selected liquid/expert model improved all-observation MAE and
RMSE over Graph WaveNet at all three horizons, while using fewer parameters and
less CPU inference time. Routing gains were modest and statistically inconclusive.
This is a fixed-budget, single-network, single-seed result—not a converged or
state-of-the-art traffic benchmark claim.

## Validation selection and ablations

All candidates saw 23,967 training windows and 3,416 validation windows. Selection
used masked validation MAE over **all twelve horizons**, without generating new
test-model forecasts during search. The architecture and checkpoint were frozen
in `selection.json` before test evaluation. All candidates selected epoch 10.

| Candidate | Validation MAE, mph | Parameters | Training time, seconds |
| --- | ---: | ---: | ---: |
| **CfC + experts + prior (`regimeliquid`)** | **2.877905** | 78,448 | 133.09 |
| CfC + experts, no prior | 2.878736 | 78,448 | 131.59 |
| CfC + global head, no experts | 2.906846 | 54,004 | 118.85 |
| GRU + experts + prior | 2.916753 | 74,416 | 110.57 |

Graph WaveNet's validation MAE was **3.112312**, with 300,952 parameters and
523.71 seconds training time. Total candidate-search training time was **494.10
seconds**, not just the selected model's 133.09 seconds. Search overhead and
serialization are additional. The baselines did not receive an equivalent
four-architecture hyperparameter search.

Interpretation:

- The liquid/expert model won this validation comparison.
- The global-head and GRU ablations were approximately 1.0% and 1.3% worse in
  validation MAE, respectively. They also differ in parameter count.
- **The observed-regime prior improved validation MAE by only 0.000831 mph,
  approximately 0.029%.** This near-tie does not establish its usefulness.
- One seed cannot establish that any component's benefit is robust to
  initialization. No losing ablation was evaluated on test to choose a winner.

## Held-out forecasting

All 6,844 test windows were evaluated. Cells show **MAE / RMSE**, in mph.

### All observed speeds

| Model | 15 minutes | 30 minutes | 60 minutes |
| --- | ---: | ---: | ---: |
| Persistence | 3.499 / 6.418 | 4.206 / 8.052 | 5.380 / 10.344 |
| LSTM | 3.141 / 6.179 | 3.830 / 7.724 | 4.947 / 9.681 |
| Graph WaveNet | 2.922 / 5.568 | 3.458 / 6.792 | 4.253 / 8.370 |
| **RGL-Flow (`regimeliquid`)** | **2.800 / 5.356** | **3.201 / 6.376** | **3.645 / 7.358** |

Relative to Graph WaveNet, MAE decreased **4.18% / 7.44% / 14.28%** and RMSE
decreased **3.80% / 6.12% / 12.08%**. Valid observation counts were identical:
1,245,040 / 1,245,022 / 1,244,986 at the respective horizons.

### Congested speeds

Congestion is held-out observed speed below 40 mph, excluding missing readings.
All models share the same masks: 153,722 / 153,716 / 153,708 observations.

| Model | 15 minutes | 30 minutes | 60 minutes |
| --- | ---: | ---: | ---: |
| Persistence | 7.337 / 11.503 | 9.673 / 14.843 | 13.422 / 19.317 |
| LSTM | 7.840 / 12.396 | 10.397 / 15.757 | 15.120 / 20.627 |
| Graph WaveNet | 6.782 / 10.800 | 8.754 / 13.656 | 11.655 / 17.280 |
| **RGL-Flow** | **6.703 / 10.616** | **8.646 / 13.468** | **10.801 / 16.142** |

Congested MAE decreased **1.17% / 1.24% / 7.33%**. Short-horizon congested
improvements are small; the stronger result is the 60-minute horizon.

### Paired day-block uncertainty

Resampled the 24 forecast-origin days with 1,000 bootstrap draws, keeping each
day's sensors and overlapping windows together. Positive differences mean lower
MAE for RGL-Flow. These intervals are conditional on the fitted seed-42 models
and this network/test period; they do not measure training-seed variability.

| Horizon | Group | MAE improvement, mph | 95% day-block interval, mph |
| --- | --- | ---: | ---: |
| 15 min | All | 0.122 | [0.103, 0.139] |
| 30 min | All | 0.257 | [0.215, 0.294] |
| 60 min | All | 0.607 | [0.506, 0.699] |
| 15 min | Congested | 0.079 | [-0.017, 0.165] |
| 30 min | Congested | 0.108 | [-0.100, 0.290] |
| 60 min | Congested | 0.854 | [0.412, 1.256] |

Overall gains and the 60-minute congestion gain have positive intervals.
The short-horizon congestion intervals include zero.

## Uncertainty outputs

The model emits ordered 10/50/90% estimates. Validation-based inflation targets
80% interval coverage, but all twelve correction values were **zero**: the raw
learned intervals already exceeded the required validation coverage. Calibration
does not shrink overly wide intervals.

| Horizon | Test empirical coverage | Mean interval width, mph |
| --- | ---: | ---: |
| 15 min | 88.55% | 11.46 |
| 30 min | 87.83% | 12.03 |
| 60 min | 86.95% | 12.92 |

These are conservative empirical intervals, not precisely calibrated 80%
probabilities. Neither temporal dependence nor cross-network shift is covered by
an asserted distribution-free guarantee. Expert weights are routing weights,
not calibrated congestion-risk probabilities.

## Paired sensor-graph routing

Replayed all five policies on the same **336/480** common complete trips using
20 topology-selected OD pairs, 24 forecast origins, a 15-minute departure lead
and the shared 5 mph simulation floor. The graph has 207 nodes and 1,515 directed
physical-distance connections. The original baseline values are retained because
the common-trip intersection happened to remain unchanged.

| Speed source | Mean simulated trip time, seconds | Mean paired improvement vs static |
| --- | ---: | ---: |
| Static training means | 791.66 | 0.00% |
| Persistence | 781.10 | 1.05% |
| LSTM | 786.48 | 0.49% |
| Graph WaveNet | 777.14 | 1.28% |
| RGL-Flow | **772.29** | **1.76%** |

Against Graph WaveNet, the mean time reduction was **4.85 seconds**; the mean
per-trip percentage reduction was **0.435%**. A separate departure-day bootstrap
over the 23 days with complete trips gave a 95% interval of **[-1.76, 11.51]
seconds**, or **[-0.094%, 1.027%]** for paired percentages. Both include zero.
Thus a reliable routing gain is **not established** by this small simulation.

This is an abstract sensor graph and observed-speed replay, not legal street-level
routes or customer trip measurements. The estimates depend on complete truth along
every policy's path; 144 attempted trips were excluded for at least one policy.

## CPU deployment and verification

Same-machine CPU model-forward comparison: four PyTorch threads, batch one,
207 sensors, all twelve horizons, ten warmups and 100 timed samples. HTTP, JSON,
preprocessing and startup are excluded; the proposed forward also computes its
quantile heads and gates.

| Model | Parameters | Median forward time | P95 forward time |
| --- | ---: | ---: | ---: |
| Graph WaveNet | 300,952 | 17.55 ms | 22.94 ms |
| RGL-Flow | 78,448 | **4.18 ms** | **6.19 ms** |

RGL-Flow used **73.9% fewer parameters** and approximately **4.19× less median
CPU forward time** in this measurement. The chosen checkpoint's training time
was approximately 3.94× shorter, though the complete four-candidate search cost
was comparable to one Graph WaveNet training run.

- Sixteen unit checks passed: leakage/masks, graph gradients, dynamic route replay,
  actual CfC behavior, ordered quantiles, history-only inference, calibration and
  product-input boundaries.
- A real local HTTP check passed for `/health`, `/forecast`, normalized gates,
  ordered intervals, agreement with saved evaluation forecasts, and invalid-network
  rejection. The temporary server was shut down.
- CLI prediction succeeded without loading historical training/test data.
- Fixed model-code, model-weight and graph checksums guard bundle compatibility.

## Deliverables and commercial interpretation

The deployable artifact is `results/metr-la/proposed-10ep/bundle/`; use the
[architecture/API instructions](proposed-architecture.md) to run it. Machine-readable
reports, checkpoints, predictions and `comparison_summary.json` are under
`results/metr-la/proposed-10ep/`. This document preserves the reference findings
because generated results are ignored by git.

The strongest current product story is a **small, low-latency traffic-forecasting
component with improved longer-horizon errors and conservative uncertainty**.
The study does not establish superior production ETA, cross-city transfer,
customer savings or willingness to pay. A customer-network pilot and repeated
matched-budget training seeds are the next evidence needed for commercial claims.

Existing graph-liquid and traffic-expert work prevents claiming "liquid + graph +
experts" as globally novel. The regime prior is a candidate distinguishing mechanism,
but its near-zero ablation gain means it is not currently a strong research moat.
Measured integration value is a more defensible basis for a pilot than claiming
a unique architecture.

### Reproduce

```bash
uv sync
python3 scripts/download_metr_la.py
uv run trafficbench forecast --output results/metr-la/baseline-10ep --epochs 10 --patience 5
uv run trafficbench discover --output results/metr-la/proposed-10ep --epochs 10 --patience 5
uv run trafficbench evaluate-selected --experiment results/metr-la/proposed-10ep --baseline results/metr-la/baseline-10ep
uv run trafficbench route --forecasts results/metr-la/proposed-10ep/evaluation --pairs 20 --departures 24
uv run python scripts/check_forecast_service.py
uv run python scripts/benchmark_inference.py
uv run python scripts/summarize_proposed.py
```
