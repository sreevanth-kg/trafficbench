# Multi-scale routed liquid graph candidate

## Motivation

The candidate `rglflow_msr` was added as a separate architecture after reviewing:

- [DD-RNO: A Domain-Decomposed Routed Neural Operator for Airfoil Flow Prediction](https://arxiv.org/abs/2608.13490), especially its differentiable routing of distinct physical regions to specialized subnetworks.
- [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186), for multi-scale temporal variation modeling.
- [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/abs/2005.11650), for learned directed graph dependencies and mix-hop context.
- [PatchTST: A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730), for preserving local subseries structure.

These references motivate design components; they do not establish that the
resulting combination is globally novel. The candidate should be described as a
project-specific architecture proposal until a broader literature review and
stronger ablations are completed.

## Architecture

`RGL-Flow/MSR` keeps the benchmark input boundary unchanged: only the previous
12 speed readings and time-of-day feature are used. It creates three causal
representation paths:

1. **Smooth path:** causal four-step trend, directed neighbors of that trend and
   periodic time features.
2. **Shock path:** first and second speed differences, directed neighbors of the
   changes and the current speed.
3. **Propagation path:** local speed, forward/reverse/adaptive graph context,
   directional imbalance and the local change.

Each path has its own CfC liquid encoder. A differentiable three-way router then
blends the latent states using learned evidence plus history-only soft signals
for smoothness, shock intensity and directional imbalance. The fused state is
sent through the existing graph context and ordered quantile/regime decoder.
This puts DD-RNO-style domain routing **before** decoding, whereas the original
`regimeliquid` candidate routes only among output experts after one shared
liquid encoder.

The route auxiliary objective is history-only cross entropy against the largest
of the three input-derived domain signals. It does not use future target values.
The existing regime-router auxiliary objective remains unchanged.

## 10-epoch result

Command used for the full 207-sensor dataset:

```bash
uv run trafficbench discover --output results/metr-la/msr-10ep-v2 \
  --candidates rglflow_msr --epochs 10 --patience 5 --device cuda --seed 42
uv run trafficbench evaluate-selected \
  --experiment results/metr-la/msr-10ep-v2 \
  --baseline results/metr-la/baseline-10ep --device cuda
```

The candidate used all 23,967 training windows and 3,416 validation windows.
Its best validation MAE was **2.9025 mph** at epoch 10, with 138,035
parameters. The existing 10-epoch `regimeliquid` model had validation MAE
2.8779 mph and 78,448 parameters.

### All observed test speeds

| Model | 15 min MAE/RMSE | 30 min MAE/RMSE | 60 min MAE/RMSE |
| --- | ---: | ---: | ---: |
| Graph WaveNet | 2.922 / 5.568 | 3.458 / 6.792 | 4.253 / 8.370 |
| Existing RGL-Flow | **2.800 / 5.356** | **3.201 / 6.376** | **3.645 / 7.358** |
| RGL-Flow/MSR | 2.809 / 5.380 | 3.216 / 6.405 | 3.662 / 7.383 |

The new candidate improved Graph WaveNet, but was slightly worse than the
existing RGL-Flow by 0.009, 0.015 and 0.017 mph MAE at the three horizons.

### Congested test speeds

| Model | 15 min MAE/RMSE | 30 min MAE/RMSE | 60 min MAE/RMSE |
| --- | ---: | ---: | ---: |
| Graph WaveNet | 6.782 / 10.800 | 8.754 / 13.656 | 11.655 / 17.280 |
| Existing RGL-Flow | **6.703 / 10.616** | **8.646 / 13.468** | **10.801 / 16.142** |
| RGL-Flow/MSR | 6.788 / 10.786 | 8.764 / 13.641 | 10.994 / 16.358 |

The candidate was slightly better than Graph WaveNet on 60-minute congested
MAE, but it did not beat the existing model. Its uncertainty intervals had
86.65%, 86.30% and 85.35% empirical coverage at 15/30/60 minutes.

## Decision

The 50-epoch extension was intentionally **not run**. The predefined promotion
criterion was improvement over the current 10-epoch RGL-Flow result, and the
candidate did not meet it. Existing 10-epoch and 50-epoch results remain the
project's reference results; the new checkpoint and reports are retained for
ablation and future refinement under `results/metr-la/msr-10ep-v2/`. The first
identical run is retained under `results/metr-la/msr-10ep/`.
