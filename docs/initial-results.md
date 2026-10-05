# Initial METR-LA baseline results

Executed on 2026-10-04 using the full raw dataset and an NVIDIA RTX 4060 Laptop
GPU. Both neural models trained for **10 epochs**, selecting their best checkpoint
by validation MAE. Both selected epoch 10. These are fixed-budget baseline results,
not a claim of converged or exact published-paper performance.

```bash
uv run trafficbench forecast --output results/metr-la/baseline-10ep --epochs 10 --patience 5
uv run trafficbench route --forecasts results/metr-la/baseline-10ep --pairs 20 --departures 24
```

## Data and protocol

- 34,272 five-minute observations, 207 sensors, 2012-03-01 through 2012-06-27.
- 23,967 training windows, 3,416 validation windows, 6,844 test windows.
- All windows use one hour of history and forecast the next hour.
- 70/10/20 raw-time split with non-overlapping target periods.
- Input filling is causal; normalization and static/fallback means use training
  observations only. Non-positive/non-finite labels are excluded from metrics.
- Congestion: held-out observed speed below 40 mph, excluding missing readings.
- Per-horizon valid test counts: 1,245,040 / 1,245,022 / 1,244,986.
- Per-horizon congested counts: 153,722 / 153,716 / 153,708.

## All observed test speeds

Cells show **MAE / RMSE**, in mph. Lower is better.

| Model | 15 minutes | 30 minutes | 60 minutes |
| --- | ---: | ---: | ---: |
| Persistence | 3.499 / 6.418 | 4.206 / 8.052 | 5.380 / 10.344 |
| LSTM | 3.141 / 6.179 | 3.830 / 7.724 | 4.947 / 9.681 |
| Graph WaveNet | **2.922 / 5.568** | **3.458 / 6.792** | **4.253 / 8.370** |

## Congested test speeds

Cells show **MAE / RMSE**, in mph, using identical observed-target masks for all models.

| Model | 15 minutes | 30 minutes | 60 minutes |
| --- | ---: | ---: | ---: |
| Persistence | 7.337 / 11.503 | 9.673 / 14.843 | 13.422 / 19.317 |
| LSTM | 7.840 / 12.396 | 10.397 / 15.757 | 15.120 / 20.627 |
| Graph WaveNet | **6.782 / 10.800** | **8.754 / 13.656** | **11.655 / 17.280** |

The shared per-sensor LSTM improves overall error but does worse than persistence
on congested targets. Graph WaveNet improves both groups in this run. Congestion
and longer horizons remain useful targets for the next model experiment.

## Sensor-graph routing simulation

The graph has 207 sensors and 1,515 directed connections with published physical
road distances. Twenty fixed OD pairs and 24 forecast origins yield 480 attempted
trips. **336 common complete trips** were scored for every policy; 144 were
excluded for missing ground-truth speeds or replay-horizon overflow on at least
one policy's path. Departure is 15 minutes after the forecast origin.

| Routing speed source | Mean simulated travel time, seconds | Mean paired improvement vs. static | Faster than static |
| --- | ---: | ---: | ---: |
| Training-average static speeds | 791.66 | 0.00% | — |
| Persistence | 781.10 | 1.05% | 48.51% |
| LSTM | 786.48 | 0.49% | 56.55% |
| Graph WaveNet | **777.14** | **1.28%** | **60.42%** |

Mean paired improvement averages `(static_trip_time - policy_trip_time) /
static_trip_time` over trips. It is distinct from the percentage difference
between aggregate mean travel times.

These paths belong to an **abstract sensor graph**, not a complete legal
navigation network. Endpoint speeds approximate whole connections, with a common
5 mph floor. Replay integrates held-out, piecewise-constant speeds across interval
boundaries. Results are simulated, descriptive and conditional on the common
complete-trip subset; no statistical significance or measured-trip gain is claimed.

## Runtime and verification

- LSTM training including validation/checkpointing: approximately 132 seconds.
- Graph WaveNet training including validation/checkpointing: approximately 524 seconds.
- Full test inference loops: approximately 1.91 / 5.86 seconds for LSTM / Graph WaveNet.
- Seven regression checks passed: split separation/time gaps, causal input fill,
  missing/congestion masks, directed diffusion, adaptive-adjacency gradients,
  dynamic traversal units, and scoring chosen routes on observed speeds.

Detailed machine-readable results and checkpoints are in
`results/metr-la/baseline-10ep/`. Dataset revisions, file checksums, model-code
hashes, software versions, settings, epoch curves and forecast origin indices are
recorded there. The results directory is ignored by git because it contains
generated forecasts and model weights; this summary preserves the reference run.
