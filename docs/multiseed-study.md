# Longer training and two-seed comparison

## Study started on 2026-10-04

The full-data study was launched at `results/metr-la/study-50ep-5seed/`. Its settings
and implementation checksums were recorded in `plan.json` before training.
The user subsequently reduced the study to **seeds 7 and 21 only**, comparing
**Graph WaveNet and regimeliquid only**. Completed runs and the active seed-21
regimeliquid fit were preserved. Additional training was cancelled when those
four required fits were complete. The original directory name is retained
for artifact continuity, and `plan.original.json` preserves the superseded plan.
The focused comparison is complete at `results/metr-la/study-50ep-5seed/comparison.md`.
The earlier 10-epoch results remain a separate preliminary experiment.

| Setting | Value |
| --- | --- |
| Maximum training epochs | 50 for every neural model |
| Early-stopping patience | 10 validation checks without improvement |
| Training seeds | 7, 21 |
| Batch size / initial learning rate | 64 / 0.001 |
| Optimizer | Adam; weight decay 0.0001; gradient norm clip 5 |
| Learning-rate schedule | Validation ReduceLROnPlateau; patience 3; factor 0.5 |
| Neural baseline | Graph WaveNet |
| Our model | regimeliquid: CfC/graph/experts with observed-regime prior |
| Required training runs | 2 models × 2 seeds = 4, all complete |
| Deployment seed | 21, chosen independently of test performance |

Persistence has no trainable parameters and is evaluated once. All models keep
the original one-hour history, one-hour forecast, raw-time split, causal input
filling, training-only normalization and identical forecast/congestion masks.
Auxiliary pinball/router losses remain part of regimeliquid.

Seeds 42, 87 and 123 are cancelled. The runner uses the single available RTX 4060
sequentially; current progress reflects the completed four-run comparison schedule.

## Selection and comparison

1. Each run selects its checkpoint using validation MAE over all twelve horizons.
2. The user fixed the architecture to **regimeliquid**. Only its seed-7/21 validation
   records are included in the frozen comparison choice; other completed variants
   are retained without affecting the focused comparison.
3. Test Graph WaveNet and regimeliquid at both seeds, without picking
   a favorable test seed or evaluating losing candidates to change the choice.
   Evaluation can use whatever runs are complete after architecture selection;
   paired comparisons use seed intersections and list their actual seeds.
4. Report **mean ± sample standard deviation** for MAE/RMSE at 15/30/60 minutes,
   separately for all observed targets and speeds below 40 mph.
5. Report seed-paired error differences, the number of seeds improved, and a
   bootstrap that resamples paired seeds and common forecast-origin days.
6. Replay routes for every tested seed on one common complete-trip intersection.
   Routing OD/departure selection uses fixed seed 42 independently of model seeds.
   Mean seed results and paired seed/day intervals are reported.

These are **individual model runs**, not averaged ensemble predictions. Matching
the epoch cap is a per-run budget comparison. Recorded total training time also
includes the preserved completed runs from the earlier broader plan. Best epochs,
epochs actually run and runs
hitting the cap are recorded to assess whether more training is still needed.

The public test period was already examined in the 10-epoch study. This strengthens
repeatability on that benchmark; it is not a new independent external holdout.

## Run and inspect

```bash
uv run trafficbench study --output results/metr-la/study-50ep-5seed \
  --epochs 50 --patience 10 --seeds 7 21 --deployment-seed 21 \
  --baselines graphwavenet --candidates regimeliquid --device cuda --resume
```

The original training log is preserved at `.runtime/study-50ep-5seed.log`.

Inspect its phase, completed runs, active model/seed and latest completed epoch:

```bash
uv run trafficbench study --output results/metr-la/study-50ep-5seed --stage status
```

If the process has stopped, repeat the original training command with `--resume`:

```bash
uv run trafficbench study --output results/metr-la/study-50ep-5seed \
  --epochs 50 --patience 10 --seeds 7 21 --deployment-seed 21 \
  --baselines graphwavenet --candidates regimeliquid --device cuda --resume
```

Resume restores the last completed epoch's model, optimizer, scheduler, best
checkpoint, shuffle-generator state and CPU/CUDA random states. A partially
completed epoch is repeated. Settings, source hashes and actual dataset artifact
checksums must match the recorded study. A file lock prevents two runners from
writing to the same study simultaneously.

After selection, a report can be refreshed from completed runs while training
continues (separate evaluation lock):

```bash
uv run trafficbench study --output results/metr-la/study-50ep-5seed \
  --epochs 50 --patience 10 --seeds 7 21 --deployment-seed 21 \
  --baselines graphwavenet --candidates regimeliquid --device cuda --stage evaluate
```

## Automatic outputs

- `plan.json`: predefined settings, dataset hashes, software and selection rule.
- `progress.json`: current process/phase and active training run.
- `seeds/seed-N/training_records.json`: completed model runs and learning curves.
- `seeds/seed-N/MODEL_progress.json`: most recent completed epoch and learning rate.
- `seeds/seed-N/MODEL_training_state.pt`: epoch-level recovery training state.
- `selection.json`: frozen architecture choice and all validation records.
- `evaluation/`: per-seed forecasts, metrics and common-trip routing records.
- `forecast_summary.csv`: aggregate mean/SD table.
- `comparison_report.json`: seed variability, paired intervals and training diagnostics.
- `comparison.md`: automatically generated human-readable final comparison.
- `bundle/`: chosen architecture at the predefined deployment seed, with validation
  interval corrections and history-only inference metadata.

After completion, the new bundle can be exercised with:

```bash
uv run python scripts/check_forecast_service.py \
  --bundle results/metr-la/study-50ep-5seed/bundle
uv run trafficbench predict --bundle results/metr-la/study-50ep-5seed/bundle \
  --input results/metr-la/study-50ep-5seed/bundle/example_request.json \
  --output results/metr-la/study-50ep-5seed/example_prediction.json
```

## Verification before starting the full job

- **23 unit tests passed**, including mean-over-seeds selection, sample SD,
  matching masks, and seed/day bootstrap weighting.
- Partial-seed reporting, frozen selection unaffected by later seed records, and
  cached evaluation were verified.
- A stochastic feedforward fixture recovered exactly after a simulated interruption
  on both **CPU and CUDA**, checking optimizer state, learning-rate schedule,
  shuffle order and dropout random state.
- A two-seed/two-epoch smoke study completed all 12 fits, frozen selection, test
  aggregation and paired routing. It is marked `smoke_subset`; its accuracy is
  not used as evidence of final model quality.
- Resuming that completed smoke study correctly skipped training.
- The final seed-21 inference bundle passed the real HTTP integration check,
  including agreement with saved batched test predictions.
