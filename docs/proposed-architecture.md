# RGL-Flow: regime-guided liquid graph forecasting

**Status:** implemented and evaluated; see [measured results](proposed-results.md).
The working name does not establish research novelty, patentability or commercial
superiority.

## Design

```text
One hour of past speeds + time of day
                │
     Directed upstream/downstream context
     Learned low-rank graph context
     Speed changes + periodic time features
                │
          Shared CfC liquid encoder
                │
    Local + neighboring latent state fusion
                │
    Horizon-specific expert router
       ├── Free-flow decoder
       ├── Congestion decoder
       └── Transition decoder
                │
       Ordered 10/50/90% speed quantiles
                │
        Validation-calibrated intervals
```

The `regimeliquid` variant uses the official `ncps.torch.CfC` implementation of
closed-form continuous-time liquid-derived networks. A `regimegru` ablation uses
an ordinary GRU explicitly and is never labeled a liquid network.

The existing two input channels remain the complete input information. Speed
differences, neighboring histories and sinusoidal time features are derived from
these inputs. No weather, future traffic, extra calendar variables or labels are
added at inference.

### DD-RNO adaptation

DD-RNO partitions aerodynamic queries into physically distinct regimes. We adapt
the specialization/routing idea, rather than importing its airfoil weights or its
force-integration module. Our experts specialize in speed regimes:

- Free flow: target speed at least 40 mph with less than 8 mph change from the
  latest observed speed.
- Congestion: target speed below 40 mph.
- Transition: target speed at least 40 mph and change of at least 8 mph.

Training targets supervise the router; the forward path accepts history only.
The optional observed-regime prior uses current speed and the recent 15-minute
speed change. Its log-probability contribution decays with horizon:

`router_logits(h) = learned_logits(h) + exp(-h / 6) * log(observed_prior)`

Here `h` is the five-minute forecast step. This lets near-term expert selection
use current traffic conditions without forcing the same regime an hour later.

These are traffic-informed inductive biases, **not a conservation-law PINN**.
METR-LA speed readings alone do not supply the density/flow variables needed to
validate a traffic continuity equation.

### Quantile heads

Each expert predicts a positive median through a persistence-residual head,
a nonnegative lower bound below the median, and an upper bound above it. Mixture
weights sum to one, so their convex combination preserves ordering. Training uses
the benchmark's masked point MAE plus small pinball and regime-classification
auxiliary losses. These additional objectives are recorded in the experiment.

An 80% validation-calibrated interval is computed per horizon using CQR-style
nonconformity scores. Held-out empirical coverage and interval width are reported.
Temporal correlation and distribution shift mean no future coverage guarantee is
asserted. Router weights are expert-mixing probabilities, not calibrated event risks.

## Validation-only architecture selection

Four variants use the same training windows, validation windows, seed, batch size,
optimizer settings and epoch budget:

| Variant | Liquid encoder | Expert mixture | Observed-regime prior |
| --- | --- | --- | --- |
| `regimeliquid` | CfC | Three experts | Yes |
| `liquidgraph` | CfC | One global head | No |
| `regimegru` | GRU | Three experts | Yes |
| `regimeliquid_noprior` | CfC | Three experts | No |

The lowest masked validation MAE across all twelve horizons chooses the checkpoint
and architecture. Model-code hashes and checkpoint hashes freeze the choice before
new test-model forecasts are generated. Only the chosen architecture is evaluated
on the test split; ablation test scores are not used to select a winner.

These variants have different parameter counts, which are reported. Single-seed
ablations are diagnostic rather than proofs that one component causes a gain.

```bash
uv run trafficbench forecast --output results/metr-la/baseline-10ep --epochs 10 --patience 5
uv run trafficbench discover --output results/metr-la/proposed-10ep --epochs 10
uv run trafficbench evaluate-selected --experiment results/metr-la/proposed-10ep \
  --baseline results/metr-la/baseline-10ep
uv run trafficbench route --forecasts results/metr-la/proposed-10ep/evaluation
```

The combined routing report replays every baseline and the selected model on the
same complete-trip intersection. Its trip count can differ from the earlier
baseline-only experiment; compare methods within the new combined report.

## Prior work and novelty boundary

Reviewed on 2026-10-04:

- [DD-RNO](https://arxiv.org/abs/2608.13490), 2026: physics-guided domain routing
  for airfoil surrogates; inspiration for specialist decoder allocation.
- [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898),
  2021/2022: the liquid-derived CfC building block.
- [TESTAM](https://arxiv.org/abs/2403.02600), ICLR 2024: traffic expert mixtures
  with pseudo-label gate training already exist.
- [Graph-Conditioned Mixture of GNN Experts](https://arxiv.org/abs/2605.30486),
  2026: topology/input-conditioned combinations of frozen forecasting experts.
- [Memory-Augmented Graph Liquid Time-Constant Networks](https://arxiv.org/abs/2606.15807),
  2026: graph liquid dynamics for continuous cross-domain traffic prediction.

Therefore "liquid + graph + mixture of experts" is **not** a defensible novelty
claim by itself. The specific candidate contribution is the horizon-decaying
observed-regime prior combined with joint liquid graph encoding, traffic-regime
specialists and ordered probabilistic outputs. Its benefit must survive ablations;
global originality has not been established by this targeted literature search.
If an ablation wins validation, the exported product uses that winner rather than
retaining an unhelpful component just for its name.

## Product-facing bundle and demo

The selected model exports `model.pt`, `graph.npz`, `metadata.json` and a
history-only example request. Inference uses neither training nor test datasets.
Metadata includes canonical sensor IDs, normalization, training fallback means,
validation interval corrections and artifact hashes.

```bash
uv run trafficbench predict --bundle results/metr-la/proposed-10ep/bundle \
  --input results/metr-la/proposed-10ep/bundle/example_request.json \
  --output results/metr-la/proposed-10ep/example_prediction.json

uv run trafficbench serve-forecast --bundle results/metr-la/proposed-10ep/bundle
```

Endpoints: `GET /health` and `POST /forecast`. Request shape:

```json
{
  "sensor_ids": ["sensor-a", "sensor-b"],
  "timestamps": ["2012-06-01T08:00:00", "... 12 consecutive five-minute readings ..."],
  "speeds_mph": [[60.0, 35.0], ["... one row per timestamp ..."]]
}
```

Use the bundle's full 207-sensor list in real requests; the above illustrates the
schema only. Sensor columns can arrive in any order and are aligned by ID.
Non-positive/non-finite history values are causally filled. Optional
`previous_speeds_mph` supplies observations immediately before the history window;
otherwise leading missing readings use the stored training means. Timestamps
must use the same naive network-clock convention as training.

Responses include median speeds, lower/upper intervals, all twelve forecast
horizons, expert weights, input coverage and prediction latency. This evaluated
bundle belongs to one METR-LA network. A new customer network needs its own mapped
graph and historical sensor data, training/validation, and a customer pilot.

```bash
uv run python scripts/check_forecast_service.py
uv run python scripts/benchmark_inference.py
uv run python scripts/summarize_proposed.py
```

Commercial value should be tested as a traffic-forecasting/ETA integration for
dispatch software: latency, missed-delay reduction and operational savings on
customer data. The current sensor-graph routing simulation does not establish
those savings or replace street-level route optimization.
