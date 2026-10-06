# Mechanism access, estimation and misspecification (Week 6)

The ladder asks how world-grounded success and replay error change as an explainer learns more
about the mechanism. The same action-recourse optimizer runs at every rung where it can; only the
information changes. The context builder (`explainers/mechanism_access/regimes.py`) enforces what
each regime may see, so information limits hold by construction, not by convention.

## What each regime sees

| Regime | Context fields | Mechanism the explainer plans with |
|---|---|---|
| R0 | classifier, factual units | none |
| R1 | + training trajectories and labels | none (retrieval / generative methods) |
| R2 | + dependency graph `G` on observed channels, outcome channels | none — actions are *imposed* |
| R3 | + `G` and a graph-masked fit `F̂` (family known, parameters unknown) | `estimator: masked` |
| R4 | an estimate `(Ĝ, F̂)` only | `estimator: lasso` / `neural`, or `misspecify: {kind, level}` |
| R5 | oracle world `F*` | oracle |
| R6 | oracle + action model (actionable set, bounds, controls) + feasibility constraints | oracle |

The runner builds R3/R4 mechanisms from the training split, once per configuration
(`ExplainerConfig.mechanism`). It scores each one against the true world on held-out units; the
explainer never sees these scores.

## Estimators (`twist.mechanisms.estimators`)

| Estimator | Regime | Model | Graph estimate |
|---|---|---|---|
| `masked` | R3 | least squares on the known parents (linear families); graph-masked neural transition otherwise | the given `G` |
| `lasso` | R4 | per-target BIC-selected Lasso on lagged values; coefficients below 0.01 are zeroed | support of the coefficients |
| `neural` | R4 | linear skip + one-hidden-layer tanh MLP on the lag window, trained by MSE | dense (no structure) |

Estimates expose the ordinary world interface. Abduction, Pearl replay and the differentiable
recourse optimizer run on them unchanged. On partially observed worlds they are fit on the
observations, which is all a real analyst has.

## Mechanism quality (`twist.mechanisms.quality`)

| Metric | Definition |
|---|---|
| graph precision / recall / F1 / SHD | summary graph (any lag), self-loops excluded, on observed channels |
| one-step error | mean \|x_t − F̂(lags)\| on held-out units (the oracle's value is the noise floor) |
| rollout error | noiseless rollouts from a mid-horizon prefix under F̂ vs F*, per channel std |
| **replay error** | Pearl replay of random point interventions under F̂ (abducted with F̂) vs the oracle replay of the same action, per channel std — the counterfactual accuracy that matters for explanations |

## Controlled misspecification (`twist.mechanisms.misspecification`)

`misspecify(world, kind, level)` gives the explainer a known-wrong mechanism. Level 0 reproduces
the world exactly (tested), so degradation can be read against an exact anchor.

| kind | error introduced |
|---|---|
| `coefficients` | non-zero parameters perturbed by `level` × their spread (graph kept) |
| `delete_edges` | fraction `level` of cross-channel edges removed (linear families) |
| `spurious_edges` | `level` × #edges spurious edges added (linear families) |
| `nonlinearity` | output saturation `c·tanh(F/c)`, `c` = typical \|F\| / level |
| `delay` | `(1 − level)·F(x_{t−1..}) + level·F(x_{t−2..})` |
| `noise` | heteroscedasticity misjudged (`h → h + level`); changes abduction |
| `observation` | partially observed world: the state dynamics applied to observations as if they were the state |

Replay error is not always monotone in `level`. Heavily perturbed tanh units saturate, so a very
wrong mechanism can respond *less* to an intervention than a moderately wrong one.

## Action-native recourse (`twist.explainers.action_native`)

| Explainer | Regime | Acts on | Notes |
|---|---|---|---|
| `GraphActionRecourse` | R2 | ancestors of the outcome (∩ actionable, if known) | no mechanism: the declared action is imposed and nothing propagates in its own proposal |
| `EstimatedActionRecourse` | R3/R4 (also R5) | all observed channels | plans with `F̂`; actions in observed coordinates are mapped to state channels by the evaluator |
| `OracleActionRecourse` (control) | R6 | the world's controls (or actionable channels) | projects onto amplitude bounds, max duration and lead time; drops immutable channels; prefers candidates that pass all constraints |

All three share the optimizer of the Pearl control: Adam on `CE(h(x̂), y*) + λ‖θ‖²` over candidate
start times, with point, interval, pulse or control actions.

## Mechanism record and failure attribution

Every record carries the mechanism the explainer used (`mechanism`), plus its benchmark-side
replay error and graph F1. Every unit gets an `attribution` (`evaluation/protocol.py`):

| Category | Meaning |
|---|---|
| `success` | the replayed action reaches the world target (C) |
| `outcome_disagreement` | the replay keeps the classifier's decision, but the world label differs (B holds, C fails) |
| `replay_mismatch` | the explainer reached the target under its own model, but the true replay loses the decision (A holds, B fails): mechanism estimation error for estimated mechanisms, missing action grounding for trajectory readings |
| `optimization` | the target was not reached under the explainer's own model (A fails) |
| `no_action` / `invalid_action` / `no_proposal` | nothing scorable |

## Running the ladder

```bash
twist ladder --config experiments/access_ladder.yaml          # or on SLURM:
sbatch scripts/slurm/multiworld.sbatch experiments/access_ladder.yaml ladder
```

The report (`artifacts/ladders/<name>/report/`) has the ladder table, the misspecification sweep,
two figures, and three pre-declared acceptance checks:
- the oracle action control reaches C ≥ 0.8 with feasible rate ≥ 0.95;
- every misspecification kind raises the median replay error by ≥ 1e-3 at its largest level;
- every unit is attributed.
