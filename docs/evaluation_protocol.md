# TWIST evaluation protocol — v1.0 (Week 8)

Status: **v1.0, frozen for the full experiments (Weeks 9–12).** v0.1 (Week 1) was frozen for
Weeks 2–7. v1.0 folds in the decisions recorded in [`decision_log.md`](decision_log.md) for Weeks 2–8:
- world families and the partially observed world (§2.2);
- action tiers (§3) and mechanism-access regimes (§4);
- budgets and the failure taxonomy (§5);
- shifts (§7);
- the statistical protocol and the method-comparison policy (§8);
- benchmark validity (§9).

Later changes are logged with a version bump.

## 1. Purpose

TWIST evaluates counterfactual explanations of time-series classifiers **against a world**, not
only against the model. A known temporal SCM defines each benchmark world. An explanation's declared
(or benchmark-reconstructed) action is replayed through that world for the same factual unit, and
the benchmark records whether

1. the proposal changes the classifier (model-relative success),
2. the action generates the proposal (mechanism consistency),
3. the replayed action achieves the target in the world (world-relative success),
4. the action is executable (actionability),
5. conclusions survive shifts and uncertainty (robustness), and
6. an auditor can reproduce why the explanation was selected (traceability).

## 2. Formal objects

| Object | Symbol | Code |
|---|---|---|
| World (temporal SCM) | `M* = (G*, F*, P*, ℓ*)` | `twist.worlds.TemporalSCM` |
| Factual unit | `x_{0:T-1}`, `x_t ∈ R^k` | `(T, k)` array; ids `"<world_id>/test/<index>"` |
| Target outcome | `y*` | `int` |
| Explanation | `e = (𝒜, x̂, y*, 𝒞, r, u)` | `twist.evaluation.result_schema.Explanation` |
| Action plan | `𝒜` | `twist.actions.ActionPlan` wrapping a `TemporalAction` |
| Proposal | `x̂_{0:T-1}` | `Explanation.proposed_trajectory` |
| Replay | `x^{𝒜,*} = Replay(x, 𝒜, M*)` | `twist.replay.replay_pearl` / `replay_noiseless` |
| Classifier | `h`, logits `z_h` | `twist.classifiers.TorchClassifier` |
| World label | `ℓ*(x) = 1{score(x) > 0}`: threshold `x[t_label, j] > θ`, exposure, onset, multi-output | `twist.worlds.labels` |

Vocabulary (no other terms are used for these objects):

- **proposal** — the trajectory the explainer outputs;
- **action** — an intervention plan; **native** if the explainer declared it, **reconstructed** if
  a benchmark adapter read it off a proposal;
- **reading** — the rule that maps an explanation to an action (`native`, or one of the registered
  adapters `first_change_atomic`, `first_changed_segment`, `sparse_segmentation`,
  `change_point_pulse_fit`, `full_changed_schedule`; see
  [`technical_notes/native_vs_reconstructed_actions.md`](technical_notes/native_vs_reconstructed_actions.md));
- **replay** — the trajectory the world produces under the action;
- **label** — `ℓ*` applied to a trajectory; **prediction** — `h` applied to a trajectory;
- **target** — `y*`.

### 2.1 SCM and replay

Lagged additive-noise SCM with `L ≥ 1`:

```
X^i_t = F*_i(x_{t-1}, …, x_{t-L}) + ε^i_t      (t ≥ L)
```

The first `L` timesteps of a stored trajectory are exogenous roots (`ε_t := x_t`). Abduction is the
exact subtraction `ε_t = x_t − F*(Pa(x_t))`. For a dense schedule `(mask, value, shift)`:

```
x^𝒜_t = value_t                              where mask_t      (hard do())
x^𝒜_t = F*(x^𝒜_{t-L:t}) + ε̃_t + shift_t       otherwise         (shift = soft intervention)
```

- **Pearl replay**: `ε̃ = ε` (factual innovations retained) — a unit-level counterfactual.
- **Noiseless rollout**: `ε̃_t = ε_t` for `t ≤ t_first`, `0` after — the deterministic skeleton.
- **Interventional distribution**: fresh innovations after `t_first` — `P(X | do(𝒜), prefix)`.

These answer different questions and are **never averaged**. Exactness requires that the recorded
`G, F` are the data-generating SCM, positive lags, no unmeasured confounding, additive innovations
unchanged by the action. Worlds declare these in `world.assumptions`; replay refuses worlds whose
`exact_replay` is false; such worlds are replayed by the oracle from stored exogenous variables (§2.2).

### 2.2 Partially observed worlds

In `statespace` worlds the observations `x = H(z) + ν` (with hidden channels, saturation,
missingness or irregular sampling, carried forward) do not identify the factual exogenous
variables. The benchmark therefore stores each unit's `(z, η, ν, mask)`. The oracle replays an
action on the **state** (actions act on state channels) with the factual innovations, then
re-observes with the factual `ν` and mask. World outcomes are `ℓ*` of the true state, so
`Δoutcome` can be non-zero. Readings of observed edits map to state channels through `H⁻¹`.
Abduction-based metrics (CF-faith, `D`, residual readings, causal-path consistency) are NaN, and
methods that abduct from `x` with the oracle mechanism are skipped with a recorded reason. Estimated-
mechanism recourse runs there in observed coordinates (§4); oracle posterior abduction is future work. World families, factors, labels and splits are specified in
[`world_families.md`](world_families.md).

## 3. Action tiers

| Tier | Form | Class | Status |
|---|---|---|---|
| 1 | Point intervention `do(X^S_t = a_t)` | `PointIntervention` | implemented |
| 1b | Observed-variable schedule `{(t, S_t, a_t)}` | `ScheduleIntervention` | implemented |
| 2 | Interval intervention `do(X^S_{t:t+Δ} = a_{t:t+Δ})` | `IntervalIntervention` | implemented |
| 3 | Parametric pulse `a_t = φ(t; α, τ, Δ)` (soft, additive) | `ParametricPulse` | implemented |
| 4 | High-level control `u_t ∈ U` through `F(x_t, u_t, ε_t)` | `HighLevelControl` | implemented for world-declared linear input channels (Week 2); nonlinear actuation in Weeks 4/9 |

See [`action_schema.md`](action_schema.md) for the serialized form.

## 4. Mechanism-access regimes

| Regime | Explainer information | Status |
|---|---|---|
| R0 | black-box classifier and factual input | implemented |
| R1 | + observational training trajectories | implemented |
| R2 | + causal graph `G` | implemented (context only) |
| R3 | + transition family and graph, unknown parameters | implemented: graph-masked fit (`estimator: masked`) |
| R4 | + estimated mechanism `(Ĝ, F̂)` | implemented: Lasso VAR, neural transition, or controlled misspecification |
| R5 | + oracle mechanism `M*` | implemented |
| R6 | + oracle mechanism, action model and constraints | implemented: `OracleActionRecourse` (positive control) |

Regimes are enforced by `explainers.mechanism_access.regimes.build_context`: an explainer simply
does not receive what its regime excludes.

## 5. Factual-unit and target sampling

- Factual units are drawn from the **held-out test split** of each world.
- With a fixed target class `y*`, flip candidates are `C_{y*} = {n : h(x^(n)) ≠ y*}`; the first
  `n_factuals` candidates in index order are evaluated (draft §4.3). With `target: flip`,
  `y* = 1 − h(x)`.
- **Every explainer is evaluated on the same factual units** (paired design).
- **Intention-to-treat (ITT) for comparisons.** Method comparisons use world success over *all*
  flip candidates. A unit without a scorable action counts as a failure, so denominators are
  identical across methods. `A, B, C` keep the scorable denominator; they describe what an
  action does when there is one.
- The scorable population `S_{y*}` drops units whose reading yields no action and units whose
  action fails validation against `world.action_space()`. No-action proposals **abstain**
  (`n_abstain`). Invalid actions are **rejected** (`n_invalid_action`) with readable violation
  reports. Neither is ever scored as a successful zero-effect intervention.

### 5.1 Equal treatment of explainers and budgets

Every explainer gets the same classifier interface (raw world units in, logits out; the classifier
standardizes internally), the same factual units and targets, and the same feature metadata
(variable names; mutability only at R6). From R1 on, explainers also get the same reference data:
the training trajectories and labels, subsampled to `budget.max_reference` with the explainer seed.
Per-unit budgets are shared:

| Budget | Default | Enforcement |
|---|---|---|
| model queries (samples through the classifier) | 200,000 | counted by `BudgetedModel`; exceeding → `budget_exceeded` |
| wall clock | 120 s | checked at every model call; exceeding → `timeout` |
| reference trajectories | 500 | subsample handed to retrieval-based methods |
| method-internal iterations, restarts, population | method defaults, recorded in configs | upstream parameters |

Batched in-house controls record their queries but are not interrupted mid-batch.

### 5.2 Failure taxonomy

Every unit ends in a standardized explanation or a structured failure record, never a silent drop:

| Status | Meaning |
|---|---|
| `ok` | proposal flips the classifier to the target |
| `no_target_flip` | the method returned no proposal, or one the classifier does not assign to the target |
| `abstain` | the method declines to act (controls) |
| `budget_exceeded` / `timeout` | per-unit query / time budget exhausted |
| `method_error` | the method raised; the exception message is recorded |
| `invalid_output` | wrong shape or unusable output |
| `numerical_instability` | non-finite output |
| `no_valid_action` / `infeasible_action` / `invalid_serialization` | action-level failures |

Action validation failures are counted separately (`n_invalid_action`). Methods that cannot run
on a world or regime are listed in `run_info.json` (`skipped_explainers`) with the reason.

## 6. Metrics

### Primary endpoints (pre-specified)

| Metric | Definition | Code |
|---|---|---|
| `A` | `Pr[h(x̂) = y*]` over `S_{y*}` | `metrics.outcome.abc_decomposition` |
| `B` | `Pr[h(x^{𝒜,*}) = y*]` | 〃 |
| `C = V_world` | `Pr[ℓ*(x^{𝒜,*}) = y*]` | 〃 |
| `Δtraj, Δoutcome, Δtotal` | `A−B`, `B−C`, `A−C = Δtraj + Δoutcome` | 〃 |
| `V_safe-world` | `Pr[Feasible(𝒜, x) ∧ ℓ*(x^{𝒜,*}) = y*]` | 〃 (`safe_world_success`) |
| `E_replay` | `(1/T) Σ_t mean_i |x̂_t,i − x^{𝒜,*}_t,i|` | `metrics.mechanism.replay_error` |

### Secondary / diagnostic

- **CF-faith (hard, soft)** under Pearl and noiseless semantics: read the first edited slice as one
  `do()`; residual = mean L1 over the suffix `t > t0`; hard = `residual < 1e-3`;
  soft = `exp(−residual/σ)`. `t0 ≥ T−1` or no edit → NaN (unscorable); denominators reported.
- **Do-complexity `D`**: number of timesteps that must be imposed for the mechanism to reproduce
  `x̂` (Pearl unless named). It is observed-variable schedule complexity, **not** action cost.
- **Action-grounded success**: `Pr[E_replay < 1e-3 ∧ ℓ*(x^{𝒜,*}) = y*]`.
- **Action costs** (`actions.costs`): duration, magnitude, smoothness, budget.
- **Model-relative**: validity, logit-margin change, L1/L2, IQR-normalized proximity, DTW,
  entry/channel/timestep/interval sparsity, temporal smoothness, Isolation-Forest OOD score (and
  its change vs the factual), runtime, failure rate.
- **Graded mechanism consistency** (scorable units): scale-normalized replay error, soft faith
  `exp(−E_replay_norm/σ)`, effect-profile correlation, affected-support overlap, retroactivity,
  prefix preservation, causal-path consistency.
- **Timing**: `t_label`, action `t_first`, label distance `t_label − t_first`.
- **Traceability**: provenance completeness, action round-trip, and the run audit (`twist audit`):
  array integrity, replay/outcome reproducibility, config/dataset/mechanism hash consistency.

The full definitions, ranges and populations are in [`metric_specification.md`](metric_specification.md).

### Structural controls

Controls validate the scoring semantics and are never ranked against explainers:

| Control | Role | Expected signature |
|---|---|---|
| `PearlSCMRecourse` | positive (Pearl semantics) | Pearl CF-faith 1, `D = 1`, `E_replay ≈ 0` |
| `NoiselessSCMRecourse` | positive (noiseless semantics) | noiseless CF-faith 1, Pearl CF-faith 0 |
| `FullScheduleOracle` | positive for replay, negative for compactness | `E_replay = 0`, action ≈ `T` timesteps |
| `AdversarialTrajectoryEdit` | negative (minimal-budget L∞ attack, ε ≤ 4 std) | high validity, CF-faith 0, low soft faith |
| `PlausibleButMechanismWrongEdit` | negative | plausible (`ΔOOD ≈ 0`), CF-faith 0, `E_replay` ≫ tol |
| `NoActionAbstain` | denominator | `n_scorable = 0` |
| `OracleActionRecourse` (R6) | positive (action-native recourse) | `C ≥ 0.8`, feasible ≥ 0.9 |

`twist sanity` (`make sanity`) checks these signatures on every world family against thresholds
pre-declared in `twist.evaluation.sanity.CHECKS`, and runs the audit.

### Reporting rule

Never report `A − C` without `A`, `B`, `C`, the reading, `D`, `E_replay`, feasibility, label distance
and the scorable denominator (`n_s`, abstentions, invalid actions). No scalar leaderboard. Structural controls are reported
as controls, never ranked against explainers. Trajectory-only baselines are described as "not
certified under the registered reading", never as "failing causal recourse".

## 7. Shift scenarios (Weeks 6–7)

Innovation (Laplace → Gaussian/uniform/heteroscedastic/correlated), mechanism (coefficients, edges,
delays, nonlinearity), observation (sensor noise, missingness, saturating transform), decision
(threshold, label site, classifier retraining). The classifier, mechanisms and explainer knowledge
stay those of the base world; units come from, and are replayed in, the shifted world. Stability
uses innovation-perturbed copies of the same units. Shift-VR = shifted/base validity, suppressed when
base < 0.30.

A method is *brittle under mechanism uncertainty* when it is model-valid (A ≥ 0.8) and the 95% CI
of some world-success drop lies entirely above 0.10:
- for a mechanism shift, the drop is estimated with a world-blocked bootstrap (base and shifted
  units differ);
- for the plausible-mechanism ensemble, it is the paired drop `C − p_robust`.

Details and uncertainty quantities: [`robustness.md`](robustness.md).

## 8. Statistical protocol and method-comparison policy (Week 8)

Code: `twist.evaluation.statistical_analysis`, `twist.evaluation.study`.

**Statistical units.** world → classifier → factual unit → explainer seed. Worlds are the
independent replicates. Units are paired across methods within a world × classifier.

| Quantity | Procedure |
|---|---|
| Estimates | mean (or median for `E_replay`, action fraction) with a 95% percentile CI from a hierarchical bootstrap that resamples worlds, then classifiers within worlds, then units within world × classifier, then seeds within units (1000 draws) |
| Paired comparison A − B | per unit, average over seeds; the difference is bootstrapped over worlds → classifiers → units. Report the mean difference, 95% CI, `d_z = mean(d)/sd(d)` and the probability of superiority `P(d>0) + ½P(d=0)` |
| Unpaired populations (base vs shifted units) | world-blocked bootstrap: worlds jointly, units independently per population |
| Draft replication | per world, the within-unit Δtotal = `1{h(x̂)=y*} − 1{ℓ*(x^𝒜)=y*}` of the trajectory baselines (first-slice reading), CI over classifiers → units |

**Method comparison policy.** There is no single "best method" score. A study report
(`twist study`) contains:
- a primary metric table, with CIs on every headline cell;
- a reading-sensitivity table;
- paired difference plots of ITT world success: every pair of non-control methods, reference first;
- a Pareto frontier of world success (↑) against action magnitude (↓);
- per-world distributions;
- a failure-mode taxonomy (attribution and explainer status).

A claim that one method beats another must come from a paired difference whose CI excludes 0.
Unpaired point estimates never support a comparison. Controls are reported, never ranked.

**Pre-declared study acceptance checks** (`twist.evaluation.study`):
1. **Positive controls.** Each meets its intended conditions on exact worlds, with the whole CI
   on the required side:
   - Pearl: `E_replay ≤ 1e-9`, CF-faith ≥ 0.95;
   - full-schedule oracle: `E_replay ≤ 1e-9`;
   - R6 oracle recourse: `C ≥ 0.8`, feasible ≥ 0.9.
2. **Negative controls.** Each fails at least one relevant criterion, with the whole CI on the
   failing side:
   - abstain: ITT success 0;
   - adversarial edit: CF-faith ≤ 0.1, or `C ≤ 0.5`;
   - mechanism-wrong edit: `E_replay ≥ 0.01`, or CF-faith ≤ 0.1;
   - full-schedule oracle: median action ≥ 0.9·T.
3. **Paired conclusions.** Every pair of non-control methods has a paired comparison on shared
   units with a finite CI, and every headline cell has a CI.
4. **Replication across worlds.** The draft's model–world gap (Δtotal > 0 for trajectory baselines)
   has a CI above 0 in ≥ 2 worlds from ≥ 2 families.

## 9. Benchmark validity (Week 8)

`twist validity` (`twist.evaluation.validity`) tests whether the metrics measure what they claim.

**Construct validity.** Controlled pairs are built from successful oracle point actions. Each pair
holds one property equal and varies one other. The varied metric's paired CI must exclude 0 in the
expected direction, and the held one must not differ.

| Pair | Held | Varied |
|---|---|---|
| A | model validity | mechanism consistency (replay error) |
| B | proximity | world success (equal edits on the label vs distractor channel) |
| C | replay error | feasibility |
| D | world success | action cost |
| E | plausibility | causal-path consistency |

**Discriminant validity.** These properties must not collapse into one another:
- low replay error ⇏ low action cost;
- plausibility ⇏ mechanism consistency;
- target validity ⇏ safe-world success;
- low |A − C| ⇏ good recourse when A is low.

The Week-8 report passes all nine tests on five exact dev worlds (`docs/decision_log.md`, Week 8).

## 10. Implementation conventions

| Topic | Convention |
|---|---|
| Tensor layout | batch × time × channel `(B, T, k)`; single unit `(T, k)` |
| Time index | zero-based, `t = 0 … T−1`; lag `ℓ ≥ 1`; `past[:, ℓ−1] = x_{t−ℓ}` |
| Graph | `graph[i, j, ℓ−1] = True` ⇔ `X^j_{t−ℓ} → X^i_t` |
| Label site | `t_label = −1` means terminal (`T−1`) |
| Dtype | float64 throughout (worlds, replay, classifiers) |
| Action serialization | JSON dict with `kind` (see `action_schema.md`) |
| Normalization | classifiers standardize inputs per channel with train-split mean/std, inside the model; explainers see raw world units |
| Missing data | explicit masks in partially observed (`statespace`) worlds; MCAR/block missingness, carried forward |
| Tolerances | change predicate `|x̂ − x| > 1e-3` (defines `t0`, sparsity, `S_t`, `D`); CF-faith hard threshold `1e-3`; numerical replay tolerance in tests `1e-10` |
| Seeds | `world_seed`, `classifier_seed`, `explainer_seed`, `evaluation_seed` recorded per run; child seeds via `utils.seeds.derive_seed(parent, *keys)` |
| Threads | `TWIST_NUM_THREADS` (default 1, or `SLURM_CPUS_PER_TASK`); pins torch and BLAS/OpenMP pools |
| Artifact naming | worlds `artifacts/worlds/<world_id>/`; models `artifacts/models/<world_id>__<arch>_seed_<s>/`; runs `artifacts/evaluations/<name>__<UTC>__<config_hash8>/` |
| Run manifest | append-only `artifacts/manifest.jsonl`, one `RunManifest` per explainer × reading |
| Output schema | `records.json` (one row per explanation × reading), `summary.json`/`summary.md`, `certificates/*.json` |

## 11. Limitations and claim boundaries

- Synthetic worlds define causal truth **by construction**; they say nothing about whether any real
  domain follows these mechanisms.
- Readings for trajectory-only methods are benchmark diagnostics, not semantics attributed to their
  authors.
- `D` depends on the observed-variable representation and on the change tolerance.
- Exact replay is limited to additive-noise, fully observed, unconfounded worlds. Partially
  observed worlds are replayed by the oracle from stored exogenous variables. Abduction-based
  metrics and oracle-abducting methods are undefined there and reported as such.
- Pilot-study conclusions (Week 8) come from dev worlds of a fresh-seed suite. Hidden test and
  challenge worlds are reserved for the full experiments.
- Human utility claims require a user/expert study and are out of scope.
