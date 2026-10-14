# TWIST

**T**emporal **W**orld-relative **I**ntervention, **S**imulation, and **T**raceability: a benchmark
for counterfactual explanations of time-series classifiers.

Conventional counterfactual metrics check whether an edited trajectory flips a classifier. TWIST
also asks three further questions:
- Is there an **action** that generates that trajectory?
- Does the action achieve the target **in the world**?
- Is it feasible, robust and auditable?

Every benchmark world is a known temporal structural causal model (SCM). An explanation's declared
action is replayed through the world for the same factual unit, using that unit's abducted
innovations. For trajectory-only methods, a registered benchmark reading of the proposal is
replayed instead. The gap between "the model agrees" and "the world agrees" is then measured
directly and decomposed.

> **Status.** Weeks 1–9 of the 12-week plan are implemented:
> - the benchmark core, baselines, mechanism access, robustness, statistics and validity;
> - causal discovery (PCMCI, VAR-LiNGAM) as an R4 mechanism source;
> - a semi-synthetic **energy domain bridge** with explicit action variables (battery dispatch,
>   demand response), domain feasibility constraints and real Dutch weather and demand
>   covariates.
>
> The evaluation protocol is at **v1.2** ([`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)).
> Weeks 10–12 (full campaign, analysis, release) are next.

## Contents

- [Core idea in one table](#core-idea-in-one-table)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Using the library](#using-the-library)
- [Command line and Make targets](#command-line-and-make-targets)
- [What the benchmark contains](#what-the-benchmark-contains)
- [Statistical protocol](#statistical-protocol)
- [Results so far](#results-so-far)
- [Running on Snellius / SLURM](#running-on-snellius--slurm)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Roadmap status](#roadmap-status)

## Core idea in one table

For an explanation with proposal `x̂` and action `𝒜`, replayed in the world as `x^𝒜`:

| Quantity | Question | Definition |
|---|---|---|
| `A` | Does the model accept the proposal? | `Pr[h(x̂) = y*]` |
| `B` | Does the model accept the action's real effect? | `Pr[h(x^𝒜) = y*]` |
| `C` | Does the **world** reach the target? | `Pr[ℓ*(x^𝒜) = y*]` |
| `Δtraj = A − B` | Does the action generate the proposal? | trajectory mismatch |
| `Δoutcome = B − C` | Do model and world agree on the replay? | outcome disagreement |
| `E_replay` | How far is the proposal from what the action does? | mean abs. deviation `x̂` vs `x^𝒜` |
| `D` | How many timesteps must be imposed to reproduce `x̂`? | do-complexity (not a cost) |
| safe-world success | Is the action feasible *and* successful in the world? | `Pr[Feasible ∧ ℓ*(x^𝒜) = y*]` |

`A − C` is never reported without `A, B, C`, the reading, `D`, `E_replay`, feasibility and the
denominators. There is no scalar leaderboard.

## Installation

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
make install    # .venv with Python 3.11; editable install with the dev, baselines and discovery extras
make check      # ruff + mypy + pytest
```

The `baselines` extra installs the third-party counterfactual library
[`counterfactual-explanations-for-time-series`](https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series)
at a pinned commit. It provides Wachter, COMTE, M-CELS, CONFETTI, Native-Guide and Latent-CF.
The `discovery` extra installs [Tigramite](https://github.com/jakobrunge/tigramite) (PCMCI, GPL-3.0).
VAR-LiNGAM is implemented in-house. Containers: `docker build -t twist .` or `make container`
(Apptainer, for HPC).

## Quickstart

```bash
make toy        # world → GRU → explainers → A/B/C → certificates, in under a minute on 1 core
make controls   # positive/negative structural control suite
make sanity     # metric sanity report: controls × {linear, nonlinear, spring} + audits
make validity   # construct/discriminant validity of the metrics (≈7 min on 1 core)
```

`make toy` runs `twist run --config experiments/toy.yaml` and writes
`artifacts/evaluations/toy_linear__<UTC>__<hash>/`:

```
config.resolved.json   records.json   summary.json   summary.md   run_info.json
certificates/<explainer>__<reading>__<id>.json      figures/case__<explainer>__<reading>.png
```

Proposal and replay arrays go to `artifacts/explanations/<run_id>/`, one file per explainer,
seed and reading.

Every run also appends one manifest record per explainer × reading to
`artifacts/manifest.jsonl`. `twist audit <run_dir>` re-replays every certificate and checks all
hashes. Example output (linear toy world, 20 flip candidates):

| method | reading | V | P-faith | D | E_rep | A | B | C | Δtraj | Δtotal | n_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PearlSCMRecourse ‡ | native | 1.00 | 1.00 | 1.0 | ≈0 | 1.00 | 1.00 | 0.95 | 0.00 | 0.05 | 20 |
| NoiselessSCMRecourse ‡ | native | 1.00 | 0.00 | 14.5 | 9e-2 | 1.00 | 0.55 | 0.55 | 0.45 | 0.45 | 20 |
| Wachter | first_change_atomic | 1.00 | 0.00 | 39.95 | 5e-2 | 1.00 | 0.00 | 0.10 | 1.00 | 0.90 | 20 |
| Wachter | first_changed_segment | 1.00 | 0.00 | 39.95 | 7e-4 | 1.00 | 0.70 | 0.70 | 0.30 | 0.30 | 20 |
| Wachter | full_changed_schedule | 1.00 | 0.00 | 39.95 | 7e-4 | 1.00 | 1.00 | 0.95 | 0.00 | 0.05 | 20 |

‡ structural controls. The trajectory-only baseline flips the model, but no compact reading of its
edit reproduces it. `Δtraj` falls only as the reading imposes more of the edit (`D ≈ T`). See
[`docs/technical_notes/native_vs_reconstructed_actions.md`](docs/technical_notes/native_vs_reconstructed_actions.md).

## Using the library

```python
import numpy as np
from twist.worlds import make_world
from twist.actions import PointIntervention
from twist.replay import replay_pearl, replay_noiseless
from twist.metrics.mechanism import do_complexity

world = make_world("linear", "demo", seed=0, k=5, T=40, density=0.3)
x, eps = world.simulate(100, np.random.default_rng(0))
world.label.fit(x)

action = PointIntervention(t=20, targets=[0, 1], values=[1.0, -0.5])
x_cf = replay_pearl(x[0], action, world)        # unit-level counterfactual (factual innovations)
x_nl = replay_noiseless(x[0], action, world)     # deterministic skeleton
print(world.label(x_cf), do_complexity(world, x[:1], x_cf[None]))   # world label, D = [1]
```

To add an explainer:
1. Subclass `twist.explainers.base.Explainer`.
2. Declare its metadata: role, regime, data and gradient needs, whether it is stochastic.
3. Return `Explanation` objects, with a native `ActionPlan` if the method declares actions.
4. Register it in `twist.explainers.registry`.

Its information is restricted to its mechanism-access regime by `build_context`
([`docs/mechanism_access.md`](docs/mechanism_access.md)).

## Command line and Make targets

```bash
twist run --config experiments/toy.yaml               # one paired experiment
twist sanity                                          # metric sanity report (controls + audits)
twist audit artifacts/evaluations/<run_id>            # traceability audit of a run
twist build-suite --config suites/dev.yaml            # multi-world suite: splits, hidden worlds, integrity
twist check-worlds artifacts/suites/twist_dev_v1      # re-run integrity checks
twist pilot --config experiments/pilot.yaml           # Week 5: worlds × classifiers × explainers × seeds
twist ladder --config experiments/access_ladder.yaml  # Week 6: mechanism-access ladder + misspecification
twist robustness --config experiments/robustness.yaml # Week 7: shifts, stability, p_robust, tolerance
twist validity --config experiments/validity.yaml     # Week 8: construct/discriminant validity
twist study --config experiments/study.yaml           # Week 8: pilot multi-world study (paired statistics)
twist study --config experiments/study.yaml --report-only   # rebuild the study report from finished runs
twist discovery --config experiments/discovery.yaml   # causal discovery as a mechanism source (graph vs recourse)
twist build-suite --config suites/energy.yaml         # Week 9: semi-synthetic energy suite (real NL covariates)
twist bridge --config experiments/bridge.yaml         # Week 9: energy domain bridge (controls, domain constraints)
twist compat                                          # regenerate the explainer compatibility matrix
```

Config paths resolve relative to `configs/`. Multi-world commands are resumable: a task is reused
only when both its config hash and the code fingerprint match. `make help` lists all targets:
`install, lint, format, typecheck, test, test-fast, smoke, toy, controls, sanity, audit, suite-dev,
suite-pilot, pilot, ladder, robustness, validity, study, study-report, discovery, suite-energy,
bridge, compat, check,
clean-artifacts, container`.

## What the benchmark contains

| Component | Contents | Docs |
|---|---|---|
| **Worlds** | Five families: linear VAR, nonlinear SCM, spring system, regime-switching, and partially observed state-space. A generator varies 12 factors (lag, density, strength, stability, noise law, label type, observation, missingness, sampling, confounding, …). Labels are threshold, exposure, onset or multi-output. Splits are dev/val/test/challenge, with hidden test/challenge mechanisms. Every world passes integrity checks (causal route, distractor, contractivity, diversity). | [`world_families.md`](docs/world_families.md), [`dataset_card.md`](docs/dataset_card.md) |
| **Domain bridge** | Semi-synthetic `energy` family: a distribution transformer with PV, a battery under a peak-shaving policy, and demand response with rebound. Inputs are real OPSD Netherlands weather and demand (2016–2019, CC-BY 4.0) entering as root innovations, and the outcome is an IEC-style hot-spot overload. It has native controls with action cards (meaning, unit, limits, delay, cost, side effects, descendants) and domain constraints (battery capacity, C-rate, DR energy cap). | [`energy_domain.md`](docs/energy_domain.md) |
| **Actions** | Point, schedule, interval, parametric pulse and high-level control, all compiled to a dense `(mask, value, shift)` replay contract. Feasibility constraints and costs are included. | [`action_schema.md`](docs/action_schema.md) |
| **Replay** | Exact abduction for additive-noise worlds, with Pearl replay, noiseless rollout and the interventional distribution. Partially observed worlds use an oracle replay from stored exogenous variables. | [`evaluation_protocol.md`](docs/evaluation_protocol.md) §2 |
| **Readings** | `native` (declared action), or five diagnostic readings of trajectory-only proposals: first slice, first segment, sparse segmentation, pulse fit, full schedule. | [`native_vs_reconstructed_actions.md`](docs/technical_notes/native_vs_reconstructed_actions.md) |
| **Classifiers** | GRU, LSTM, TCN, Transformer, and a logistic model on features. Training is deterministic on CPU. | `configs/classifiers/` |
| **Explainers** | Third-party baselines (cfts library: Wachter, COMTE, M-CELS, CONFETTI, Native-Guide, Latent-CF); in-house NUN and Wachter reference; action-native recourse at graph / estimated / oracle access. Structural controls: Pearl/noiseless SCM recourse, full-schedule oracle, R6 oracle recourse, adversarial edit, mechanism-wrong edit, abstain. Budgets are shared and failures are recorded with a status. | [`compatibility_matrix.md`](docs/compatibility_matrix.md) |
| **Mechanism access** | Regimes R0–R6, enforced by construction. Estimators: masked linear or neural on the true graph (R3); Lasso VAR, neural transition, and causal discovery (PCMCI, VAR-LiNGAM) followed by a masked fit (R4). Discovered graphs are scored against the true graph. Controlled misspecification and graph corruption are available. | [`mechanism_access.md`](docs/mechanism_access.md) |
| **Metrics** | Model-relative (validity, proximity, DTW, sparsity, OOD); mechanism-relative (CF-faith, D, E_replay, graded faith, causal-path consistency, retroactivity); world-relative (A/B/C, safe-world and action-grounded success); actionability; traceability. | [`metric_specification.md`](docs/metric_specification.md) |
| **Robustness** | A shift library at the innovation, mechanism, observation and decision levels. Also: innovation-perturbation stability, tolerance profiles, `p_robust` over plausible mechanisms, calibration of reported probabilities, and a CI-aware brittleness verdict. | [`robustness.md`](docs/robustness.md) |
| **Traceability** | Per-run manifests, config/dataset/mechanism hashes, code fingerprints, action certificates, and the `twist audit` re-replay. | [`reproducibility.md`](docs/reproducibility.md) |

## Statistical protocol

Statistical units are world → classifier → factual unit → explainer seed
(`twist.evaluation.statistical_analysis`).
- **Every estimate** gets a 95% CI from a four-level hierarchical bootstrap.
- **Methods are compared** only by paired differences on the same flip candidates, using
  intention-to-treat world success, where a missing action counts as failure. Each comparison
  reports the mean difference with its CI, `d_z`, and the probability of superiority.
- **Unpaired populations**, such as base vs shifted worlds, use a world-blocked bootstrap.
- **Reports contain** a primary metric table, paired forest plots, a Pareto frontier (world
  success vs action magnitude), per-world distributions and a failure-mode taxonomy. There is no
  aggregate score.

## Results so far

All numbers below are from dev worlds; hidden test and challenge worlds are reserved for the full
campaign. Full details are in [`docs/decision_log.md`](docs/decision_log.md) and the reports
under `artifacts/`.

| Week | Result |
|---|---|
| 3 | **Metric sanity:** 42/42 pre-declared control signatures pass on linear, nonlinear and spring worlds, and the audits reproduce every certificate. |
| 4 | **Dev suite:** 40 worlds across 5 families pass integrity and family-diversity checks. |
| 5 | **Pilot** (3 worlds × GRU/TCN × 7 baselines): every unit yields an output or a structured failure. The draft's model-vs-world gap reproduces: Δtotal > 0 for every trajectory baseline under the first-slice reading. |
| 6 | **Mechanism-access ladder:** world success rises from 0.13–0.21 (R0/R1 baselines) to 0.65 (R2 graph), 0.92–0.96 (R3/R4 estimated mechanisms) and 0.97–0.98 (R5/R6 oracle). Baselines fail mostly by replay mismatch, and misspecification degrades C. |
| 7 | **Robustness:** mechanism shifts hurt precise action-native plans most; the oracle drops from 0.97 to 0.42–0.64. The CI-aware brittleness verdict flags R4-Lasso, R6-oracle and Pearl recourse, and clears the other six methods. Innovation, observation and decision shifts move C by ≤ 0.1. |
| 8 | **Validity:** all five construct-validity pairs and all four discriminant tests pass (`artifacts/validity/`). **Pilot study** (25 worlds): the model–world gap replicates in 25/25 worlds; 2/4 pre-declared checks pass (details below). |
| 9 | **Energy bridge** (5 semi-synthetic worlds, real NL covariates; `artifacts/bridge/`): 3/3 checks pass. Domain constraints reject 99% of the constraint-blind battery dispatches that avert an overload: they overfill or overdrain the battery. Constraint-aware planning raises acceptability only to 0.05 (+0.047 [+0.020, +0.080], HE1). An abstract `do(grid)` set-point looks acceptable 0.75 of the time (realizability gap +0.70, HE2). Trajectory baselines are 100% invalid because they edit weather and demand. Exploratory: the feasible single-window dispatch ceiling is 0.21. |
| 8b | **Causal discovery** (10 worlds; `artifacts/discovery/`): PCMCI, VAR-LiNGAM and Lasso recover the graph with F1 0.64–0.72. Planning on a PCMCI graph instead of the true one costs +0.24 [+0.06, +0.42] world success (paired; H2 supported). Graph F1 alone does not predict success (H1 not supported). In exploratory analysis, missing edges hurt and spurious ones barely do. |

### Week-8 pilot multi-world study

`twist study`, report in `artifacts/studies/study_week8/report/`. Design: 25 fresh-seed dev
worlds (5 per family) × GRU/TCN × 100 flip candidates × 3 seeds; 5 baselines and 6 controls;
141k records. Every number is a mean with a 95% four-level hierarchical-bootstrap CI.

| Method (first-slice reading) | world success (ITT) | A | C | Δtotal = A − C | feasible |
|---|---|---|---|---|---|
| NativeGuide | 0.46 [0.37, 0.54] | 1.00 | 0.50 [0.43, 0.57] | 0.50 [0.43, 0.57] | 0.87 |
| Wachter (cfts) | 0.23 [0.17, 0.29] | 0.99 | 0.25 [0.19, 0.31] | 0.74 [0.68, 0.80] | 0.80 |
| NUN | 0.21 [0.15, 0.27] | 1.00 | 0.23 [0.17, 0.29] | 0.77 [0.71, 0.83] | 0.88 |
| COMTE | 0.18 [0.13, 0.24] | 1.00 | 0.18 [0.13, 0.24] | 0.82 [0.76, 0.87] | 0.94 |
| LatentCF | 0.15 [0.10, 0.21] | 0.99 | 0.24 [0.17, 0.30] | 0.76 [0.68, 0.82] | 0.99 |
| R6 oracle recourse ‡ | 0.80 [0.67, 0.91] | 0.95 | 0.80 [0.67, 0.91] | 0.15 [0.06, 0.26] | 1.00 |

- **The draft's model–world gap replicates in 25/25 worlds, across all five families.** Per-world
  Δtotal is 0.50–0.90, and every CI excludes 0.
- **Paired comparisons** use the same 5000 units per method.
  - NativeGuide beats every other baseline by +0.23 to +0.31, with every CI excluding 0.
  - LatentCF trails NUN and Wachter, because 31% of its units get no flip.
  - Wachter, COMTE and NUN are statistically indistinguishable.
- **Pareto front** (world success vs action magnitude): COMTE, LatentCF, NUN and NativeGuide.
- **Pre-declared checks: 2/4 pass** (paired conclusions; replication). The positive and negative
  control checks fail on the CI rule, although their point estimates meet the thresholds:
  - **R6 oracle recourse** (C lower bound 0.67 < 0.8) succeeds on exposure and onset labels. On
    threshold and multi labels in linear and regime worlds, it optimizes the classifier, which
    the world then disagrees with.
  - **The mechanism-wrong edit** is too weak a perturbation in linear and regime worlds
    (E_replay ≈ 1e-3), though it is clear in nonlinear and spring worlds.

  See `docs/decision_log.md` (Week 8, item 10).


## Running on Snellius / SLURM

Interactive nodes have small CPU/memory limits and kill long CPU-bound processes. Keep
`TWIST_NUM_THREADS=1` there (torch and BLAS pools are pinned to it). Use SLURM for multi-world runs:

```bash
sbatch scripts/slurm/multiworld.sbatch experiments/pilot.yaml                            # Week-5 pilot
sbatch scripts/slurm/multiworld.sbatch experiments/access_ladder.yaml ladder             # Week-6 ladder
sbatch scripts/slurm/multiworld.sbatch experiments/robustness.yaml robustness            # Week-7 robustness
sbatch --cpus-per-task=32 scripts/slurm/multiworld.sbatch experiments/study.yaml study   # Week-8 study
sbatch scripts/slurm/build_suite.sbatch                                                   # full suite
```

Tasks run as parallel single-threaded processes (`--jobs = SLURM_CPUS_PER_TASK`). Check the
partition, account and time limit against your allocation before submitting.

## Repository layout

```
configs/              worlds/, classifiers/, explainers/, experiments/, suites/
src/twist/
  actions/            action classes, dense schedules, readings (adapters), feasibility, costs
  worlds/             SCM base, five families + semi-synthetic energy, labels, shifts, factors, suites, integrity
  data/energy/        real NL weather/demand profiles (OPSD, CC-BY 4.0) with provenance JSON
  replay/             abduction, Pearl replay, noiseless rollout, interventional sampling
  classifiers/        GRU/LSTM/TCN/Transformer/logistic classifiers and training
  mechanisms/         estimators (masked, Lasso VAR, neural), causal discovery, quality, misspecification
  explainers/         base API, budgets, regimes R0–R6, baseline adapters, controls, action-native recourse
  metrics/            conventional, mechanism, outcome, traceability, robustness, sensitivity, aggregation
  evaluation/         protocol, paired + multi-world runners, audit, sanity, pilot/ladder/robustness
                      reports, statistical_analysis, validity, study, discovery, bridge
  reporting/          certificates, benchmark cards, compatibility matrix, tables, figures
  utils/              seeds/threads, logging, serialization/hashing/fingerprints
scripts/              CLI wrappers, SLURM batch scripts, energy-profile fetcher
tests/                worlds, replay, actions, metrics, controls, mechanisms, robustness,
                      evaluation (statistics, validity, study), reproducibility
artifacts/            generated suites, runs and reports (git-ignored)
docs/                 protocol v1.2, specifications, cards, energy domain, decision log
paper/                figures, tables, appendix, results manifest
```

## Documentation

- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md): **protocol v1.0**. Formal objects, actions, regimes, sampling, metrics, controls, shifts, the statistical protocol, validity, conventions and claim boundaries.
- [`docs/metric_specification.md`](docs/metric_specification.md): every metric and statistic, with its definition, population, code and tests.
- [`docs/world_families.md`](docs/world_families.md): world families, generator factors, labels, integrity checks, splits.
- [`docs/action_schema.md`](docs/action_schema.md): action kinds, serialization, control interfaces.
- [`docs/mechanism_access.md`](docs/mechanism_access.md): R0–R6, estimators, misspecification, action-native recourse, failure attribution.
- [`docs/energy_domain.md`](docs/energy_domain.md): the energy bridge — data provenance, mechanism, action cards, domain constraints, claim boundary, domain-review questionnaire.
- [`docs/robustness.md`](docs/robustness.md): shifts, stability, tolerance sensitivity, uncertainty, brittleness.
- [`docs/compatibility_matrix.md`](docs/compatibility_matrix.md): explainer metadata, generated by `twist compat`.
- [`docs/technical_notes/native_vs_reconstructed_actions.md`](docs/technical_notes/native_vs_reconstructed_actions.md): how to read results across readings.
- [`docs/reproducibility.md`](docs/reproducibility.md): environment, manifests, determinism.
- [`docs/decision_log.md`](docs/decision_log.md): every decision and deviation from the plan, week by week.
- [`docs/benchmark_card.md`](docs/benchmark_card.md), [`docs/dataset_card.md`](docs/dataset_card.md).

## Roadmap status

| Week | Focus | Status |
|---|---|---|
| 1 | Protocol v0.1, scaffold, CI, toy run | **done** |
| 2 | Action API, replay oracle, readings, feasibility, costs | **done** |
| 3 | Metrics, certificates, structural controls | **done**: sanity report 42/42 |
| 4 | Multi-world suite (5 families, 12 factors, 5 label types, hidden splits) | **done** |
| 5 | Classifiers, third-party baselines, budgets, pilot | **done** |
| 6 | Mechanism-access ladder, estimators, action-native recourse, misspecification | **done**: 3/3 checks |
| 7 | Shifts, stability, tolerance sensitivity, uncertainty, robustness | **done**: 3/3 checks |
| 8 | Statistical protocol, benchmark validity, pilot multi-world study, protocol v1.0 | **done**: validity 9/9; study checks 2/4 (two controls fail on the CI rule; see results) |
| 9 | Causal discovery as a mechanism source; semi-synthetic realism bridge | **done**: discovery 3/3 checks (+0.24 cost of not knowing the graph); energy bridge 3/3 checks, HE1 and HE2 supported. Domain-expert review pending |
| 10–12 | Full experiments, analysis, publication package | not started |

The TSCausalCF method is being developed separately and will plug in through the explainer API.
