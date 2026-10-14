# World families and generator cards (Week 4)

Every world is a temporal SCM with a known mechanism, a label rule `ℓ*` and an action space.
Worlds are built from a family and named factor levels (`twist.worlds.factors.resolve_world`).
Suites (`twist build-suite`) sample many worlds per family into dev/val/test/challenge splits.
Each generated world gets its own card (`WORLD_CARD.md`) with mechanism summary, label, designated
structure, action space, replay disclosure and integrity results.

## Families

| Family | State equation | Exact abduction from `x` | Factor knobs |
|---|---|---|---|
| `linear` | stable sparse VAR(L): `x_t = Σ_ℓ A_ℓ x_{t−ℓ} + s·η_t` | yes | lag, density, strength (entry range), stability (radius cap; oscillatory = rotation on the route pair) |
| `nonlinear` | `x^i_t = decay_i x^i_{t−1} + gain·tanh(MLP_i(masked lags)) + s·η` | yes | lag, density, strength (gain), stability (decay range; negative decay = oscillatory) |
| `spring` | coupled springs, symplectic Euler; positions + velocities | yes | density, strength (stiffness, heterogeneous per edge), stability (damping γ and tether κ; γ = κ = 0 is the draft's undamped system) |
| `regime` | `x_t = F_{r_t}(x_{t−L:t}) + s·η`, two linear regimes | yes (regime is a function of observed past) | lag, density, strength, stability, regime driver (state: `r_t = 1{w·x_{t−1} > 0}`; exogenous: observed AR(1) driver channel, not actionable) |
| `statespace` | any of the above as latent state `z`, observed `x = H(z) + ν` with LOCF | **no** (oracle replay from stored exogenous) | state family, observation (noisy / saturating nonlinear / partial), missingness (MCAR / block), sampling (regular / irregular), confounding (latent AR(1) confounder of label and distractor, hidden) |

**Domain bridge (Week 9).**

| Family | State equation | Exact abduction from `x` | Factor knobs |
|---|---|---|---|
| `energy` | semi-synthetic distribution transformer: real NL weather/demand covariates as root innovations; battery with a default peak-shaving policy; demand response with rebound; IEC-style hot-spot thermal model; ageing stress (distractor). Native controls `u_dispatch`, `u_dr` | yes | pv_penetration, battery, season, energy_label |

`energy` is listed in `DOMAIN_FAMILIES`, not `FAMILIES`, so default suites do not include it. Its
own noise model, mechanism, constraints and claim boundary are described in
[`energy_domain.md`](energy_domain.md).

All synthetic families share:

- **Noise** — Gaussian, Laplace, heteroscedastic (`s_i = 1 + h·tanh|x^i_{t−1}|`, exogenous `η`
  still exactly abducted) or equicorrelated across channels.
- **Labels** — terminal threshold, interior threshold, distributed exposure, event onset, and
  multi-output (`and` of the label channel's terminal value and its route source's late exposure).
- **Designated structure** (`world.roles`) — `label_channel` (0); a `route_source` with a
  guaranteed lag-1 edge into the label channel; and a `distractor` that is driven by the route
  source but causes nothing (predictive, non-causal). Statespace worlds may also have a hidden
  `latent_confounder`; regime worlds have a `regime_driver`.

The draft's three families are reproduced exactly by the generator defaults (regression-tested).
Factor-driven worlds enable route/distractor designation, heterogeneous springs and the
stability mappings above.

## Replay semantics by family

- **Exact families** (linear, nonlinear, spring, regime, and statespace with identity
  observation): `η_t = (x_t − F(x_{t−L:t}))/s(·)` is recovered exactly from `x`, so unit-level
  Pearl replay, CF-faith, `D` and residual-based readings are defined.
- **Partially observed** (`statespace` with any non-identity observation): the factual exogenous
  variables `(z, η, ν, missingness mask)` are stored per unit. The oracle replays the action on
  the state and re-observes it with the factual sensor exogenous. Actions act on **state** channels;
  readings of observed edits map through `H⁻¹`; world outcomes are labels of the **true state**.
  Abduction-based metrics are NaN, and methods that abduct from `x` (the SCM-recourse controls,
  the full-schedule oracle) are skipped with a recorded reason. Estimated-mechanism recourse (R3/R4)
  does run there: it fits `F̂` on the observations and plans in observed coordinates, which the
  evaluator maps to state channels. Posterior abduction with the oracle mechanism is not
  implemented yet; it belongs with the uncertainty work (Week 7).

## Integrity and acceptance (every suite world)

`twist.worlds.integrity.check_world` must pass, and each world must differ by ≥ 5% (relative
functional distance on the increment over persistence) from every accepted world of its family.
Candidates that fail are rejected and resampled; rejections are listed in the private manifest
and counted in `integrity_report.md`.

| Check | Criterion |
|---|---|
| finite, bounded, no_growth | finite; max \|z\| < 1e3; std(last quarter)/std(second quarter) < 10 |
| graph_shape, no_lag1_self_loops, graph_covers_parameters | well-formed; no lag-1 self-loops on system channels (spring kinematics, drivers and confounders excepted); non-zero parameters ⊆ dependency graph |
| action_space | non-empty actionable set; controls well-formed |
| label_fitted, label_balance, label_support | fitted; P(y=1) ∈ [0.2, 0.8]; support after the roots |
| causal_route_graph | an actionable non-label channel reaches the label support (any action time before the site) |
| causal_route_effect | some do(x + 1 sd) on a route channel moves the label margin by > 1% of its sd under oracle replay |
| distractor_exists, designated_distractor_non_causal | a non-ancestor of the label exists; the designated distractor is one |
| empty_action_replay | no-action replay reproduces every unit (≤ 1e-9) |
| exactness_disclosed | the exact-abduction flag matches the observation model |

## Splits

| Split | Default size / family | Mechanism | Factor pool |
|---|---|---|---|
| dev | 10 | visible | standard |
| val | 10 | visible | standard |
| test | 20 | hidden | standard |
| challenge | 10 | hidden | challenge (L5, oscillatory, heteroscedastic/correlated noise; nonlinear/partial observation, missingness, irregular sampling, latent confounding) |

Hidden worlds keep data and a redacted card in `worlds/<id>/`. Mechanism, roles, config, seed and
full card live in `hidden/<id>/` and load only with `load_suite_world(..., allow_hidden=True)`. Set
`TWIST_SUITE_SECRET` when building to derive hidden seeds from a secret. Otherwise anyone with the
public suite seed can regenerate them, and the dataset card says so.

Configs: `configs/suites/dev.yaml` (2 worlds per split and family, k=6, T=50, N=1000; 40 worlds)
and `configs/suites/v1.yaml` (the plan's 10/10/20/10, k=10, T=100, N=5000; 250 worlds — build
with `scripts/slurm/build_suite.sbatch`).
