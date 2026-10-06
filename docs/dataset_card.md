# Dataset card — TWIST synthetic worlds

Every built suite also gets a generated, suite-specific card (`<suite>/DATASET_CARD.md`) with exact
counts, factor distributions per split, replay exactness and the hidden-seed disclosure. This page
describes what is common to all suites.

## Composition

Five world families (details in [`world_families.md`](world_families.md)):

| Family | Mechanism | Exact abduction from observations |
|---|---|---|
| `linear` | stable sparse VAR(L) | yes |
| `nonlinear` | per-node masked MLP with decay | yes |
| `spring` | coupled springs (positions + velocities), optional damping and tether | yes |
| `regime` | two linear regimes switched by the observed state or an observed exogenous driver | yes |
| `statespace` | any of the above as a latent state, observed through noise, saturation, hidden channels, missingness or irregular sampling | no — oracle replay from stored exogenous |

The draft's three reported families (LinearSCM-T, NlinearSCM-T, SpringSCM-T) are reproduced
exactly by `configs/worlds/{linear,nonlinear,spring}_draft.yaml`.

## Per-world contents

- `data.npz`: observations `x` `(N, T, k)`, world labels `y` (labels of the *true state*),
  `n_train`, innovations `eps`. Partially observed worlds also store the state and sensor exogenous
  variables (`exo_state`, `exo_obs_noise`, `exo_obs_mask`).
- `world_card.json` / `WORLD_CARD.md`: mechanism summary, label, designated structure (route
  source, distractor, latent confounder, regime driver), action space, observation model, replay
  disclosure, integrity results. Redacted for hidden worlds.
- `mechanism.npz`, `world_config.json` (generator family, factors, seed) and `integrity.json`
  live in the public directory for dev/val worlds and under `hidden/` for test/challenge worlds.

## Splits and visibility

dev and val worlds are visible. test and challenge mechanisms are hidden until final evaluation.
Challenge worlds come from an out-of-pool factor set (longer lags, oscillatory dynamics,
heteroscedastic/correlated noise, partial and nonlinear observation, missingness, irregular
sampling, latent confounding). Within each world, units are split into train (default 80%) and
test.

## Generation and quality control

Worlds are generated from named factor levels with seeds derived from the suite seed (or from a
secret for hidden splits). Every accepted world passes `check_world` (stability, graph, action
space, label balance, causal route by graph and by intervention, distractor, replay, disclosure).
It also differs by ≥ 5% in mechanism from every other world of its family. Rejected candidates
are recorded.

## Assumptions disclosure

Exact unit-level replay from observations requires the recorded mechanism to be the
data-generating SCM, positive lags, no unmeasured confounding, and innovations that are additive
(or scale-additive with a known scale) and unchanged by the action. Partially observed worlds break
identifiability from `x` on purpose. The benchmark's oracle still replays them exactly from the
stored exogenous variables, but methods that abduct from `x` alone cannot.

## Claim boundary

These worlds define causal truth by construction. They are evaluation oracles, not models of any
real domain. A result on them supports claims about synthetic-world certification only.
