# Technical note: native versus reconstructed actions

*Week 2 deliverable. Applies to protocol v0.1; see `docs/action_schema.md` for formats.*

## 1. The problem

Most temporal counterfactual methods output an **edited trajectory** `x̂`. A causal-action
explanation instead names an **intervention** `𝒜` and lets the data-generating process produce
the trajectory: `x^𝒜 = Replay(x, 𝒜, M*)`. The two objects coincide only if some action generates
the edit. To score a trajectory-only method on world-relative criteria, the benchmark must
therefore *choose* an action for it. That choice is the benchmark's hypothesis, not the method's
claim, and must never be reported as if it were.

## 2. Definitions

| Term | Meaning | Flag in records |
|---|---|---|
| **Native action** | Declared by the explainer as part of its output | `is_native_action: true`, reading `native` |
| **Reconstructed action** | Produced by a benchmark adapter from `(x, x̂)` | `is_native_action: false`, `adapter_name` |
| **Reading** | The rule that maps an explanation to an action | `reading` / `action_semantics` |
| **Fit error** | `E_replay(x̂, Replay(x, 𝒜, M*))` of a reconstructed action | `action.fit_error` |

An explainer that declares actions (`Explanation.declares_actions`) is evaluated **only** under its
own action. A trajectory-only explainer is evaluated under **every registered reading** in the
experiment config, and each reading is reported as its own row. Readings are never averaged.

## 3. Registered readings

All readings use the same change predicate `|x̂ − x| > tol` (default `1e-3`).

| Reading | Action | Intervention type | Reproduces `x̂` exactly when… |
|---|---|---|---|
| `first_change_atomic` | first edited slice, changed coordinates | hard `do()`, 1 timestep | `x̂` is the Pearl continuation of one atomic action |
| `first_changed_segment` | every changed entry in the first contiguous edited run | hard `do()` schedule | the edit after that run is mechanism-generated |
| `sparse_segmentation` | the **onset** of every per-channel edited run | hard `do()` at onsets | `x̂` is a sequence of atomic actions whose effects form separate runs |
| `change_point_pulse_fit` | one parametric pulse fit to the mechanism residual `r = x̂ − (F(x̂ lags) + ε)` | **soft** additive shift | the needed shift is a single rect or raised-cosine pulse |
| `full_changed_schedule` | every changed entry | hard `do()` everywhere | always (by construction) |

The hard readings are nested in the entries they impose:
first slice ⊆ segment onsets ⊆ full schedule. They span from "one action must explain
everything" to "everything is imposed". The pulse reading is the only soft one. It keeps every
structural equation and the factual innovations, and asks whether one smooth input shift explains
the edit. `change_point_pulse_fit` needs the world (`REQUIRES_WORLD`) because the residual is
defined through the mechanism.

**Do-complexity `D` is a property of the proposal, not of a reading.** It counts the timesteps
whose residual exceeds `tol`, i.e. the minimum hard schedule that reproduces `x̂` under Pearl
semantics. A reading's own schedule size (`action_timesteps`) is reported separately.

## 4. How to read the results

Toy run (`make toy`, linear world, 20 flip candidates). Rows are one trajectory-only baseline under
five readings plus native-action controls:

| method | reading | D | imposed/active steps | A | B | C | Δtraj | median E_rep |
|---|---|---|---|---|---|---|---|---|
| Wachter | first_change_atomic | 39.95 | 1.0 | 1.00 | 0.00 | 0.10 | 1.00 | 4.9e-2 |
| Wachter | sparse_segmentation | 39.95 | 23.3 | 1.00 | 0.05 | 0.05 | 0.95 | 4.7e-2 |
| Wachter | change_point_pulse_fit | 39.95 | 39.9 | 1.00 | 0.35 | 0.35 | 0.65 | 1.0e-1 |
| Wachter | first_changed_segment | 39.95 | 36.25 | 1.00 | 0.70 | 0.70 | 0.30 | 7.4e-4 |
| Wachter | full_changed_schedule | 39.95 | 39.65 | 1.00 | 1.00 | 0.95 | 0.00 | 6.9e-4 |
| PearlSCMRecourse (point) | native | 1.0 | 1.0 | 1.00 | 1.00 | 0.95 | 0.00 | 3.6e-17 |
| PearlSCMRecourse[interval] | native | 2.4 | 3.0 | 1.00 | 1.00 | 1.00 | 0.00 | 2.9e-17 |
| PearlSCMRecourse[control] | native | 3.95 | 4.0 | 1.00 | 1.00 | 1.00 | 0.00 | 3.2e-17 |

- The spread across readings is the finding. No compact reading reproduces the Wachter proposal.
  Δtraj falls only as the reading imposes more of the edit, and it reaches ~0 only when nearly
  every timestep is imposed (`D ≈ T`). The correct statement is "*not certified as action-grounded
  under any registered compact reading*", not "Wachter fails causal recourse". Wachter never
  declared an action.
- Small replay error does not imply agreement in outcome. Wachter edits nearly every timestep
  from `t = 0`, so its first edited run covers 36 of 40 steps. That reading's median `E_rep`
  (7e-4) is below the change tolerance, yet `Δtraj = 0.30`. The proposals sit on the classifier's
  decision boundary, and sub-tolerance replay differences flip the prediction. Read `E_rep`
  together with `A/B/C`, never as a substitute for them.
- The same proposal makes the pulse reading degenerate: the dominant residual run spans the whole
  horizon, so the fitted "pulse" is a 40-step shift with the largest error of all readings.
- Native controls have `E_rep ≈ 0` and `Δtraj = 0` because their proposal *is* the replay of
  their declared action.
- First-slice **CF-faith** is a proposal diagnostic tied to the atomic-`do()` reading. It is
  expected to be 0 for native interval and control actions, which act over several steps or
  through shifts. For native explainers, mechanism consistency is `E_replay` of the declared
  action. CF-faith remains relevant as a comparison with the draft.
- A soft native action can have `D > 1`: a 4-step control needs about 4 imposed timesteps when
  rewritten as hard `do()`s. `D` describes the trajectory's representation, not the action's cost.
  Report `action_timesteps` and the costs in `actions.costs` for that.

## 5. Validation and immutability

Every action, native or reconstructed, is validated against `world.action_space()` before replay:
actionable channels, value bounds, control existence and bounds, and horizon. Invalid actions are
**not replayed**. They are reported with readable violations
(`record["action_violations"]`, certificate `action_validation`) and counted in `n_invalid_action`,
separately from abstentions (`n_abstain`, no action) and from the scorable denominator.

A reading can be invalid even when the proposal looks harmless. In
`tests/test_controls/test_week2_controls.py`, the proposal edits only mutable channels `x1..x3`.
But `x0` (immutable) depends on them, so holding `x0` at its factual value requires a shift on
`x0`, and the pulse reading is rejected. Domain constraints therefore change which explanations
are acceptable, as the plan requires.

## 6. Hard versus soft interventions

| | Hard `do()` (point, schedule, interval) | Soft shift (pulse, high-level control) |
|---|---|---|
| Structural equation at the entry | replaced by the set value | kept; `+ shift` added |
| Factual innovation at the entry | discarded | kept |
| Downstream propagation | from the set value | from the shifted value |
| Typical use | observed-variable recourse, the draft's readings | treatments, dosing, dispatch, actuators |

Both are replayed with the same Pearl (or noiseless) noise semantics. High-level controls enter
through a world-declared linear input map with optional delay (`ControlChannel`). Nonlinear or
state-dependent actuation is deferred to the Week 4 state-space worlds and the Week 9 bridge.

## 7. Limitations

- Every reading depends on `tol`. `D`, the onsets and the pulse support move with it (tolerance
  sensitivity is Week 7).
- `sparse_segmentation` cannot detect a second action on a channel already perturbed by
  propagation from the first. Such an action is absorbed into one run and only the run's onset
  is imposed.
- `change_point_pulse_fit` fits a single pulse to the run with the largest residual energy.
  Multi-pulse edits receive the dominant pulse and a non-zero fit error.
- Reconstructed actions are evaluated only against observed variables and the declared controls.
  They do not cover domain-level actions the method may have had in mind.
