# Action schema — v0.1

Every action serializes to a JSON object with a `kind` field and round-trips exactly through
`twist.actions.action_from_dict`. Observed-variable actions compile to a **dense schedule**
`(mask, value, shift)`, each of shape `(T, k)`, which is the only thing replay consumes:

- `mask[t, i] = True` → hard `do(X^i_t = value[t, i])` (structural equation replaced);
- otherwise `X^i_t = F_i(Pa) + ε^i_t + shift[t, i]` (soft additive intervention when `shift ≠ 0`).

Channels are referenced by integer index; names come from `world.variable_names`.

## Action kinds

### `point` — `do(X^S_t = a_t)`

```json
{"kind": "point", "t": 30, "targets": [0, 2], "values": [1.25, -0.4]}
```

### `schedule` — `I = {(t, S_t, a_t)}`

```json
{"kind": "schedule", "entries": [[4, [0], [0.5]], [5, [0, 1], [0.7, 0.1]]]}
```

The `full_changed_schedule` reading of an edited trajectory has this form.

### `interval` — `do(X^S_{t:t+Δ} = a_{t:t+Δ})` over `[start, end)`

```json
{"kind": "interval", "start": 24, "end": 32, "targets": [3], "values": [0.15]}
```

`values` is one value per target (held constant) or a `Δ × |S|` nested list.

### `pulse` — soft additive pulse `a_t = φ(t; α, τ, Δ)`

```json
{"kind": "pulse", "start": 10, "duration": 6, "targets": [1], "amplitude": 0.5,
 "shape": "raised_cosine"}
```

Keeps the mechanism and the factual innovations and adds `φ` (`rect` or `raised_cosine`).

### `control` — high-level control `u_t ∈ U`

```json
{"kind": "control", "control": "u_drive", "start": 20, "end": 24, "values": [0.4, 0.4, 0.2, 0.0]}
```

The control must be declared by the world as a `ControlChannel` in its action space:

```yaml
action_space:
  actionable: [1, 2, 3, 4]        # optional; default every channel (x0 immutable here)
  lower: -5.0                     # optional per-channel (or scalar) bounds on do() values
  upper: 5.0
  controls:
    u_drive: {loadings: [0.0, 1.0, 0.5, 0.0, 0.0], delay: 1, lower: -3.0, upper: 3.0,
              unit: a.u., description: exogenous drive on x1 and x2}
```

`u_t` enters additively through the loadings `delay` steps later:
`x_{t+delay} = F(Pa) + ε + loadings · u_t`. It is a soft intervention that keeps the mechanism and
the factual innovations. Compile it with `action.compile(world.action_space())`; replay does this
automatically. `to_schedule(T, k)` raises because there is no world interface. Nonlinear or
state-dependent actuation is deferred to the Week 4 state-space worlds and the Week 9 bridge.

## Action plan (the auditable wrapper)

```json
{
  "semantics": "native_point_pearl",
  "is_native_action": true,
  "adapter_name": null,
  "fit_error": null,
  "target_variables": [0, 1, 2, 3, 4],
  "constraints": [],
  "action": {"kind": "point", "t": 30, "targets": [0, 1, 2, 3, 4], "values": [...]}
}
```

## Native and reconstructed actions

`is_native_action: true` means the explainer declared the action. For trajectory-only explainers
the benchmark applies **registered readings** (`twist.actions.trajectory_adapters`) and marks the
result `is_native_action: false` with `adapter_name` and `fit_error` (= `E_replay` of the
reconstructed action against the proposal):

| Adapter | Reading | Semantics string |
|---|---|---|
| `first_change_atomic` | first edited slice as one `do()` on its changed coordinates | `trajectory_inferred_first_slice` |
| `first_changed_segment` | first contiguous run of edited timesteps as a schedule | `trajectory_inferred_first_segment` |
| `full_changed_schedule` | every edited entry imposed | `trajectory_inferred_full_schedule` |
| `sparse_segmentation` | onset of every per-channel edited run imposed as `do()` | `trajectory_inferred_segment_onsets` |
| `change_point_pulse_fit` | one soft pulse fit to the mechanism residual of the dominant change (needs world) | `trajectory_inferred_pulse` |

Adapter signature: `adapter(x, x_hat, tol, world=None) -> ActionPlan | None`. See
[`technical_notes/native_vs_reconstructed_actions.md`](technical_notes/native_vs_reconstructed_actions.md)
for what each reading tests and how to interpret results across readings.

The two endpoint readings expose complementary ambiguities: a large first-slice error means one
inferred action cannot generate the remaining edits; a perfect full-schedule replay may require
imposing nearly every timestep (`D ≈ T`). A reconstructed action is never presented as the
method's claim.

## Validation and feasibility

- `TemporalAction.validate(ActionSpace)` returns readable violations: channel range, time range,
  non-actionable variables, value bounds, non-finite values. `check()` raises
  `ActionValidationError` with all of them. Controls are also checked for existence, bounds and
  horizon.
- The evaluator validates every action, native or reconstructed, before replay. Invalid actions
  are not replayed. They get `invalid_action: true` and `action_violations` in their record and
  certificate, and are counted in `n_invalid_action`, separately from abstentions.
- Feasibility constraints (`twist.actions.feasibility`) are evaluated on
  `(factual, schedule, replayed trajectory)` and return `ConstraintResult(passed, violation,
  message)`: `immutable_variables`, `amplitude_bounds`, `state_bounds`, `max_rate_of_change`,
  `intervention_budget`, `max_duration`, `min_lead_time`, `forbidden_combination`. They are
  configured in experiment YAML as `{name: ..., <params>}`.

## Costs versus do-complexity

`actions.costs` reports duration `Σ_t 1{a_t ≠ 0}`, magnitude `Σ_t ||a_t||_1`, smoothness
`Σ_t ||a_t − a_{t−1}||²`, and budget `Σ_t c(a_t)`, where `a_t` is `value − factual` for hard
interventions and `shift` for soft ones. These describe a declared action. **Do-complexity `D`**
describes a *proposal* (timesteps that must be imposed to reproduce it) and is never called an
action cost.
