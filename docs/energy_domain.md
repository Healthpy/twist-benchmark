# Energy domain bridge — semi-synthetic distribution-grid worlds (Week 9)

The bridge carries TWIST from abstract synthetic mechanisms to a domain with **explicit action
variables, physical limits and real temporal covariates**, while keeping a controlled oracle. The
family is `energy` (`twist.worlds.semi_synthetic`); the suite is `configs/suites/energy.yaml`
(`twist_energy_v1`); the study is `configs/experiments/bridge.yaml` (`twist bridge`).

## Claim boundary

The energy worlds are **semi-synthetic**:
- **Real:** the weather and demand covariates (Open Power System Data, Netherlands 2016–2019).
- **Simulated:** the transformer, battery, policy and demand-response dynamics. They are an
  explicit simulator inspired by distribution-grid physics, including a simplified IEC 60076-7
  thermal model.

Causal truth holds by construction **for the simulator only**:
- No result is evidence about a real feeder, battery or set of customers.
- The domain constraints are illustrative operating limits, not a grid code.
- The overload label is a class-balanced threshold on the simulated hot spot, not a protection
  setting.

The bridge tests whether TWIST's evaluation logic carries over to native domain actions and domain
feasibility. It does not validate the simulator against reality.

## Data: real covariates

| Channel source | Series | Unit in the world |
|---|---|---|
| demand | ENTSO-E Transparency, `NL_load_actual_entsoe_transparency` | ÷ mean over 2016–2019 (≈ 1) |
| temperature | MERRA-2 population-weighted, `NL_temperature` | `(°C − 10)/10` |
| irradiance | MERRA-2, `NL_radiation_direct_horizontal + NL_radiation_diffuse_horizontal` | kW/m² |

- **Provenance.** `scripts/fetch_energy_profiles.py` downloads the OPSD packages (time series
  2020-10-06, weather 2020-09-16; CC-BY 4.0). It writes `src/twist/data/energy/opsd_nl_2016_2019.npz`
  (35 064 hours, 270 kB) and a provenance JSON. Every load verifies the arrays' SHA-256.
- **Why irradiance and not NL solar generation.** PV is derived from irradiance because the OPSD NL
  solar-generation series is incomplete: it collapses to a few MW in 2019.
- **Scaling.** National demand is scaled to the feeder, so the feeder has the *shape* of national
  demand. That is a stated simplification.
- **Test data.** `profiles: synthetic` gives a self-contained stand-in with the same layout,
  used in tests.

**Units.** A unit is a window of consecutive real hours. After a 48-hour burn-in it starts at
00:00 UTC, so `t = 0…49` spans day 1 00:00 to day 3 01:00, and the day-2 evening peak falls at
`t ≈ 40–43`.

**How the real data enter the SCM.** The exogenous channels are roots whose innovations *are* the
real profiles:
- abduction recovers them exactly;
- replay holds them fixed, so an action never changes the weather or the demand forecast.

**Covariate overlap.** Units of one world are drawn from ~480 (winter) to ~1460 (all-year) days, so
train and test units can share calendar days. They then share covariates but never innovations.

## Mechanism

Hourly steps. Power is in per-unit of the transformer rating (1 p.u. = 10 MW). Every equation
uses inputs at `t − 1` (strictly lagged, `L = 1`).

| # | channel | meaning | structural equation |
|---|---|---|---|
| 0 | `temp_fcst` | next-hour ambient temperature | root (real) |
| 1 | `ghi_fcst` | next-hour irradiance | root (real) |
| 2 | `load_fcst` | next-hour demand | root (real) |
| 3 | `grid` | transformer loading `K` (negative = reverse flow) | `nd − (dis − chg) + r·backlog + η` |
| 4 | `soc` | battery state of charge | `(1 − δ)·soc − (dis/η_b − chg·η_b)/E + η` |
| 5 | `backlog` | demand deferred by DR | `(1 − r)·backlog + η` |
| 6 | `hot_spot` | winding hot-spot temperature, `(°C − 80)/20` | `a·hs + (1 − a)·(θ_ss − 80)/20 + η` |
| 7 | `stress` | insulation ageing stress | `0.9·stress + 0.1·softplus((θ_hs − 98)/6) + η` |

- **Forecast net demand:** `nd = c_load·load − c_pv·ghi`.
- **Default peak-shaving policy:** `p = P·tanh(g(nd − τ)/P)`. It discharges above the target `τ`
  and charges below it. Discharge is capped by the energy above SoC 0.1, and charge by the
  headroom below SoC 0.95. Efficiency is `η_b = 0.95` and self-discharge is `δ = 5·10⁻⁴/h`.
- **Thermal model:** steady-state hot spot `θ_ss = θ_amb + Δθ_r(1 + R·K²)/(1 + R)`, a first-order
  lag with time constant 2–4 h (`a = exp(−1/τ)`), and the IEC ageing law (rate doubles every
  6 K above 98 °C).
- **World parameters**, drawn from the seed within the factor levels:

| parameter | range |
|---|---|
| `c_load` | 0.8–1.0 p.u. |
| `c_pv` | low 0.2–0.4, high 0.7–1.0 p.u. per kW/m² |
| `P` | small 0.1–0.2, large 0.25–0.4 p.u.; `E = 2P` (fixed C-rate 0.5) |
| `g` | 1–3 |
| `τ` | 1.0–1.2 × `c_load` |
| `r` (rebound) | 0.1–0.3 /h |
| `Δθ_r` | 70–85 K |
| `R` | 4–8 |
| DR capacity | 0.15–0.3 p.u. |

**Graph.** grid ← load, ghi, soc, backlog; soc ← load, ghi, soc; backlog ← backlog; hot_spot ←
temp, grid, hot_spot; stress ← hot_spot, stress.

**Roles.**
- `hot_spot` is the label channel.
- `grid` is the route source.
- `stress` is the distractor: a downstream symptom of overheating that predicts the label but
  causes nothing.
- Self-dynamics are physical (storage, thermal inertia, deferred demand), so the integrity check
  allows lag-1 self-loops for this family, as for springs.

**Labels** (factor `energy_label`; thresholds are fitted at the training median of the score):
- `overload_event` — the hot spot exceeds the threshold at any hour of the second half;
- `evening_peak` — the same, restricted to day 2, 15–20 UTC;
- `thermal_exposure` — mean hot spot over the last 12 h.

The fitted threshold is reported in °C on each world card. In winter worlds it lies around
90–110 °C.

**Suite factors.**

| factor | levels |
|---|---|
| `pv_penetration` | low, high |
| `battery` | small, large |
| `season` | winter, all; summer only in the challenge pool, since summer windows rarely load the transformer |
| `energy_label` | see above |

## Action variables (domain action cards)

Each world's card is generated from its action space by
`twist.reporting.benchmark_cards.domain_action_card` and included in the bridge report.
Descendants come from the world graph.

| action variable | kind | physical meaning | unit | limits | delay | cost | side effects | causal descendants |
|---|---|---|---|---|---|---|---|---|
| `u_dispatch` | control: grid −u, soc −u/E | battery redispatch on top of the default policy (+ discharge / − charge) | p.u. (10 MW) | ±P | 0 h | ≈ 50 €/MWh (degradation) | drains or fills the battery, so less peak-shaving later; with the policy it can exceed the inverter rating | grid, soc, hot_spot, stress |
| `u_dr` | control: grid −u, backlog +u | demand-response curtailment | p.u. (10 MW) | [0, DR capacity] | 1 h (notice) | ≈ 150 €/MWh (compensation) | deferred demand returns at rate `r`: a rebound peak after the event | grid, backlog, hot_spot, stress |
| `grid` | `do()` on an observed channel | abstract loading set-point (load shedding or upstream redispatch by unspecified means) | p.u. | unbounded | 0 h | not modelled | cuts the loading's own mechanism | hot_spot, stress |

**Not actionable:**
- weather and demand forecasts (exogenous);
- `soc` and `backlog` (moved only by power flows);
- `hot_spot` (outcome);
- `stress` (symptom).

Observation-edit explanations that change these channels are rejected as invalid actions before
replay.

**What the controls do.**
- **Soft interventions.** Controls enter through a fixed linear input map. Replay keeps the
  mechanism and the factual innovations and only adds the control's effect.
- **Exact replay.** Replaying a control is deterministic. The bridge check re-replays every
  serialized control action on the reloaded world and compares replay hashes.
- **The policy does not see the control.** A control can therefore drain the battery below empty,
  overfill it, or exceed the inverter rating. These are physically impossible outcomes, and the
  domain constraints reject them on the replayed trajectory.

## Domain feasibility constraints

| id | generic class | rule | physical meaning |
|---|---|---|---|
| `soc_capacity` | `state_bounds` on `soc` | `0 ≤ soc ≤ 1` along the replay | the battery cannot store more than its capacity or deliver energy it does not have |
| `battery_c_rate` | `max_rate_of_change` on `soc` | `|Δsoc| ≤ 0.55` per hour | inverter rating: policy plus redispatch within C-rate 0.5 (÷ efficiency, + noise tolerance) |
| `dr_energy_cap` | `intervention_budget` on `backlog` | `Σ u ≤ 0.8` p.u.·h | contracted curtailment energy per event (8 MWh) |

- **Generic constraint:** `amplitude_bounds` (|action| ≤ 3).
- **Channel-restricted bounds.** Constraint classes take `channels` and an `id`, so a domain
  constraint is a configured instance of a generic one and is reported under its own name.
- **What OracleActionRecourse (R6) does.** It receives the constraints. It penalises violations of
  trajectory and budget constraints during optimisation (`lambda_feas`, default 100) and selects
  feasible candidates. With only projectable constraints (every pre-Week-9 configuration), the
  objective is unchanged.

## The bridge study

Rows of `configs/experiments/bridge.yaml`, run on 5 dev worlds × GRU × 60 units. The target is
"no overload" for units predicted to overload.

| rows | action form |
|---|---|
| Wachter, NativeGuide, NUN | trajectory explanations read as actions (first-slice adapter) |
| `SetPoint[R5 grid]` | abstract `do(grid)` set-point (native observed-variable action) |
| `Dispatch[R5 constraint-blind]`, `Dispatch[R6 constraint-aware]` | battery redispatch |
| `DR[R5 constraint-blind]`, `DR[R6 constraint-aware]` | demand response |
| NoActionAbstain | no action |

Each row is scored for acceptability (world target reached under oracle replay ∧ constraints pass)
twice: under the generic constraint alone, and under generic + domain constraints.

**Pre-declared checks** (`twist.evaluation.bridge_report`):
1. Every control row has scorable native control actions on every world.
2. Every scorable control action replays from its serialized form to the stored hash.
3. Among the constraint-blind dispatch's generic-acceptable explanations, the share a domain
   constraint rejects has a world-bootstrap CI with lower bound > 0.

**Hypotheses:**
- **HE1:** constraint-aware planning raises domain acceptability over the constraint-blind
  optimizer.
- **HE2:** the abstract set-point is acceptable more often than constraint-aware dispatch (the
  realizability gap of observation-level actions).

**Behaviour seen while building** (not results):
- A constraint-blind optimizer avoids an evening overload by **overcharging the battery** before
  the peak (SoC > 1) and spending energy it does not have. `soc_capacity` rejects it.
- **Demand response is a weak lever.** Rebound returns the deferred demand into the peak, and DR
  started too early *raises* the peak hot spot (≈ +9 °C at t = 30). Timing matters.

## Results (bridge_week9, 2026-10-13)

5 dev worlds × GRU × 60 units predicted to overload; report
`artifacts/bridge/bridge_week9/report/`. **3/3 checks pass.**

| row | acceptable, generic | acceptable, domain |
|---|---|---|
| constraint-blind dispatch (R5) | 0.47 | 0.00 (99% of its successes overfill or overdrain the battery) |
| constraint-aware dispatch (R6) | 0.07 | 0.05 |
| demand response (R5, R6) | 0.08–0.09 | 0.08–0.09 |
| abstract `do(grid)` set-point | 0.75 | 0.75 |
| trajectory baselines | 0 (all invalid: they edit weather, demand or thermal channels) | 0 |

- **HE1:** +0.047 [+0.020, +0.080], supported.
- **HE2:** +0.70 [+0.46, +0.92], supported.
- **Exploratory feasible ceiling:** exhaustive single-window dispatch averts 0.21 of overloads
  within SoC and C-rate limits, against 0.77 ignoring them. The battery's energy is the binding
  limit, and the R6 planner reaches about a quarter of the ceiling. See `decision_log.md`,
  Week 9, items 13–14.

## Domain review (task 5) — questionnaire

**Status: pending; this needs a domain expert** (distribution-grid operations or a DSO planning
engineer). Send the reviewer one world card, its action card, one demonstration figure from the
bridge report and this questionnaire. Record the answers in `decision_log.md`.

1. **Action meanings.** Are `u_dispatch` (redispatch on top of a peak-shaving policy) and `u_dr`
   (curtailment with 1 h notice and rebound) understandable as described? Is anything ambiguous?
2. **Constraints.**
   - Are SoC capacity, the C-rate limit and a per-event DR energy cap credible as the binding
     limits?
   - What is missing: reverse-power-flow limits, voltage limits, minimum SoC reserves, DR event
     frequency, ramping, or N−1 security?
3. **Thermal outcome.** Is a hot-spot event over the day-2 evening a sensible target for
   "avoid overload"? Should the threshold be physical (e.g. 98 °C / 120 °C) rather than
   class-balanced?
4. **Certificates.** Would an action certificate (control, window, MW values, cost, which
   constraints pass, replayed loading/SoC/hot spot) help an operator decide? What would they need
   added (uncertainty, forecast error, alternatives)?
5. **Failure modes.** Which failures matter most: overcharging or overdraining the battery,
   rebound peaks, actions that rely on forecast details, or actions that edit uncontrollable
   inputs (weather, demand)?
6. **Realism.** Which simplifications would most change conclusions: national demand shape, linear
   control input map, policy blind to the control, single thermal time constant?
