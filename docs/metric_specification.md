# Metric specification — v1.0 (Week 8)

All metrics are reported jointly and never collapsed into one score. **Population** says which
units a summary averages over:

- *proposal*: units with a proposal;
- *scorable*: units with a proposal and a valid, non-empty action under the reading;
- *CF-scorable*: proposals whose first change is at `t0 < T−1`.

`tol` is the change tolerance (default `1e-3`), `IQR_i` the training-split interquartile range of
channel `i`, and `σ` the soft-faith scale (`evaluation.soft_sigma`, default 1). Record keys are the
fields of `records.json`; summary keys are the fields of `summary.json`.

## Model-relative

| Metric | Definition | Range / better | Population | Record key → summary key | Code | Tests |
|---|---|---|---|---|---|---|
| Validity `V` | `1{h(x̂) = y*}` | {0,1} ↑ | proposal | `valid` → `validity` | `conventional.validity` | test_metrics |
| Logit-margin change | `m(x̂) − m(x)`, `m = z[y*] − z[1−y*]` | ℝ ↑ | proposal | `margin_change` | `conventional.logit_margin` | test_week3_metrics |
| L1 / L2 proximity | `‖x̂ − x‖_p` over the flattened `T×k` | ≥0 ↓ | proposal | `l1`, `l2` | `conventional.proximity_lp` | test_metrics |
| Normalized proximity | `(1/Tk) Σ |x̂−x| / (IQR_i+ε)` | ≥0 ↓ | proposal | `d_norm` | `conventional.normalized_proximity` | — |
| DTW | multivariate DTW, local cost mean `|·|/IQR`, path cost / T; optional band | ≥0 ↓ | proposal | `dtw` | `conventional.dtw_distance` | test_week3_metrics |
| Entry / channel / timestep sparsity | fraction left unchanged (`|x̂−x| ≤ tol`) | [0,1] ↑ | proposal | `unchanged_*` | `conventional.sparsity` | test_metrics |
| Interval sparsity | number of contiguous edited runs (per channel; per timestep) | ≥0 ↓ | proposal | `channel_segments`, `time_segments` | `conventional.interval_sparsity` | test_week3_metrics |
| Temporal smoothness | mean squared 2nd difference of `x̂ − x` | ≥0 ↓ | proposal | `temporal_smoothness` | `conventional.temporal_smoothness` | — |
| OOD / plausibility | Isolation-Forest score (fit on training split), higher = typical; also `ΔOOD = ood(x̂) − ood(x)` | ℝ ↑ | proposal | `ood_proposal`, `ood_factual` → `ood_proposal`, `ood_delta` | `conventional.OODScorer` | test_week3_metrics |
| Diversity | mean pairwise normalized L1 within a set of proposals for one unit | ≥0 | per unit, ≥2 proposals | — (seed sets, Week 5) | `conventional.diversity` | test_week3_metrics |
| Runtime | explainer seconds per unit | ≥0 ↓ | all | `runtime_seconds` | — | — |
| Failure rate | fraction with status ≠ `ok` (no flip, abstain, timeout, …) | [0,1] ↓ | all | → `failure_rate` | `evaluation.protocol.summarize` | test_week3_controls |

## Mechanism-relative

| Metric | Definition | Range / better | Population | Key | Code | Tests |
|---|---|---|---|---|---|---|
| Hard CF-faith (Pearl / noiseless) | first edited slice read as one `do()`; `1{mean L1 suffix residual < 1e-3}`; NaN if `t0 ≥ T−1` | {0,1,NaN} ↑ | CF-scorable | `cf_faith_{pearl,noiseless}_hard` | `mechanism.cf_faith` | test_replay, test_metrics |
| Soft first-slice CF-faith | `exp(−residual/σ)` | (0,1] ↑ | CF-scorable | `cf_faith_*_soft` | `mechanism.cf_faith` | — |
| Do-complexity `D` | timesteps whose structural residual exceeds `tol` (Pearl or noiseless) | 0…T ↓ | proposal | `D_pearl`, `D_noiseless` | `mechanism.do_complexity` | test_replay, test_metrics |
| Replay error `E_replay` | `(1/T) Σ_t mean_i |x̂ − x^{𝒜,*}|` | ≥0 ↓ | scorable | `replay_error_pearl` (summary: median) | `mechanism.replay_error` | test_replay |
| Normalized replay error | `mean |x̂ − x^{𝒜,*}| / (IQR_i+ε)` | ≥0 ↓ | scorable | `replay_error_pearl_norm` | `mechanism.normalized_replay_error` | test_week3_metrics |
| Soft faith (graded, scale-normalized) | `S_faith = exp(−E_replay_norm/σ)` | (0,1] ↑ | scorable | `soft_faith_replay` | `mechanism.soft_faith` | test_week3_metrics |
| Effect-profile correlation | Pearson corr. of `x̂ − x` and `x^{𝒜,*} − x` for `t ≥ t_first` | [−1,1] ↑ | scorable | `effect_corr` | `mechanism.effect_profile_correlation` | test_week3_metrics |
| Affected-support overlap | Jaccard of changed entries of `x̂` and `x^{𝒜,*}` | [0,1] ↑ | scorable | `support_overlap` | `mechanism.support_overlap` | test_week3_metrics |
| Retroactivity violation | fraction of the proposal's changes at `t < t_first(𝒜)` | [0,1] ↓ (0 = none) | scorable | `retroactive_fraction` | `mechanism.retroactivity` | test_week3_metrics |
| Prefix preservation | fraction of pre-action entries unchanged | [0,1] ↑ | scorable | `prefix_preservation` | `mechanism.prefix_preservation` | test_week3_metrics |
| Causal-path consistency | fraction of the proposal's changes reachable from the action's entries in the time-unrolled dependency graph | [0,1] ↑ | scorable | `causal_path_consistency` | `mechanism.causal_path_consistency` | test_week3_metrics |

## World-relative

| Metric | Definition | Population | Key | Code |
|---|---|---|---|---|
| `A` | `Pr[h(x̂) = y*]` | scorable | `proposal_model_success` | `outcome.abc_decomposition` |
| `B` | `Pr[h(x^{𝒜,*}) = y*]` | scorable | `replay_model_success` | 〃 |
| `C = V_world` | `Pr[ℓ*(x^{𝒜,*}) = y*]` | scorable | `replay_world_success` | 〃 |
| `Δtraj, Δoutcome, Δtotal` | `A−B`, `B−C`, `A−C` (exact identity) | scorable | `delta_*` | 〃 |
| Action-grounded success | `Pr[E_replay < 1e-3 ∧ ℓ*(x^{𝒜,*}) = y*]` | scorable | `action_grounded_success` | 〃 |
| Safe-world success | `Pr[Feasible ∧ ℓ*(x^{𝒜,*}) = y*]` | scorable | `safe_world_success` | 〃 |
| Denominators | flip candidates, scorable, abstentions, invalid actions | all | `n_flip_candidates`, `n_scorable`, `n_abstain`, `n_invalid_action` | 〃 |
| Timing | `t_label`, action `t_first`, label distance | scorable | `median_label_distance` | `protocol` |

## Actionability

| Metric | Definition | Key | Code |
|---|---|---|---|
| Validity of the action | passes `action_space` validation; violations listed | `invalid_action`, `action_violations` | `TemporalAction.validate` |
| Feasible rate | all configured constraints pass | `feasible` → `feasible_rate` | `actions.feasibility` |
| Costs | duration, magnitude, smoothness, budget, coordinates | `costs` | `actions.costs` |
| Action size | active timesteps of the action's schedule | `action_timesteps` → `mean_action_timesteps` | `DenseSchedule.n_timesteps` |

## Traceability

| Metric | Definition | Key / artifact | Code |
|---|---|---|---|
| Provenance completeness | fraction of required provenance fields present | `provenance_completeness` | `traceability.provenance_completeness` |
| Action round-trip | serialize → deserialize → identical | `action_roundtrip_ok` → `action_roundtrip_rate` | `traceability.action_roundtrip_ok` |
| Array integrity | saved proposal/replay hash to the recorded `proposal_hash`/`replay_hash` | `audit.json` | `evaluation.audit` |
| Replay reproducibility | fresh replay of the serialized action matches the saved replay (≤ 1e-9); world label, classifier decision and `E_replay` reproduce | `audit.json` | `evaluation.audit` |
| Config / dataset / mechanism hash consistency | resolved config, data and mechanism hash to the manifest values | `audit.json` | `evaluation.audit` |
| Objective and constraint logging | selection objective recorded; constraint status logged | `audit.json` | `evaluation.audit` |

## Robustness and uncertainty (Week 7; see `robustness.md`)

| Metric | Definition | Key / location | Code |
|---|---|---|---|
| Interventional target probability | P(ℓ*(X^A) = y* \| do(A), prefix), 16 fresh-noise rollouts in the true world | `p_world_interventional` | `protocol._interventional_target_prob` |
| Reported target probability | explainer's own P(h(X^A) = y*) under its mechanism | `p_target_reported` (`uncertainty.p_target_model`) | `_SCMRecourse._predicted_target_prob` |
| Calibration | mean \|reported − interventional\|; ECE vs realized outcomes | robustness report | `metrics.robustness.probability_calibration` |
| `p_robust` | fraction of plausible mechanisms under which the action's replay reaches the target | robustness report | `metrics.robustness.ensemble_outcomes` |
| Action stability | mean of onset agreement, variable Jaccard and outcome agreement under innovation perturbation | robustness report | `metrics.robustness.stability_pairs` |
| Tolerance profile | CF-faith, D and unchanged entries at τ ∈ {1e-5…1e-1} × channel std | robustness report | `metrics.sensitivity.tolerance_profile` |
| Ranking stability | Kendall τ of the method ranking by D vs τ = 1e-3 | robustness report | `metrics.sensitivity.ranking_stability` |
| CIs | hierarchical bootstrap, worlds → units, 95% percentile | robustness report | `metrics.aggregation.hierarchical_bootstrap` |

## Mechanism quality (benchmark-side, for R3/R4 mechanisms; `mechanisms.quality`)

| Metric | Definition | Range / better | Key (`run_info.json` → `mechanisms.<label>`) | Tests |
|---|---|---|---|---|
| Graph precision / recall / F1 / SHD | summary graph (any lag), self-loops excluded | [0,1] ↑ / ≥0 ↓ | `graph_precision`, `graph_recall`, `graph_f1`, `graph_shd` | test_discovery |
| Lagged precision / recall / F1 | over (target, source, lag) triples, cross edges, graphs padded to a common lag | [0,1] ↑ | `graph_lag_precision`, `graph_lag_recall`, `graph_lag_f1` | test_discovery |
| Lag accuracy | among correctly recovered summary edges, the fraction with exactly the true lag set | [0,1] ↑ | `graph_lag_accuracy` | test_discovery |
| One-step / rollout / replay error | see `mechanism_access.md` | ≥0 ↓ | `one_step_error`, `rollout_error`, `replay_error` | test_mechanisms |
| Discovery record | method, settings, units used, runtime, instantaneous links (deleted/added counts for corruption) | — | `discovery` | test_discovery |

## Statistics and comparisons (Week 8; protocol §8)

| Quantity | Definition | Population | Code |
|---|---|---|---|
| ITT world success | `1{scorable ∧ ℓ*(x^{𝒜,*}) = y*}`; no action = failure | all flip candidates | `statistical_analysis.world_success_itt` |
| ITT validity | `1{proposal ∧ h(x̂) = y*}` | all flip candidates | `proposal_success_itt` / `valid` |
| Action fraction | `action_timesteps / T` (median reported) | scorable | `study._cols` |
| CI | 95% percentile, hierarchical bootstrap worlds → classifiers → units → seeds | — | `bootstrap_draws`, `column_estimates`, `estimate` |
| Paired difference | per-unit (seed-averaged) `v_A − v_B` on shared units, mean with a CI over worlds → classifiers → units | units both methods saw | `paired_difference` |
| `d_z` | `mean(d) / sd(d)` over units | 〃 | 〃 |
| Probability of superiority | `P(d > 0) + ½ P(d = 0)` | 〃 | 〃 |
| Blocked difference | `stat(A) − stat(B)` for populations sharing worlds but not units | — | `blocked_difference` |
| Pareto front | methods not dominated in (world success ↑, action magnitude ↓) | non-control methods | `pareto_front` |

## Reporting rules

- Never report `A − C` without `A`, `B`, `C`, the reading, `D`, `E_replay`, feasibility, label
  distance and the denominators.
- First-slice CF-faith is a diagnostic tied to the atomic-`do()` reading. For native multi-step or
  soft actions, mechanism consistency is `E_replay` / soft faith of the declared action.
- `D` is observed-variable schedule complexity, not an action cost.
- Controls are reported as controls (`is_control`) and never ranked against explainers.
- Every reported estimate carries a CI. Method comparisons are paired differences on shared units;
  unpaired point estimates never support a claim that one method beats another.
