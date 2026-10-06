# Decision log

## Week 1 — 2026-10-05 (repository setup)

Decisions that deviate from, or refine, `plan.md`:

1. **Package namespace.** The plan lists modules directly under `src/` (`src/actions`,
   `src/utils`, ...). They live under `src/twist/` instead so they install as one package
   (`import twist.actions`) and do not shadow generic top-level names such as `utils`.
2. **`twist.classifiers` added.** The plan has `configs/classifiers/` but no source package for
   classifiers; GRU/LSTM models and training live in `src/twist/classifiers/`.
3. **Configuration.** Pydantic-validated YAML (`twist.config`) rather than Hydra; experiment
   configs reference component configs by path under `configs/`.
4. **Experiment tracking.** Append-only JSONL manifest (`artifacts/manifest.jsonl`), no MLflow/W&B.
5. **Dense schedule as the replay contract.** Every observed-variable action compiles to
   `(mask, value, shift)`; one backend-agnostic recursion serves NumPy (oracle) and torch
   (differentiable controls), so controls and evaluation can't disagree about semantics.
6. **Root convention.** The first `L` stored timesteps are exogenous roots (`ε_t := x_t`), making
   abduction exact for every stored timestep.
7. **Linear rescaling option.** `rescale: exact` scales a sparse VAR up to the target spectral
   radius. Under the draft's `cap` rule, a small sparse world (k=5) had ρ≈0.26, so single-slice
   actions at `t0 ≥ T/4` had numerically zero effect on the terminal label and the controls could
   not flip it. The draft presets keep `cap`; the toy world uses `exact`. This belongs in the
   Week-4 "stability" generator factor.
8. **Spring integrator.** Symplectic Euler uses the updated velocity in the position step
   (`p_{t+1} = p_t + dt·v_{t+1}`), which is what makes it symplectic and keeps the system linear.
   Check this against the code behind the draft's results.
9. **Nonlinear hidden activation.** The draft doesn't specify the hidden activation of the per-node
   MLP; tanh is used here.
10. **Precision and threads.** float64 end to end. Torch intra-op threads default to 1
    (`TWIST_NUM_THREADS`): on the shared interactive node, 48 threads made GRU training ~60×
    slower and the first toy run was killed.
11. **Wachter baseline.** The toy run uses an in-house Wachter-style reference implementation;
    the draft's third-party `cfts` adapters arrive in Week 5.

## Week 2 — 2026-10-05 (action representations and replay oracle)

1. **Actions compile against the action space.** `TemporalAction.compile(space)` is the single
   entry point used by replay and the evaluator, so world-dependent actions such as high-level
   controls go through the same path as `do()` actions.
2. **High-level controls are linear input channels.** A world declares `ControlChannel`s
   (loadings, delay, bounds, unit, description). A control is a soft shift that keeps the
   mechanism and the innovations. This covers the plan's fourth action form in the existing
   observed-state worlds without inventing domain dynamics. Nonlinear actuation is deferred to the
   Week 4 state-space worlds and the Week 9 bridge.
3. **`sparse_segmentation` = per-channel segment onsets.** The plan only names this adapter. It is
   defined as the reading that sits between first-slice and full-schedule: impose the first entry
   of every edited run and let the mechanism generate the rest. It needs no world.
4. **`change_point_pulse_fit` works in residual space.** It fits the shift the mechanism would
   need (`x̂ − F(x̂ lags) − ε`), not the raw edit. A pulse that exactly matches the residual
   therefore reproduces the proposal under Pearl replay. It needs the world.
5. **Invalid actions are rejected, not dropped.** Validation runs before replay. Violations go into
   the record, provenance and certificate, and are counted in `n_invalid_action`, separately from
   abstentions.
6. **Structural controls cover every native form.** `PearlSCMRecourse` and `NoiselessSCMRecourse`
   take `action_form ∈ {point, interval, pulse, control}`. They respect the actionable set and
   project onto value and control bounds every step, so their actions always validate. The default
   is still the draft's point control, and non-point variants are named `Name[form]`.
7. **Manifest `explainer` is the display name** (e.g. `PearlSCMRecourse[interval]`), with the class
   in `explainer_class`.
8. **Observation from the toy run.** A reconstructed reading can have median `E_replay` below the
   change tolerance and still lose 30% of target outcomes (`Δtraj = 0.30`). Wachter proposals sit
   on the decision boundary. This supports the reporting rule that `E_replay` never replaces
   `A/B/C`.

## Week 3 — 2026-10-05 (metrics, certificates and structural controls)

1. **Graded faith is scale-normalized.** `S_faith = exp(−E_replay_norm/σ)` uses the replay error
   divided by the per-channel training IQR, so it is comparable across channels and families. The
   draft's first-slice soft score is kept separately (`cf_faith_*_soft`).
2. **Causal-path consistency uses the dependency graph of `F`, not just the recorded `G`.**
   The nonlinear family's `decay_i·x^i_{t−1}` term is a self-dependence missing from the recorded
   graph, so `TemporalSCM.dependency_graph()` adds it. Otherwise legitimate propagation would be
   flagged as causally unsupported.
3. **Controls.**
   - `FullScheduleOracle` imposes the nearest training trajectory whose *world* label is the target,
     so it gets zero replay error with a full schedule.
   - `PlausibleButMechanismWrongEdit` reuses the SCM-recourse optimizer under
     `perturb_mechanism(world)`: same graph, strengths perturbed by 0.5 × their spread, and
     abduction under the wrong mechanism, so the prefix and noise texture look real.
   - `perturb_mechanism` is the first piece of the Week-6 misspecification generator.
4. **Adversarial control changed after the first sanity run.** With a fixed budget
   (ε = 0.25 std per entry) it flipped only 65% / 30% / 15% of units (linear / nonlinear / spring),
   failing the pre-declared "high validity" check. It became a *minimal-budget* attack: an ε ladder
   [0.1, 0.25, 0.5, 1.0] std, keeping the smallest budget that flips, which is the standard
   definition of an adversarial example. **The check threshold (0.8) was not changed.**
5. **Spring adversarial check still fails (0.50 at ε ≤ 1 std).** A diagnostic showed this is not
   the optimizer: 4× more steps change nothing. The spring classifier's flip candidates sit far
   from the boundary (median logit margin −6.5), and the label channel has std 5.6. Flipping needs
   2–4 std per entry (85% at 2σ, 100% at 4σ). Such edits are no longer "small", so the ladder was
   **not** extended on my own initiative.
   **Resolved (user decision):** the ladder was extended to [0.1, 0.25, 0.5, 1, 2, 4] std. The
   control tests "model-valid but not mechanism-consistent", not edit size. The chosen budget is
   recorded per unit (`selection_objective.epsilon_selected`), so large edits stay visible and the
   control is not described as a "small perturbation" control.
6. **DTW bug caught by a hand-constructed test.** `window=0` was silently clamped to 1, so it
   never computed the lockstep (diagonal) alignment. Fixed.
7. **Traceability audit.** Runs now save proposals and replays per (explainer, reading) under
   `artifacts/explanations/<run_id>/`, record their hashes, and write `run_info.json` paths.
   `twist audit` re-replays every certificate from its serialized action and checks integrity,
   reproducibility (≤ 1e-9) and hashes. All three sanity runs pass.
8. **Certificates have a Markdown rendering** structured as the plan's seven audit questions
   (what action, which mechanism, which trajectory, target achieved, constraints, uncertainty,
   why selected).

## Week 4 — 2026-10-05 (multi-world synthetic benchmark suite)

1. **Partially observed worlds compose rather than subclass.** `ObservedSCM` wraps any exact state
   world with an `ObservationModel`. A unit-level world API (`simulate_units`, `replay_units`,
   `outcome`, `to_state_plan`) is shared by every world, so the pipeline (runner, evaluator,
   audit) is the same for exact and partially observed worlds.
2. **The oracle replays from stored exogenous variables, not from abduction.** The benchmark
   generated every unit, so the unit-level counterfactual is defined by its stored `(z, η, ν,
   mask)` even when `x` does not identify them. Abduction-based metrics are NaN on such worlds, and
   methods that abduct from `x` are skipped with a recorded reason (runner `skipped_explainers`).
   They are not dropped silently.
3. **World outcomes are labels of the true state.** On partially observed worlds the classifier
   sees noisy, partial observations, so `Δoutcome` (classifier vs world rule) can now be non-zero.
   In a probe run, adversarial and Wachter edits gave `Δoutcome = −0.30 / +0.10`.
4. **Heteroscedastic noise keeps exactness.** The exogenous variable is the standardized `η`, and
   the state-dependent scale is part of the mechanism (`step = F + s·η`), so abduction stays
   exact (tested to 1e-12).
5. **Regime switching uses observed drivers, so replay stays exact.** A truly latent regime is
   obtained by hiding the exogenous driver channel with the observation model.
6. **Spring stability needed a tether.** Damping alone leaves the spectral radius at exactly 1:
   the Laplacian's zero mode plus uncoupled particles make positions random-walk. Contractive and
   weakly stable springs therefore add a tether κ. The draft's undamped spring stays at γ = κ = 0.
7. **Integrity checks were corrected by the first probe.** (a) A causal route is tested from any
   action time before the label site, since path lengths are constrained without self-loops.
   (b) The empirical route effect is the best of a few probes. (c) A real generator bug: the
   regime driver weights included the distractor, which made it causal. Fixed.
8. **Family diversity is an acceptance rule.** The first dev build had identical spring worlds
   (4 particles leave about 3 free coupling choices, and the draft uses one stiffness). It also
   had near-duplicate statespace worlds, because a distance on `F` is dominated by the near-identity
   integrator. Fixes: per-edge heterogeneous stiffness and ≥ 5 particles in factor-built springs,
   a distance on the increment over persistence, and rejection of candidates < 5% from an accepted
   world of the same family. The 0.05 threshold was set before the first build and not changed.
9. **Rejection sampling is pre-declared and recorded.** Dev suite: 4 of 44 candidates rejected
   (3 weak causal routes in saturating nonlinear/regime worlds, 1 too similar).
10. **Hidden splits.** Mechanisms, configs, seeds and roles of test/challenge worlds live under
    `hidden/`. Without `TWIST_SUITE_SECRET` their seeds derive from the public suite seed, so
    hiding is a no-peeking discipline. The dataset card states which applies.
11. **The statespace standard pool uses noisy observation only.** Identity observation would make
    statespace worlds exact duplicates of the other families. That level remains available for
    controlled ablations.
12. **Thread policy in the library.** `run_paired_experiment` now applies `TWIST_NUM_THREADS`
    itself. A direct library call without the CLI had made training 50× slower on the shared node.

## Week 5 — 2026-10-05 (classifiers, baseline adapters, budgets, pilot)

1. **Third-party baselines come from cfts, pinned.** Wachter (gradient), COMTE, M-CELS,
   CONFETTI, Native Guide and Latent-CF are adapted from the cfts library (MIT) at commit
   `86c9e6e2`, installed through the optional `baselines` extra rather than vendored. Upstream code
   is not modified. TSCausalCF is listed as *planned*; the authors will add it.
2. **Variant choices.** Where cfts has several variants of a method, the one closest to the
   original paper/repo was adapted:
   - COMTE: `comte_cf`, distractor channel swaps, not the gradient variants;
   - M-CELS: `mcels_reimp.mcels_generate`, the faithful reimplementation;
   - CONFETTI: `confetti_nsga_cf`, NSGA-II;
   - Wachter: `wachter_gradient_cf`, as in the draft;
   - Native Guide: `native_guide_uni_cf`, GradientShap windows; it handles multivariate input
     despite its name.

   Library default hyperparameters are used and recorded in `configs/explainers/`.
3. **Retrieval and generative baselines.** NUN (nearest reference trajectory predicted as the
   target) is in-house; Latent-CF is the generative/plausibility baseline, with its autoencoder
   trained once per run and shared across units.
4. **The in-house Wachter is renamed `WachterReference`.** `Wachter` now means the cfts method,
   as in the draft.
5. **Reference data is an R1 resource.** Retrieval-based methods need training trajectories and
   labels, so they run under R1 (observational data), never R0. The context builder enforces this.
6. **Import workaround.** `cfts/__init__.py` imports all 35 methods, including `pyts`, whose
   numba functions compile at import without caching (~55 s per process, every process). The
   adapters register a path-only stub of the `cfts` package and import only the needed submodules
   (~8 s). Upstream code is unchanged, and an already imported real package is used as is.
7. **Budgets and failures.** Every third-party model call goes through `BudgetedModel` (query and
   time accounting, float32 channel-first view of the float64 classifier). Every unit ends in a
   standardized output or a structured failure record. Third-party exceptions become
   `method_error` with the message kept, never a crash and never a silent drop.
8. **Classifier suite.** GRU, LSTM, TCN, a small transformer, and an interpretable logistic
   regression on engineered temporal features. The latter is fit by scikit-learn and loaded into a
   differentiable torch module, so gradient-based explainers can query it. On terminal-threshold
   labels it is near-perfect, because one of its features is the label channel's last value.
   This is the "trivial classifier" risk from the risk register, and a reason to use the
   non-terminal label families.
9. **Classifier robustness** is reported under a variance-matched uniform-innovation shift (same
   mechanism and label), as in the draft's shift table.
10. **Pilot design fixed before running.** Worlds: the first dev world of each draft family with a
    threshold label (`linear_dev_001`, `nonlinear_dev_000`, `spring_dev_000`). Classifiers: GRU and
    TCN. Methods: 5 cfts baselines + Latent-CF + NUN + structural controls. 50 factual units;
    3 seeds for stochastic explainers (deterministic ones run once).
11. **The pilot runs on a compute node.** Two local attempts on the interactive node were killed
    after ~18–30 CPU-minutes (`BrokenProcessPool`). There were no OOM kills (peak 4.3 GB of the
    8 GB cap); the session cgroup is capped at 4 CPUs / 8 GB, and long CPU-heavy processes appear
    to be killed there. With the user's approval the pilot was submitted as SLURM job 27621612
    (`scripts/slurm/multiworld.sbatch`, 16 rome cores). Multi-world runs are now resumable per
    task (a completed run with the same config hash is reused).
12. **Pilot outcome** (job 27621612, 41.5 min, 4,800 explanations;
    `artifacts/pilots/pilot_week5/report/`). All three Week-5 acceptance checks pass.
    - Every (run, method, seed) group is complete, with no crashes, timeouts or budget overruns.
      Failures are structured `no_target_flip`: LatentCF 282/900, M-CELS 233/900.
    - The model-vs-world gap is reproduced for all seven baselines under the first-slice reading.
      Δtotal = +0.27 (NativeGuide) to +0.87 (LatentCF), and almost all of it is Δtraj
      (|Δoutcome| ≤ 0.05), the draft's pattern.
    - The positive controls pass. Seeds are stable: the across-seed std of C is ≤ 0.06.
    - These are pooled pilot point estimates, with no uncertainty yet (Week 8).
13. **Upstream behaviour noted, not changed.** cfts' `wachter_gradient_cf` initializes from a
    random reference trajectory rather than the factual. It often starts already in the target
    class and stops after ~3 queries, so it behaves like retrieval (D = T). This is consistent
    with the draft's very large Wachter L1. `WachterReference` (in-house) starts from the factual.

## Week 6 — 2026-10-05 (mechanism-access ladder)

1. **One optimizer, varying information.** Graph-only (R2), estimated (R3/R4), oracle (R5) and
   oracle + action model (R6) recourse share the Pearl control's optimizer (`_SCMRecourse`, now
   with mechanism / action-space / feasibility hooks). Differences in outcome are attributable to
   information, not to algorithm choice.
2. **R2 has no mechanism.** A graph-only explainer cannot propagate, so its proposal is the
   imposed edit itself. It acts only on ancestors of the outcome channels (the outcome node is part
   of the graph knowledge at R2+). In the probe its edits never flipped the classifier (A = 0),
   while the true world propagated them (B = 0.375). That is the gap this rung exists to show.
3. **R3 = known graph + family, unknown parameters.** Graph-masked least squares for linear-family
   worlds, a graph-masked neural transition otherwise. For non-linear families this is an
   approximation of "family known", and it is documented as such.
4. **R4 estimators.** BIC-selected Lasso VAR (structure-aware; the support gives Ĝ) and a neural
   transition (linear skip + MLP; no structure). The Lasso support threshold was raised from
   1e-3 to 1e-2 after a test showed BIC-Lasso keeps tiny spurious coefficients (precision 0.6,
   recall 1.0 → F1 1.0); true coefficients are ≥ 0.06. LassoCV was replaced by LARS-BIC: it is
   faster, and the scikit-learn API change broke `n_alphas`.
5. **Estimates are worlds.** Estimated and misspecified mechanisms implement the world interface,
   so abduction, replay and differentiable recourse run on them unchanged. Only the runner scores
   them against the true world (graph recovery, one-step, rollout and **replay** error).
6. **Misspecification is controlled and anchored.** Seven kinds; level 0 reproduces the world
   exactly (tested). Replay error can be non-monotone in level for saturating mechanisms (observed
   for nonlinear coefficient perturbation), because a very wrong saturated mechanism responds less
   to interventions.
7. **Estimated recourse on partially observed worlds.** It plans in observed coordinates; native
   actions tagged `@observed` are mapped to state channels by the evaluator. `to_state_plan` now
   maps interval, pulse and control actions too; this gap was caught by a test. Oracle-mechanism
   posterior abduction remains unimplemented (Week 7, uncertainty).
8. **Failure attribution** splits every unit into success / outcome disagreement / replay
   mismatch / optimization / no action. That makes the plan's "where do failures arise" criterion
   checkable per method.
9. **Ladder run on SLURM with the user's approval** (job 27630864). It uses the Week-5 pilot
   worlds, the GRU classifier, 50 units, and one seed (all ladder explainers are deterministic).
10. **First ladder run (job 27630864) — oracle check failed; start times widened.** With start-time
    candidates {0.5, 0.75}·T, the R6 oracle control reached C = 0.53 (< 0.8 pre-declared).
    Attribution showed the failures were *optimization*, not mechanism: replay mismatch 0,
    outcome disagreement 0.01, and C | A = 0.91–1.00 per world. In the contractive linear world
    (ρ ≤ 0.6) the oracle flipped only 8% of units, because an action 10–24 steps before the label
    is attenuated by ≈ 0.6¹⁰–0.6²⁴. That is the draft's intervention-to-label distance effect.
    With the user's decision, every recourse rung (R2–R6, for comparability) now uses
    {0.5, 0.75, 0.9}·T. The threshold was not changed. The candidates are capped at 0.9·T, not the
    0.96·T the lead-time constraint would allow: at T = 50 a 3-step interval from t = 48 would
    impose the label site directly and trivialize the control. The first run is kept under
    `artifacts/ladders/access_ladder_week6/superseded_t0_mid/`.
11. **Second ladder run (job 27631133).** With the widened start times, the oracle reached
    C = 0.97, but the check still failed on feasibility (0.91 < 0.95). All 13 "infeasible" oracle
    actions exceeded the amplitude bound by 4e-16 to 9e-16: a floating-point bug in the
    feasibility check, since the projection clamps exactly to the bound and `(x + θ) − x ≠ θ`. Bound
    constraints now use a 1e-9 tolerance (regression-tested). With the user's decision, the ladder
    was rerun fresh (job 27651224); the second run is kept under
    `artifacts/ladders/access_ladder_week6/superseded_feas_roundoff/`.
12. **Resume now requires the same code.** Completed runs were reused on a matching config hash
    alone, so a rerun after a code fix would have silently returned stale results. Runs now record
    a fingerprint of the `twist` source tree, and reuse needs both hashes to match (tested).
13. **Final ladder (job 27651224; `artifacts/ladders/access_ladder_week6/report/`).** All three
    Week-6 checks pass:
    - **oracle control:** C = 0.97 with feasible rate 1.00;
    - **misspecification degrades estimated recourse:** C falls from 0.98 to 0.56 / 0.57 /
      0.94 at level 1 for coefficients / delay / deleted edges;
    - **every unit is attributed.**

    World success climbs the ladder:

    | Rung | R0/R1 baselines | R2 graph-only | R3 masked fit | R4 Lasso | R4 neural | R5 oracle | R6 oracle |
    |---|---|---|---|---|---|---|---|
    | C | 0.13–0.21 | 0.65 | 0.96 | 0.94 | 0.92 | 0.98 | 0.97 |

    - **Baselines fail by replay mismatch** (0.78–0.85), i.e. missing action grounding.
    - **R2 fails differently:** largely outcome disagreement (0.23), because its imposed edits
      fool the classifier without the world agreeing.
    - **Estimated rungs:** small replay-mismatch shares (0.03–0.05).
    - **The median replay error for coefficient misspecification is non-monotone in level, while
      C declines.** This is the saturation effect noted in item 6.

## Week 7 — 2026-10-06 (robustness, uncertainty, action sensitivity)

1. **What shifts mean here.** A shift changes the deployment world only. The classifier, reference
   data, estimated mechanisms and oracle knowledge stay those of the base world, and units are
   drawn from and replayed in the shifted world. Explanations are computed with old knowledge and
   enacted in a changed world. Retraining shifts swap the evaluation classifier instead.
2. **Shifted populations are persisted** with their world, and manifests carry the evaluated
   dataset hash, so shifted, retrained and perturbed runs pass the traceability audit (tested).
   The misspecified-mechanism wrapper became saveable for that reason.
3. **Stability perturbs innovations, not observations.** Adding noise to `x` would leave the world
   manifold. Perturbing every innovation by 0.05 × std and regenerating the unit gives a nearby,
   coherent unit in any world family (the unperturbed case reproduces exactly; tested).
4. **Two probabilities.** Unit-level replay is deterministic. Uncertainty about enacting an action
   now is the interventional target probability (fresh noise after the action, true world).
   Action-native explainers report their own estimate under their mechanism as their uncertainty
   field `u`; the report scores that report's calibration against the truth.
5. **`p_robust` ensemble:** the world plus seven strength-perturbed copies (0.25 × spread), same graph
   and label. This is a neighbourhood of plausible mechanisms, not a posterior; Bayesian posteriors
   over SCMs remain a nice-to-have in the plan.
6. **Pre-declared thresholds:** model-valid A ≥ 0.8; brittle = world-success drop ≥ 0.10 under a
   mechanism shift or the ensemble; replay-error sensitivity bands 0.01 / 0.05. Shifts, amounts,
   ensemble and tolerance grid are fixed in `experiments/robustness.yaml` before running.
7. **Scope.** M-CELS is excluded from the robustness study for cost (~7 s/unit across 33 tasks).
   Oracle posterior abduction on partially observed worlds is still not implemented; observation
   shifts measure the cost of assuming exact observations.
8. **Run on SLURM with the user's approval** (job 27654834): 3 pilot worlds × (base, perturbed, 9
   shifts), GRU, 50 base / 30 shifted units, 2 seeds for stochastic baselines in base runs.
9. **Seed-overwrite bug, found by the first robustness run (job 27654834).** Saved proposal arrays
   were keyed by `explainer__reading` only, so with several seeds each seed overwrote the previous
   one. The first-seed proposals of every stochastic method were lost (tolerance check 5/9;
   no seed variability). Arrays are now keyed per seed and the report reads method names from the
   records; a regression test checks that every seed's arrays survive and the audit passes. The
   Week-5 pilot was affected too: only seed 503's arrays survive, so its certificates for seeds
   501/502 cannot be audited. Its dashboard was computed from records and is unaffected. The
   robustness study was rerun with the user's approval (job 27655335); the first run is kept
   under `superseded_seed_overwrite/`.
10. **Brittleness verdict: kept as pre-declared, made CI-aware in Week 8 (user decision).** The
    Week-7 rule (point-estimate drop ≥ 0.10) flagged 8/9 methods in the first run. Shifted runs
    have 30 units and CI half-widths of about 0.15–0.3, so many of those flags are within noise.
    Week 8 replaces the rule with a paired-bootstrap one (flag only when the drop's 95% interval
    lies above 0.10).
11. **Final robustness study** (job 27655335; `artifacts/robustness/robustness_week7/report/`).
    All three Week-7 checks pass: 99/99 method×condition rows have CIs, 9/9 methods have tolerance
    profiles, and the brittleness verdict discriminates. The run is deterministic: every number
    shared with the superseded run is identical.
    - **Mechanism shifts hurt action-native recourse most.** The oracle drops from 0.97 to
      0.56 / 0.64 / 0.42 (coefficients / delay / edge deletion), the R4 Lasso recourse similarly.
      `p_robust` gives the same picture (oracle 0.69, Lasso 0.74, Pearl 0.56). Plans that exploit a
      precise mechanism are the most model-valid and the most mechanism-brittle.
    - **Innovation, observation and decision shifts move C by ≤ 0.1** for most methods.
    - **Tolerance:** method ranking by D is stable (Kendall τ ≥ 0.91 for τ ≤ 1e-2·std; 0.78 at 1e-1).
      CF-faith of non-Pearl methods rises only at τ ≥ 1e-2·std, i.e. their edits are not near-faithful.
    - **Reported target probabilities** are within 0.07–0.13 of the true interventional probability.
    - **Seed variability:** cfts Wachter's proposals vary maximally across seeds (diversity ≈ 1.0),
      a consequence of its random-reference initialisation.

## Week 8 — 2026-10-06 (statistical protocol, benchmark validity, pilot study, protocol v1.0)

1. **The brittleness verdict is CI-aware** (decided in Week 7). A method is flagged only when the
   95% interval of a world-success drop lies above 0.10:
   - mechanism shifts use a world-blocked bootstrap, because base and shifted units differ;
   - the ensemble drop `C − p_robust` is paired per unit.

   On the Week-7 runs this flags ActionRecourse[R4 lasso], OracleActionRecourse[interval] and
   PearlSCMRecourse (drops 0.58–0.66, CI lower bounds ≥ 0.45) and clears the other six. The report
   was regenerated from the existing runs; no rerun was needed.
2. **Statistical protocol** (`evaluation.statistical_analysis`):
   - a four-level hierarchical bootstrap (worlds → classifiers → units → seeds);
   - paired unit-level differences, with seeds averaged within a unit, plus `d_z` and the
     probability of superiority;
   - world-blocked differences for unpaired populations.

   Method comparisons use **ITT world success**: a missing action is a failure, so denominators
   are equal across methods. `A, B, C` keep the scorable denominator. To keep 4-level CIs over
   ~10⁴ records cheap, bootstrap index draws are computed once per method and reused for every
   metric (`bootstrap_draws`, `column_estimates`).
3. **Validity pairs are built from oracle point actions strengthened to reach the target.** The
   oracle recourse optimiser stops at the classifier's decision boundary. On a logistic classifier
   it flipped the model without the world for 15/15 units. Each action is therefore amplified by
   the smallest factor in {1, 1.25, 1.5, 2, 3, 4} for which both the classifier and the world reach
   the target. Units with no such factor are excluded. The pair constructions themselves are
   unaffected.
4. **Validity criteria use the spec's thresholds.** "Not mechanism-consistent" means
   `E_replay ≥ 1e-3`, the action-grounded threshold. A first draft used 0.01; mechanism-wrong edits
   on linear worlds have `E_replay` of 0.002–0.01, which made the check fail for a reason unrelated
   to plausibility.

   The "low Δtotal with low A" check uses under-powered oracle actions (0.1×). Their proposal is
   their own replay, so A = B. A near-no-op adversarial edit was tried first, but it yields no
   scorable action.

   Pair B (equal one-entry edits at the label site, on the label channel vs the distractor)
   applies only to threshold labels, where the label site is a single entry.
5. **Validity report** (`artifacts/validity/validity_week8/`): 5 exact dev worlds, GRU, 60 units.
   All nine tests pass.
   - **A:** validity held identical; the varied replay error differs by −0.0054 [−0.013, −0.0004].
   - **B:** proximity is identical; world success differs by 0.86 [0.78, 0.92].
   - **C:** replay error is identical; feasibility differs by 1.0.
   - **D:** world success is identical; magnitude differs by −2.9 [−5.3, −0.6].
   - **E:** ΔOOD is −0.002 [−0.006, 0.002]; causal-path consistency differs by 0.82 [0.58, 1.0].
   - **Discriminant:**
     - exact replay spans action sizes 1–50 of T = 50;
     - 74% of mechanism-wrong edits are plausible yet inconsistent;
     - the trajectory baseline has validity 1.00 but safe-world success 0.13;
     - under-powered actions have A = 0.11, C = 0.21.
6. **BLAS threads are pinned.** `configure_threads` now also limits the numpy/scikit-learn
   BLAS/OpenMP pools (threadpoolctl). Two local validity runs were killed on the interactive node
   until `OMP_NUM_THREADS=1` was set; the pool size now follows `TWIST_NUM_THREADS`.
7. **The pilot study uses a fresh suite** (`suites/pilot.yaml`, seed 2108, `twist_pilot_v1`):
   - 5 dev worlds per family, 25 in total, all passing integrity;
   - 1500 units per world, so every world × classifier has ≥ 100 flip candidates.

   The worlds are independent of the twist_dev_v1 worlds used for tuning in Weeks 5–7. Design: GRU
   and TCN, 100 units, seeds 501–503, five baselines (cfts Wachter, COMTE, NativeGuide, LatentCF,
   NUN) and six controls (Pearl, full-schedule oracle, R6 oracle recourse, abstain, adversarial,
   mechanism-wrong). CONFETTI and M-CELS are excluded for cost. Readings are the first slice
   (method reading) and the full schedule (sensitivity). The feasibility constraint is amplitude
   ≤ 3. Min-lead-time was dropped because label sites vary across worlds. Acceptance checks and
   control conditions were fixed in code before the run.
8. **The study ran on SLURM with the user's approval** (job 27658557, rome, 32 CPUs).
9. **Code fingerprints are taken when a run starts, once per process.** They were computed when
   `run_info.json` was written, from the files on disk. Edits made while a long job runs (here,
   formatting and type fixes in the reporting modules during job 27658557) therefore relabelled
   runs whose workers had loaded the earlier code. The fingerprint is now cached per process at
   the first run's start. Report rebuilds (`--report-only`) accept same-config runs made by
   other code, with a warning. Resume logic is unchanged and still needs matching code. The
   only code changed during the job was reporting, typing and the fingerprint itself; the
   simulation, explainer and metric code was not.
10. **Pilot study results** (job 27658557, 49 min on 32 cores, ≈ 26 SBU;
    `artifacts/studies/study_week8/report/`). There are 50 runs and 141k records, with
    5000 flip candidates per method (4000 for oracle controls, which skip the state-space worlds).

    Pre-declared checks: **2/4 pass.** Both failures are reported as they are; the criteria were
    not changed after the run.
    - **Paired conclusions: pass.** All 10 method pairs are compared on 5000 shared units over
      25 worlds, and every headline cell has a CI.
    - **Draft replication: pass, in 25/25 worlds and all 5 families.** The trajectory baselines'
      model–world gap under the first-slice reading is Δtotal 0.50–0.90 per world. Every CI
      excludes 0.
    - **Positive controls: fail.** Pearl (E_replay ≈ 1e-17, CF-faith 1.00) and the full-schedule
      oracle (E_replay 0) pass. The R6 oracle recourse misses `C ≥ 0.8` on the CI rule:
      C = 0.80 [0.67, 0.91].
      - It succeeds on exposure and onset labels (C 0.91–1.00).
      - On threshold and multi labels in linear and regime worlds, A ≈ 1 but C = 0.36–0.56. It
        optimizes the classifier under the true mechanism, and there the classifier and the
        world disagree (attribution: outcome disagreement 0.15).
      - nonlinear_dev_003 is an optimization failure (A = 0.09).
    - **Negative controls: fail.** Abstain, adversarial (CF-faith 0, C 0.17) and the
      full-schedule oracle (median action = T) fail as intended. The mechanism-wrong edit
      (strengths perturbed by 0.5 × spread) is not reliably caught: E_replay = 0.012 [0.002,
      0.14], CF-faith 0.09 [0.02, 0.18].
      - Nonlinear and spring worlds: E_replay 0.15–0.64, CF-faith 0.
      - Linear and regime worlds: E_replay 2e-4–4e-3. The perturbation barely changes how a
        late action propagates there, so the control is too weak in those worlds.

    Method findings (paired, ITT world success; no unpaired claims):
    - **NativeGuide** beats every other baseline by +0.23 to +0.31, with every CI excluding 0.
      Its segment-swap edits are short (D ≈ 14 imposed timesteps vs ≈ 50 for the other
      baselines). Why its first edited slice then replays to the target more often has not been
      analysed yet.
    - **LatentCF** is below NUN (−0.06 [−0.11, −0.02]) and below Wachter (−0.08 [−0.13, −0.04]),
      driven by 31% no-flip failures.
    - **Wachter, COMTE and NUN** are indistinguishable (all CIs include 0).
    - **Pareto front:** COMTE, LatentCF, NUN and NativeGuide; Wachter is dominated.
    - **Under the full-schedule reading,** baseline world success rises (COMTE 0.91), but only by
      imposing about T timesteps, as in Weeks 1–5.
11. **The study report bootstrap was vectorized after the run.** Recursive Python draws cost about
    46 s per method per 1000 draws, and the report took about 25 of the job's 49 minutes. The
    draws now sample each level of the flattened hierarchy with numpy (0.3 s per 1000 draws). The
    resampling scheme is the same, checked against the reference bootstrap in a test.
