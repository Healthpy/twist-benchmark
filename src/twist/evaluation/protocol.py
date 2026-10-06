"""Evaluation protocol: score explanations against the world oracle under registered readings.

For each explanation the evaluator records model-relative metrics of the proposal, then for each
action reading (the declared action for native explainers; the registered diagnostic adapters for
trajectory-only explainers) replays the action through the world and records replayed classifier
and world outcomes, replay error, costs, feasibility and a provenance record. Readings are never
averaged together.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from twist.actions.costs import action_costs
from twist.actions.feasibility import Constraint, check_feasibility
from twist.actions.schemas import ActionPlan
from twist.actions.trajectory_adapters import ADAPTERS, REQUIRES_WORLD
from twist.classifiers.models import TorchClassifier
from twist.config import EvaluationConfig
from twist.evaluation.result_schema import Explanation, ProvenanceRecord
from twist.metrics import conventional as conv
from twist.metrics import mechanism as mech
from twist.metrics.mechanism import cf_faith, do_complexity, replay_error
from twist.metrics.outcome import abc_decomposition
from twist.metrics.traceability import action_roundtrip_ok, provenance_completeness
from twist.replay.pearl_replay import to_dense
from twist.utils.serialization import hash_arrays
from twist.worlds.base_scm import TemporalSCM

NATIVE = "native"


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def evaluate_explanations(
    world: TemporalSCM,
    classifier: TorchClassifier,
    factuals: np.ndarray,
    explanations: list[Explanation],
    cfg: EvaluationConfig,
    constraints: list[Constraint] | None = None,
    iqr: np.ndarray | None = None,
    ood: conv.OODScorer | None = None,
    exogenous: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Return one record per (explanation, reading).

    Records carry the proposal and replay arrays under the private keys ``_proposal`` and
    ``_replay`` so the runner can persist them for audits; strip ``_*`` keys before writing JSON.

    ``exogenous`` holds the factual units' exogenous variables (``state``, ``eta``, and for
    partially observed worlds the sensor noise and masks); replay, world outcomes, feasibility and
    costs use them through ``world.replay_units`` / ``world.outcome``. Mechanism metrics that need
    abduction from the observations (CF-faith, ``D``, causal-path consistency, the pulse reading)
    are computed only when ``world.assumptions["exact_abduction"]`` holds and are NaN otherwise.
    """
    constraints = constraints or []
    N, T, k = factuals.shape
    exact = bool(
        world.assumptions.get("exact_abduction", world.assumptions.get("exact_replay", True))
    )
    if exogenous is None:
        if not exact:
            raise ValueError(f"world {world.world_id} needs the units' exogenous variables")
        exogenous = {"state": factuals}
    states = exogenous["state"]
    k_state = world.k_state
    t_label = world.label.site(T)
    has_prop = np.array([e.proposed_trajectory is not None for e in explanations])
    X_hat = np.stack(
        [
            e.proposed_trajectory if e.proposed_trajectory is not None else factuals[i]
            for i, e in enumerate(explanations)
        ]
    )

    # ---- proposal-level (reading-independent) metrics
    h_prop = classifier.predict(X_hat)
    h_fact = classifier.predict(factuals)
    if exact:
        f_pearl = cf_faith(
            world, factuals, X_hat, "pearl", cfg.faith_tol, cfg.change_tol, cfg.soft_sigma
        )
        f_nl = cf_faith(
            world, factuals, X_hat, "noiseless", cfg.faith_tol, cfg.change_tol, cfg.soft_sigma
        )
        D_pearl = do_complexity(world, factuals, X_hat, "pearl", cfg.change_tol).astype(float)
        D_nl = do_complexity(world, factuals, X_hat, "noiseless", cfg.change_tol).astype(float)
    else:
        t0_obs, _ = conv.first_change(factuals, X_hat, cfg.change_tol)
        f_pearl = f_nl = {
            "hard": _nan(N),
            "soft": _nan(N),
            "t0": t0_obs,
            "scorable": np.zeros(N, bool),
        }
        D_pearl = D_nl = _nan(N)
    sp = conv.sparsity(factuals, X_hat, cfg.change_tol)
    l1, l2 = conv.proximity_lp(factuals, X_hat, 1), conv.proximity_lp(factuals, X_hat, 2)
    dnorm = conv.normalized_proximity(factuals, X_hat, iqr) if iqr is not None else _nan(N)
    scale = iqr if iqr is not None else np.ones(k)
    targets = np.array([e.target for e in explanations])
    margin_fact = conv.logit_margin(classifier.logits(factuals), targets)
    margin_prop = conv.logit_margin(classifier.logits(X_hat), targets)
    dtw = conv.dtw_distance(factuals, X_hat, scale, cfg.dtw_window) if cfg.compute_dtw else _nan(N)
    segs = conv.interval_sparsity(factuals, X_hat, cfg.change_tol)
    smooth = conv.temporal_smoothness(factuals, X_hat)
    ood_fact = ood.score(factuals) if ood is not None else _nan(N)
    ood_prop = ood.score(X_hat) if ood is not None else _nan(N)

    # ---- assign readings: the declared action for action-declaring explainers, otherwise the
    # registered diagnostic adapters (never attributed to the method)
    readings: dict[str, dict[int, ActionPlan | None]] = defaultdict(dict)
    reading_errors: dict[tuple[str, int], list[str]] = {}
    for i, e in enumerate(explanations):
        for r in [NATIVE] if e.declares_actions else cfg.readings:
            if r == NATIVE:
                plan = e.action
                if plan is not None and plan.semantics.endswith("@observed"):
                    plan = world.to_state_plan(plan)  # planned on observed channels
                readings[r][i] = plan
            else:
                if not has_prop[i]:
                    readings[r][i] = None
                elif not exact and r in REQUIRES_WORLD:
                    readings[r][i] = None
                    reading_errors[(r, i)] = [
                        f"reading {r!r} needs exact abduction from observations, which this "
                        "partially observed world does not allow"
                    ]
                else:
                    plan = ADAPTERS[r](factuals[i], X_hat[i], cfg.change_tol, world=world)
                    readings[r][i] = world.to_state_plan(plan)

    # ---- invalid actions are rejected with a violation report and never replayed
    space = world.action_space()
    violations: dict[tuple[str, int], list[str]] = dict(reading_errors)
    for reading, plan_map in readings.items():
        for i, plan in plan_map.items():
            if plan is not None and (v := plan.action.validate(space)):
                violations[(reading, i)] = v

    records: list[dict[str, Any]] = []
    for reading, plan_map in readings.items():
        applies = sorted(plan_map)
        sub_plans = [None if (reading, i) in violations else plan_map[i] for i in applies]
        sub_exo = {name: arr[applies] for name, arr in exogenous.items()}
        x_rep_sub, s_rep_sub = world.replay_units(sub_exo, sub_plans, "pearl")
        x_rep_nl_sub, _ = world.replay_units(sub_exo, sub_plans, "noiseless")
        x_rep = dict(zip(applies, x_rep_sub, strict=True))
        h_rep = dict(zip(applies, classifier.predict(x_rep_sub), strict=True))
        y_world = dict(zip(applies, world.outcome(s_rep_sub), strict=True))
        s_rep = dict(zip(applies, s_rep_sub, strict=True))
        e_rep = dict(zip(applies, replay_error(X_hat[applies], x_rep_sub), strict=True))
        e_rep_nl = dict(zip(applies, replay_error(X_hat[applies], x_rep_nl_sub), strict=True))
        p_int = _interventional_target_prob(world, sub_exo, sub_plans, targets[applies], cfg)
        p_int_map = dict(zip(applies, p_int, strict=True))
        e_norm = dict(
            zip(
                applies, mech.normalized_replay_error(X_hat[applies], x_rep_sub, scale), strict=True
            )
        )
        ood_rep = dict(
            zip(
                applies,
                ood.score(x_rep_sub) if ood is not None else _nan(len(applies)),
                strict=True,
            )
        )
        plans = plan_map
        for i in applies:
            e, plan = explanations[i], plans[i]
            action_violations = violations.get((reading, i), [])
            invalid = bool(action_violations)
            scorable = bool(has_prop[i] and plan is not None and not invalid)
            sched = to_dense(plan, T, k_state, space) if plan is not None and not invalid else None
            if scorable and plan is not None and not plan.native:
                plan.fit_error = float(e_rep[i])
            feas = (
                check_feasibility(constraints, states[i], sched, s_rep[i])
                if sched is not None
                else None
            )
            t_first = sched.first_time if sched is not None else None
            x, xh, xr = factuals[i], X_hat[i], x_rep[i]
            graded: dict[str, float] = {
                "replay_error_pearl_norm": float("nan"),
                "soft_faith_replay": float("nan"),
                "effect_corr": float("nan"),
                "support_overlap": float("nan"),
                "retroactive_fraction": float("nan"),
                "prefix_preservation": float("nan"),
                "causal_path_consistency": float("nan"),
            }
            if scorable and sched is not None:
                graded = {
                    "replay_error_pearl_norm": float(e_norm[i]),
                    "soft_faith_replay": float(mech.soft_faith(e_norm[i], cfg.soft_sigma)),
                    "effect_corr": mech.effect_profile_correlation(x, xh, xr, t_first or 0),
                    "support_overlap": mech.support_overlap(x, xh, xr, cfg.change_tol),
                    "retroactive_fraction": mech.retroactivity(x, xh, t_first, cfg.change_tol),
                    "prefix_preservation": mech.prefix_preservation(x, xh, t_first, cfg.change_tol),
                    "causal_path_consistency": mech.causal_path_consistency(
                        world, x, xh, sched.active, cfg.change_tol
                    )
                    if exact
                    else float("nan"),
                }
            rec: dict[str, Any] = {
                "explanation_id": e.explanation_id,
                "explainer": e.explainer,
                "reading": reading,
                "factual_id": e.factual_id,
                "target": e.target,
                "status": e.status.value,
                "has_proposal": bool(has_prop[i]),
                "scorable": scorable,
                "invalid_action": invalid,
                "action_violations": action_violations,
                "native_action": e.native_action,
                "action_semantics": plan.semantics if plan is not None else "none",
                "mechanism_access_regime": e.mechanism_access_regime,
                "mechanism": e.mechanism,
                # proposal / model-relative
                "h_factual": int(h_fact[i]),
                "h_proposal": int(h_prop[i]),
                "valid": bool(has_prop[i] and h_prop[i] == e.target),
                "l1": float(l1[i]),
                "l2": float(l2[i]),
                "d_norm": float(dnorm[i]),
                "unchanged_entries": float(sp["unchanged_entries"][i]),
                "unchanged_channels": float(sp["unchanged_channels"][i]),
                "unchanged_timesteps": float(sp["unchanged_timesteps"][i]),
                "channel_segments": int(segs["channel_segments"][i]),
                "time_segments": int(segs["time_segments"][i]),
                "dtw": float(dtw[i]),
                "temporal_smoothness": float(smooth[i]),
                "margin_factual": float(margin_fact[i]),
                "margin_proposal": float(margin_prop[i]),
                "margin_change": float(margin_prop[i] - margin_fact[i]),
                "ood_factual": float(ood_fact[i]),
                "ood_proposal": float(ood_prop[i]),
                "ood_replay": float(ood_rep[i]) if scorable else float("nan"),
                "runtime_seconds": float(e.runtime_seconds),
                "explainer_seed": e.explainer_seed,
                "queries": e.diagnostics.get("queries"),
                "error": e.diagnostics.get("error"),
                "t0": int(f_pearl["t0"][i]),
                "t_label": t_label,
                "cf_faith_pearl_hard": float(f_pearl["hard"][i]),
                "cf_faith_pearl_soft": float(f_pearl["soft"][i]),
                "cf_faith_noiseless_hard": float(f_nl["hard"][i]),
                "cf_faith_noiseless_soft": float(f_nl["soft"][i]),
                "cf_faith_scorable": bool(f_pearl["scorable"][i]),
                "D_pearl": float(D_pearl[i]),
                "D_noiseless": float(D_nl[i]),
                # action / replay
                "action_t_first": t_first,
                "label_distance": (t_label - t_first) if t_first is not None else None,
                "action_timesteps": sched.n_timesteps if sched is not None else 0,
                "h_replay": int(h_rep[i]),
                "world_replay": int(y_world[i]),
                "replay_error_pearl": float(e_rep[i]) if scorable else float("nan"),
                "replay_error_noiseless": float(e_rep_nl[i]) if scorable else float("nan"),
                "p_world_interventional": float(p_int_map[i]) if scorable else float("nan"),
                "p_target_reported": e.uncertainty.get("p_target_model"),
                "feasible": feas.passed if feas is not None else None,
                "feasibility": feas.to_dict() if feas is not None else None,
                "costs": action_costs(sched, states[i]) if sched is not None else None,
                "action": plan.to_dict() if plan is not None else None,
                "action_roundtrip_ok": action_roundtrip_ok(plan),
                **graded,
                "proposal_hash": hash_arrays(xh) if has_prop[i] else None,
                "replay_hash": hash_arrays(xr) if scorable else None,
                "_proposal": xh if has_prop[i] else None,
                "_replay": xr if scorable else None,
            }
            rec["attribution"] = attribute_outcome(rec)
            rec["provenance"] = build_provenance(e, plan, rec, constraints)
            rec["provenance_completeness"] = provenance_completeness(rec["provenance"])[
                "completeness"
            ]
            records.append(rec)
    return records


ATTRIBUTIONS = (
    "success",
    "outcome_disagreement",
    "replay_mismatch",
    "optimization",
    "no_action",
    "invalid_action",
    "no_proposal",
)


def _interventional_target_prob(
    world: Any,
    exo: dict[str, np.ndarray],
    plans: list[Any],
    targets: np.ndarray,
    cfg: EvaluationConfig,
) -> np.ndarray:
    """P(ℓ*(X^𝒜) = y* | do(𝒜), factual prefix) by fresh-noise rollouts in the true world.

    Unit-level Pearl replay answers "what would have happened"; this answers "what would happen if
    the action were enacted now", the quantity a reported target probability should match.
    """
    n = len(plans)
    S = int(cfg.interventional_samples)
    if S <= 0 or n == 0:
        return np.full(n, np.nan)
    from twist.replay import replay_interventional_distribution

    state_world = getattr(world, "state", world)
    rng = np.random.default_rng(0)
    out = np.full(n, np.nan)
    live = [i for i, p in enumerate(plans) if p is not None]
    if live:
        z = replay_interventional_distribution(
            exo["state"][live], [plans[i] for i in live], state_world, S, rng
        )  # (S, B, T, k)
        y = world.outcome(z.reshape(-1, *z.shape[2:])).reshape(S, len(live))
        out[live] = (y == np.asarray(targets)[live][None]).mean(axis=0)
    return out


def attribute_outcome(rec: dict[str, Any]) -> str:
    """Where a unit's world-level failure arises (plan §Week 6 acceptance criterion).

    * ``optimization``: the explainer did not reach the target under its own model (A fails);
    * ``replay_mismatch``: it did, but the true world does not reproduce the proposal closely
      enough to keep the classifier's decision (B fails). For estimated mechanisms this is
      mechanism estimation error; for trajectory readings it is missing action grounding;
    * ``outcome_disagreement``: the replay keeps the classifier's decision but the world label
      disagrees (C fails while B holds);
    * ``success``: the replayed action achieves the world target (C holds).
    """
    if not rec["has_proposal"]:
        return "no_proposal"
    if rec["invalid_action"]:
        return "invalid_action"
    if not rec["scorable"]:
        return "no_action"
    t = rec["target"]
    if rec["world_replay"] == t:
        return "success"
    if rec["h_proposal"] != t:
        return "optimization"
    if rec["h_replay"] != t:
        return "replay_mismatch"
    return "outcome_disagreement"


def build_provenance(
    e: Explanation, plan: ActionPlan | None, rec: dict[str, Any], constraints: list[Constraint]
) -> dict[str, Any]:
    failure_modes = []
    if not rec["has_proposal"]:
        failure_modes.append("no_proposal")
    if plan is not None and not plan.native:
        failure_modes.append("action_reconstructed_by_benchmark_adapter")
    if rec["scorable"] and rec["replay_error_pearl"] > 1e-3:
        failure_modes.append("proposal_not_generated_by_action_under_pearl_replay")
    if rec["scorable"] and rec["world_replay"] != e.target:
        failure_modes.append("replayed_action_misses_world_target")
    if rec["feasible"] is False:
        failure_modes.append("infeasible_action")
    if rec["invalid_action"]:
        failure_modes.append("invalid_action")
    return ProvenanceRecord(
        explanation_id=e.explanation_id,
        factual_id=e.factual_id,
        target_outcome=e.target,
        native_action=e.native_action,
        action_semantics=rec["action_semantics"],
        action=rec["action"],
        selection_objective=e.selection_objective,
        active_constraints=[c.name for c in constraints],
        candidate_count=e.candidate_count,
        selection_rank=e.selection_rank,
        mechanism_access_regime=e.mechanism_access_regime,
        proposed_metrics={
            k: rec[k]
            for k in (
                "h_proposal",
                "valid",
                "l1",
                "unchanged_entries",
                "t0",
                "cf_faith_pearl_hard",
                "cf_faith_noiseless_hard",
                "D_pearl",
                "D_noiseless",
            )
        },
        replayed_metrics={
            k: rec[k]
            for k in (
                "h_replay",
                "world_replay",
                "replay_error_pearl",
                "replay_error_noiseless",
                "action_t_first",
                "label_distance",
                "feasible",
            )
        },
        uncertainty=e.uncertainty,
        known_failure_modes=failure_modes,
    ).model_dump()


def _stat(values: np.ndarray, mask: np.ndarray, fn=np.nanmean) -> float:
    v = values[mask]
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size else float("nan")


def summarize(records: list[dict[str, Any]], faith_tol: float = 1e-3) -> list[dict[str, Any]]:
    """Per (explainer, reading) summary. Always reports A, B, C, D, readings and denominators.

    Proposal-level statistics use units with a proposal; replay/action statistics use the
    scorable population; CF-faith uses its own scorable subset (``n_cf_faith_scorable``).
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r["explainer"], r["reading"])].append(r)
    rows = []
    for (explainer, reading), rs in sorted(groups.items()):

        def arr(key: str, rs: list[dict[str, Any]] = rs) -> np.ndarray:
            return np.array([np.nan if r[key] is None else r[key] for r in rs], dtype=float)

        scorable = np.array([r["scorable"] for r in rs])
        has = np.array([r["has_proposal"] for r in rs])
        cf_s = np.array([r["cf_faith_scorable"] for r in rs])
        targets = np.array([r["target"] for r in rs])
        feasible = np.array([bool(r["feasible"]) for r in rs])
        faithful = np.nan_to_num(arr("replay_error_pearl"), nan=np.inf) < faith_tol
        abc = abc_decomposition(
            arr("h_proposal"),
            arr("h_replay"),
            arr("world_replay"),
            targets,
            scorable,
            feasible=feasible,
            replay_faithful=faithful,
            invalid=np.array([r["invalid_action"] for r in rs]),
        )
        ld = arr("label_distance")
        everyone = np.ones(len(rs), bool)
        status_ok = np.array([r["status"] == "ok" for r in rs])
        rows.append(
            {
                "explainer": explainer,
                "reading": reading,
                "is_control": bool(rs[0].get("is_control", False)),
                # model-relative (units with a proposal)
                "validity": _stat(arr("valid"), has),
                "margin_change": _stat(arr("margin_change"), has),
                "l1": _stat(arr("l1"), has),
                "d_norm": _stat(arr("d_norm"), has),
                "dtw": _stat(arr("dtw"), has),
                "unchanged_entries": _stat(arr("unchanged_entries"), has),
                "unchanged_channels": _stat(arr("unchanged_channels"), has),
                "unchanged_timesteps": _stat(arr("unchanged_timesteps"), has),
                "channel_segments": _stat(arr("channel_segments"), has),
                "temporal_smoothness": _stat(arr("temporal_smoothness"), has),
                "ood_proposal": _stat(arr("ood_proposal"), has),
                "ood_delta": _stat(arr("ood_proposal") - arr("ood_factual"), has),
                "runtime_seconds": _stat(arr("runtime_seconds"), everyone),
                "failure_rate": float(1.0 - status_ok.mean()) if len(rs) else float("nan"),
                "status_counts": dict(Counter(r["status"] for r in rs)),
                "attribution": dict(Counter(r["attribution"] for r in rs)),
                "queries": _stat(arr("queries"), everyone),
                # mechanism-relative
                "cf_faith_pearl": _stat(arr("cf_faith_pearl_hard"), cf_s),
                "cf_faith_noiseless": _stat(arr("cf_faith_noiseless_hard"), cf_s),
                "cf_faith_pearl_soft": _stat(arr("cf_faith_pearl_soft"), cf_s),
                "n_cf_faith_scorable": int(cf_s.sum()),
                "D_pearl": _stat(arr("D_pearl"), has),
                "D_noiseless": _stat(arr("D_noiseless"), has),
                "replay_error_pearl": _stat(arr("replay_error_pearl"), scorable, np.nanmedian),
                "replay_error_pearl_norm": _stat(
                    arr("replay_error_pearl_norm"), scorable, np.nanmedian
                ),
                "soft_faith_replay": _stat(arr("soft_faith_replay"), scorable),
                "effect_corr": _stat(arr("effect_corr"), scorable),
                "support_overlap": _stat(arr("support_overlap"), scorable),
                "retroactive_fraction": _stat(arr("retroactive_fraction"), scorable),
                "prefix_preservation": _stat(arr("prefix_preservation"), scorable),
                "causal_path_consistency": _stat(arr("causal_path_consistency"), scorable),
                "ood_replay": _stat(arr("ood_replay"), scorable),
                # action / timing
                "median_label_distance": _stat(ld, scorable, np.nanmedian),
                "mean_action_timesteps": _stat(arr("action_timesteps"), scorable),
                # traceability
                "provenance_completeness": _stat(arr("provenance_completeness"), everyone),
                "action_roundtrip_rate": _stat(arr("action_roundtrip_ok"), everyone),
                **abc,
            }
        )
    return rows
