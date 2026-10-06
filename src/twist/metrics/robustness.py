"""Robustness metrics (Week 7): shift degradation, factual-perturbation stability, tolerance
sensitivity and multi-mechanism robustness ``p_robust(A) = mean_j 1{ℓ^{M_j}(x^A) = y*}``."""

from __future__ import annotations

from typing import Any

import numpy as np


def shift_validity_retention(base: float, shifted: float, min_base: float = 0.30) -> float:
    """Shift-VR = shifted/base validity; suppressed (NaN) when base validity < ``min_base``."""
    return shifted / base if base >= min_base else float("nan")


def p_robust(world_outcomes: np.ndarray, target: int) -> float:
    """Fraction of candidate mechanisms under which the replayed action achieves the target."""
    return float(np.mean(np.asarray(world_outcomes) == target))


# ---------------------------------------------------------------- Week 7: stability, ensembles
def unit_index(factual_id: str) -> int:
    return int(factual_id.rsplit("/", 1)[1])


def stability_pairs(
    base: list[dict], perturbed: list[dict]
) -> dict[tuple[str, str], dict[str, float]]:
    """Explanation stability under a small factual perturbation, per (explainer, reading).

    Records are paired by explainer, reading, explainer seed and unit index. Per pair:
    *onset* agreement (same action start), *variables* (Jaccard of intervened variables),
    *cost* (1 − relative difference of action magnitude), *outcome* (same world outcome on replay),
    *target* (both proposals reach / miss the target the same way). ``action_stability`` is the
    mean of onset, variables and outcome.
    """
    index = {
        (r["explainer"], r["reading"], r.get("explainer_seed"), unit_index(r["factual_id"])): r
        for r in perturbed
    }
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for b in base:
        p = index.get(
            (b["explainer"], b["reading"], b.get("explainer_seed"), unit_index(b["factual_id"]))
        )
        if p is None:
            continue
        d = acc.setdefault(
            (b["explainer"], b["reading"]),
            {k: [] for k in ("onset", "variables", "cost", "outcome", "target", "n")},
        )
        d["n"].append(1.0)
        d["target"].append(float(b["valid"] == p["valid"]))
        if b["scorable"] and p["scorable"]:
            d["onset"].append(float(b["action_t_first"] == p["action_t_first"]))
            vb = set(b["action"]["target_variables"]) if b["action"] else set()
            vp = set(p["action"]["target_variables"]) if p["action"] else set()
            d["variables"].append(len(vb & vp) / len(vb | vp) if vb | vp else 1.0)
            cb, cp = (
                (b["costs"] or {}).get("magnitude", 0.0),
                (p["costs"] or {}).get("magnitude", 0.0),
            )
            d["cost"].append(1.0 - abs(cb - cp) / (abs(cb) + abs(cp)) if cb or cp else 1.0)
            d["outcome"].append(float(b["world_replay"] == p["world_replay"]))
    out = {}
    for key, d in acc.items():
        m = {k: float(np.mean(v)) if v else float("nan") for k, v in d.items() if k != "n"}
        m["n_pairs"] = float(len(d["n"]))
        parts = [m["onset"], m["variables"], m["outcome"]]
        m["action_stability"] = (
            float(np.nanmean(parts)) if np.isfinite(parts).any() else float("nan")
        )
        out[key] = m
    return out


def mechanism_ensemble(
    world: Any, size: int = 8, rel_scale: float = 0.25, seed: int = 0
) -> list[Any]:
    """Plausible mechanisms around the world: the world itself plus ``size − 1`` copies with
    strengths perturbed by ``rel_scale`` × their spread (same graph, noise and label)."""
    from twist.worlds.shifts import perturb_mechanism

    state = getattr(world, "state", world)
    return [state] + [perturb_mechanism(state, rel_scale, seed=seed + j) for j in range(1, size)]


def ensemble_outcomes(
    ensemble: list[Any], factual_state: np.ndarray, plans: list[Any]
) -> np.ndarray:
    """World label of each action replayed (unit-level, Pearl) under each candidate mechanism:
    array ``(len(ensemble), len(plans))``. Abduction uses each candidate on the factual state."""
    from twist.replay import replay_pearl

    out = np.zeros((len(ensemble), len(plans)), int)
    for j, m in enumerate(ensemble):
        z = replay_pearl(factual_state, plans, m)
        out[j] = m.label(z)
    return out


def probability_calibration(
    p_reported: np.ndarray, outcome: np.ndarray, p_true: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """How well a reported target probability matches reality: mean |p̂ − p| against the true
    interventional probability, and ECE of p̂ against realized outcomes."""
    p_reported, outcome, p_true = map(np.asarray, (p_reported, outcome, p_true))
    ok = np.isfinite(p_reported) & np.isfinite(p_true)
    if not ok.any():
        return {"abs_gap": float("nan"), "ece": float("nan"), "n": 0.0}
    pr, y, pt = p_reported[ok], outcome[ok].astype(float), p_true[ok]
    bins = np.minimum((pr * n_bins).astype(int), n_bins - 1)
    ece = sum(
        (bins == b).mean() * abs(y[bins == b].mean() - pr[bins == b].mean())
        for b in range(n_bins)
        if (bins == b).any()
    )
    return {"abs_gap": float(np.abs(pr - pt).mean()), "ece": float(ece), "n": float(ok.sum())}
