"""How good is a mechanism the explainer uses? Computed by the benchmark against the true world.

* graph recovery — precision / recall / F1 / SHD of the summary graph (any lag; self-loops excluded),
  lagged precision / recall / F1 over (target, source, lag) triples, and lag accuracy: among the
  correctly recovered summary edges, the fraction whose set of lags is exactly right;
* one-step error — mean |x_t − F̂(x_{t−L:t})| on held-out trajectories (the oracle's equals the
  innovation scale, the irreducible floor);
* rollout error — noiseless rollouts from a factual prefix under F̂ vs F* (deterministic skeleton);
* replay error — unit-level Pearl replay of random point interventions under F̂ (abducted with
  F̂) vs the oracle replay of the same action, normalized by channel std. This is the quantity
  that matters for counterfactual explanations.
* functional distance — the suite's relative distance on the increment over persistence.

For partially observed worlds the estimate lives in observation space: graphs are compared on the
observed channels, and oracle replays use the stored exogenous variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twist.actions.observed_variable import PointIntervention
from twist.actions.schemas import ActionPlan
from twist.replay import replay_noiseless, replay_pearl


def _summary(G: np.ndarray) -> np.ndarray:
    s = np.asarray(np.any(np.asarray(G, bool), axis=2))
    np.fill_diagonal(s, False)
    return s


def _pad(G: np.ndarray, L: int) -> np.ndarray:
    G = np.asarray(G, bool)
    if G.shape[2] < L:
        G = np.concatenate([G, np.zeros((*G.shape[:2], L - G.shape[2]), bool)], axis=2)
    return G


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return prec, rec, (2 * prec * rec / (prec + rec) if tp else 0.0)


def graph_recovery(est: np.ndarray, true: np.ndarray) -> dict[str, float]:
    e, t = _summary(est), _summary(true)
    tp, fp, fn = int((e & t).sum()), int((e & ~t).sum()), int((~e & t).sum())
    prec, rec, f1 = _prf(tp, fp, fn)
    # lagged level (cross edges only, graphs padded to a common maximal lag)
    L = max(np.shape(est)[2], np.shape(true)[2])
    el, tl = _pad(est, L), _pad(true, L)
    cross = ~np.eye(el.shape[0], dtype=bool)[:, :, None]
    el, tl = el & cross, tl & cross
    ltp, lfp, lfn = int((el & tl).sum()), int((el & ~tl).sum()), int((~el & tl).sum())
    lprec, lrec, lf1 = _prf(ltp, lfp, lfn)
    both = np.argwhere(e & t)
    lag_acc = (
        float(np.mean([np.array_equal(el[i, j], tl[i, j]) for i, j in both]))
        if len(both)
        else float("nan")
    )
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "shd": fp + fn,
        "lag_precision": lprec,
        "lag_recall": lrec,
        "lag_f1": lf1,
        "lag_accuracy": lag_acc,
    }


def one_step_error(mech: Any, x: np.ndarray) -> float:
    pred = mech.one_step_means(x)
    L = mech.L
    return float(np.abs(x[:, L:] - pred[:, L:]).mean())


def _observed_graph(world: Any) -> np.ndarray:
    G = world.dependency_graph()
    idx = list(getattr(world, "obs_index", range(G.shape[0])))
    return G[np.ix_(idx, idx)]


def mechanism_quality(
    mech: Any,
    world: Any,
    x: np.ndarray,
    exo: dict[str, np.ndarray],
    n_units: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = min(n_units, len(x))
    xs = x[:n]
    sub = {k: v[:n] for k, v in exo.items()}
    exact = bool(world.assumptions.get("exact_abduction", True))
    T, L = world.T, max(mech.L, world.L)
    sd = xs.reshape(-1, xs.shape[-1]).std(0) + 1e-8
    out: dict[str, Any] = {"mechanism": mech.world_id, "one_step_error": one_step_error(mech, x)}
    out["oracle_one_step_error"] = one_step_error(world, x) if exact else float("nan")
    out.update(
        {
            f"graph_{k}": v
            for k, v in graph_recovery(
                mech.graph if mech.graph.shape[0] == x.shape[-1] else mech.dependency_graph(),
                _observed_graph(world),
            ).items()
        }
    )
    # rollout error of the deterministic skeleton from a mid-horizon prefix (empty action at t0)
    t0 = T // 2
    if exact:
        anchor = [PointIntervention(t0, [0], [float(v)]) for v in xs[:, t0, 0]]
        roll_est = replay_noiseless(xs, anchor, mech)
        roll_true = replay_noiseless(xs, anchor, world)
        out["rollout_error"] = float((np.abs(roll_est - roll_true)[:, t0:] / sd).mean())
    else:
        out["rollout_error"] = float("nan")
    # unit-level replay error of random point interventions
    chans = rng.integers(0, x.shape[-1], n)
    times = rng.integers(L, T - 1, n)
    acts = [
        PointIntervention(int(t), [int(c)], [float(xs[i, t, c] + sd[c])])
        for i, (t, c) in enumerate(zip(times, chans, strict=True))
    ]
    est_rep = replay_pearl(xs, acts, mech)
    plans = [world.to_state_plan(ActionPlan(a, native=False, semantics="probe")) for a in acts]
    true_rep, _ = world.replay_units(sub, plans)
    out["replay_error"] = float((np.abs(est_rep - true_rep) / sd).mean())
    return out
