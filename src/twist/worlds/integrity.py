"""Automated world integrity checks (plan §Week 4 deliverables and acceptance criteria).

``check_world`` verifies, for one generated world and its simulated population:

* stability — finite and bounded trajectories, no growth across the horizon;
* graph — well-formed, consistent with the parameters, no lag-1 self-loops on system channels
  (exogenous drivers and latent confounders are allowed AR(1) dynamics);
* action space — non-empty, valid indices, well-formed controls;
* label — fitted, balanced, support inside the horizon after the roots;
* causal route — an actionable channel reaches the label support in the time-unrolled graph
  *and* intervening on it measurably moves the label margin under oracle replay;
* distractor — at least one state channel is not an ancestor of the label;
* replay — the empty action reproduces every unit exactly;
* disclosure — the exact-abduction flag matches the observation model.

``mechanism_distance`` and ``check_family_diversity`` verify that worlds of one family differ at
the mechanism level, not only in sampled trajectories.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twist.actions.observed_variable import PointIntervention
from twist.metrics.mechanism import causal_reachability

BALANCE = (0.2, 0.8)
MAX_ABS = 1e3
MAX_GROWTH = 10.0
MIN_ROUTE_EFFECT = 0.01  # |Δ label margin| / std(label margin)


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def ancestors(G: np.ndarray, targets: list[int]) -> set[int]:
    """Channels with a directed path (any lags) into ``targets`` in the summary graph."""
    summary = np.asarray(np.any(np.asarray(G, bool), axis=2))  # summary[i, j] ⇔ j → i
    found: set[int] = set()
    frontier = set(targets)
    while frontier:
        parents = {int(j) for i in frontier for j in np.flatnonzero(summary[i])}
        frontier = parents - found - set(targets)
        found |= parents
    return found


def _special_channels(world: Any) -> set[int]:
    roles = getattr(world, "state", world).roles
    out = set()
    for key in ("latent_confounder",):
        if isinstance(roles.get(key), int):
            out.add(roles[key])
    if isinstance(roles.get("regime_driver"), int):
        out.add(roles["regime_driver"])
    return out


def check_world(
    world: Any, x: np.ndarray, exo: dict[str, np.ndarray], n_probe: int = 64
) -> dict[str, Any]:
    state_world = getattr(world, "state", world)
    z = exo["state"]
    T, k_state = world.T, world.k_state
    checks = []

    # --- stability
    finite = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(z)))
    checks.append(_check("finite", finite))
    amax = float(np.abs(z).max())
    checks.append(_check("bounded", amax < MAX_ABS, f"max |z| = {amax:.3g}"))
    q = T // 4
    early, late = z[:, q : 2 * q].std(), z[:, -q:].std()
    growth = float(late / max(early, 1e-12))
    checks.append(
        _check("no_growth", growth < MAX_GROWTH, f"std ratio last/second quarter {growth:.3g}")
    )

    # --- graph
    G = state_world.graph
    shape_ok = G.shape == (k_state, k_state, world.L) and G.dtype == bool
    checks.append(_check("graph_shape", shape_ok, f"{G.shape}"))
    special = _special_channels(world)
    sys_channels = [i for i in range(k_state) if i not in special]
    self_loops = [i for i in sys_channels if G[i, i, 0]]
    # physical self-dynamics (positions/velocities; storage, thermal inertia, deferred demand)
    allows_self = state_world.family in ("spring", "energy")
    checks.append(
        _check(
            "no_lag1_self_loops",
            allows_self or not self_loops,
            f"self-loops on {self_loops}" if self_loops else "",
        )
    )
    coef = getattr(state_world, "coef", None)
    if coef is not None:
        nz = np.abs(coef) > 0
        nz = nz.any(axis=0) if nz.ndim == 4 else nz
        consistent = bool(np.all(~nz.transpose(1, 2, 0) | state_world.dependency_graph()))
        checks.append(_check("graph_covers_parameters", consistent))

    # --- action space
    space = world.action_space()
    act_ok = bool(space.actionable) and all(0 <= a < k_state for a in space.actionable)
    ctrl_ok = all(len(c.loadings) == k_state for c in space.controls.values())
    checks.append(_check("action_space", act_ok and ctrl_ok, f"actionable {space.actionable}"))

    # --- label
    label = world.label
    y = world.outcome(z)
    bal = float(y.mean())
    checks.append(_check("label_fitted", label.fitted))
    checks.append(_check("label_balance", BALANCE[0] <= bal <= BALANCE[1], f"P(y=1) = {bal:.3f}"))
    support = label.support(T, k_state)
    site = label.site(T)
    checks.append(_check("label_support", support.any() and site >= world.L, f"site t={site}"))

    # --- causal route (graph): from an actionable non-label channel to the label support
    dep = state_world.dependency_graph()
    label_ch = set(label.channels)
    candidates = [a for a in space.actionable if a not in label_ch]
    route_src = state_world.roles.get("route_source")
    ordered = ([route_src] if route_src in candidates else []) + [
        a for a in candidates if a != route_src
    ]
    # an action at *some* time before the site may reach the label (path lengths are constrained
    # without self-loops), so reachability is tested from every timestep in [L, site)
    reachable = []
    for a in ordered:
        active = np.zeros((T, k_state), bool)
        active[world.L : site, a] = True
        if (causal_reachability(state_world, active) & support).any():
            reachable.append(a)
    checks.append(_check("causal_route_graph", bool(reachable), f"route channels {reachable[:5]}"))

    # --- causal route (empirical): the best of a few probes do(X^a_t = x + 1 sd) at candidate
    # times must move the label margin under oracle replay
    effect, best = 0.0, ""
    first_support = int(np.flatnonzero(support.any(axis=1)).min())
    times = sorted(
        {
            t
            for t in (
                site - 1,
                site - 2,
                site - 3,
                site - 5,
                site - 10,
                first_support - 1,
                first_support,
            )
            if world.L <= t < site
        }
    )
    n = min(n_probe, len(z))
    sub = {name: arr[:n] for name, arr in exo.items()}
    margin_sd = max(float(label.margin(z).std()), 1e-12)
    for a in reachable[:3]:
        sd = float(z[:, :, a].std()) or 1.0
        for t_act in times:
            acts = [PointIntervention(t_act, [a], [float(z[i, t_act, a] + sd)]) for i in range(n)]
            _, z_cf = world.replay_units(sub, acts)
            e = float(np.abs(label.margin(z_cf) - label.margin(z[:n])).mean() / margin_sd)
            if e > effect:
                effect, best = e, f"channel {a}, t={t_act}"
    checks.append(
        _check(
            "causal_route_effect",
            effect > MIN_ROUTE_EFFECT,
            f"max |Δmargin|/sd = {effect:.3g} ({best or 'no route'})",
        )
    )

    # --- distractor: a state channel that is not an ancestor of the label
    anc = ancestors(dep, sorted(label_ch))
    non_anc = [i for i in range(k_state) if i not in anc and i not in label_ch]
    checks.append(_check("distractor_exists", bool(non_anc), f"non-ancestors {non_anc}"))
    d = state_world.roles.get("distractor")
    if d is not None:
        checks.append(_check("designated_distractor_non_causal", d in non_anc, f"channel {d}"))

    # --- replay sanity and disclosure
    n = min(8, len(z))
    sub = {name: arr[:n] for name, arr in exo.items()}
    x_rep, z_rep = world.replay_units(sub, None)
    dev = float(max(np.abs(x_rep - x[:n]).max(), np.abs(z_rep - z[:n]).max()))
    checks.append(_check("empty_action_replay", dev <= 1e-9, f"max dev {dev:.2g}"))
    obs = getattr(world, "observation", None)
    expected_exact = obs is None or obs.is_identity
    exact_flag = bool(
        world.assumptions.get("exact_abduction", world.assumptions.get("exact_replay"))
    )
    checks.append(
        _check("exactness_disclosed", exact_flag == expected_exact, f"exact_abduction={exact_flag}")
    )
    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "label_balance": bal,
        "route_effect": effect,
    }


def mechanism_distance(a: Any, b: Any, n: int = 256, seed: int = 0) -> float:
    """Relative functional distance ``E|F_a(p) − F_b(p)| / E|F_a(p) − x_{t−1}|`` on shared random
    lag windows. Dividing by the *increment* over persistence keeps near-identity dynamics (e.g.
    physical integrators ``p + dt·v``) from masking real differences. Infinite when the worlds have
    different shapes (different by construction)."""
    sa, sb = getattr(a, "state", a), getattr(b, "state", b)
    if (sa.k, sa.L, type(sa)) != (sb.k, sb.L, type(sb)):
        return float("inf")
    rng = np.random.default_rng(seed)
    past = rng.normal(size=(n, sa.L, sa.k))
    fa, fb = np.asarray(sa.structural_mean(past)), np.asarray(sb.structural_mean(past))
    return float(np.abs(fa - fb).mean() / max(np.abs(fa - past[:, 0]).mean(), 1e-12))


def check_family_diversity(worlds: list[Any], min_distance: float = 0.05) -> dict[str, Any]:
    """All mechanism hashes are distinct and every pair differs functionally by ≥ ``min_distance``."""
    hashes = [w.card()["mechanism_hash"] for w in worlds]
    dmin = float("inf")
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            dmin = min(dmin, mechanism_distance(worlds[i], worlds[j]))
    return {
        "n_worlds": len(worlds),
        "distinct_hashes": len(set(hashes)) == len(hashes),
        "min_relative_distance": dmin,
        "passed": len(set(hashes)) == len(hashes) and dmin >= min_distance,
    }
