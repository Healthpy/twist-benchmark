"""Mechanism-relative metrics: CF-faith (hard/soft, Pearl/noiseless), do-complexity ``D``,
replay error.

* **CF-faith** reads the first edited slice of a proposal as one ``do()`` and compares the
  proposed suffix with the continuation that action generates. Unscorable (``t0 ≥ T-1``) → NaN.
* **D** counts the timesteps that must be imposed to reproduce the proposal under the named
  semantics (timesteps, not coordinates). It is observed-variable schedule complexity, not cost.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from twist.actions.observed_variable import PointIntervention
from twist.actions.trajectory_adapters import DEFAULT_CHANGE_TOL, changed_mask
from twist.metrics.conventional import first_change
from twist.replay.abduction import abduct_noise, structural_residual
from twist.replay.deterministic_rollout import noiseless_eps, replay_noiseless
from twist.replay.pearl_replay import replay_pearl
from twist.worlds.base_scm import TemporalSCM

Semantics = Literal["pearl", "noiseless"]
FAITH_TOL = 1e-3


def replay_error(
    x_hat: np.ndarray, x_replay: np.ndarray, t_from: np.ndarray | None = None
) -> np.ndarray:
    """``E_replay = (1/T) Σ_t d(x̂_t, x^A_t)`` with ``d`` the mean absolute error over channels.

    If ``t_from`` is given, only timesteps ``t > t_from`` (the suffix) are averaged.
    """
    err = np.abs(x_hat - x_replay).mean(axis=2)
    if t_from is None:
        return err.mean(axis=1)
    sel = np.arange(err.shape[1])[None, :] > np.asarray(t_from)[:, None]
    with np.errstate(invalid="ignore"):
        return np.where(sel.any(1), (err * sel).sum(1) / np.maximum(sel.sum(1), 1), np.nan)


def cf_faith(
    world: TemporalSCM,
    x: np.ndarray,
    x_hat: np.ndarray,
    semantics: Semantics,
    tol: float = FAITH_TOL,
    change_tol: float = DEFAULT_CHANGE_TOL,
    sigma: float = 1.0,
    t_action: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """First-slice CF-faith for a batch of proposals.

    Returns ``hard`` (1/0/NaN), ``soft = exp(-residual/σ)``, ``residual`` (mean L1 over suffix
    entries), ``t0`` and ``scorable``. If ``t_action`` (declared action start) is given, any edit
    before it is retroactive and scores 0.
    """
    B, T, _ = x.shape
    t0, has = first_change(x, x_hat, change_tol)
    scorable = has & (t0 < T - 1)
    m = changed_mask(x, x_hat, change_tol)
    actions = [
        PointIntervention(
            int(t0[n]), np.flatnonzero(m[n, t0[n]]).tolist(), x_hat[n, t0[n], m[n, t0[n]]].tolist()
        )
        if has[n]
        else None
        for n in range(B)
    ]
    ref = (replay_pearl if semantics == "pearl" else replay_noiseless)(x, actions, world)
    residual = replay_error(x_hat, ref, t_from=t0)
    hard = (residual < tol).astype(float)
    if t_action is not None:
        retro = has & (t0 < np.asarray(t_action))
        hard = np.where(retro, 0.0, hard)
    hard = np.where(scorable, hard, np.nan)
    soft = np.where(scorable, np.exp(-residual / sigma), np.nan)
    return {
        "hard": hard,
        "soft": soft,
        "residual": np.where(scorable, residual, np.nan),
        "t0": t0,
        "scorable": scorable,
    }


def minimal_imposed_mask(
    world: TemporalSCM,
    x: np.ndarray,
    x_hat: np.ndarray,
    semantics: Semantics = "pearl",
    tol: float = DEFAULT_CHANGE_TOL,
) -> np.ndarray:
    """Entries that must be imposed by ``do()`` for the mechanism to reproduce ``x̂``.

    Because every imposed entry takes its proposed value, the mechanism's prediction at ``t`` is
    ``F(x̂_{t-L:t}) + ε_t`` — teacher forcing on ``x̂`` — so the minimal schedule is non-recursive.
    """
    eps = abduct_noise(x, world)
    imposed = np.abs(structural_residual(x, x_hat, world)) > tol
    if semantics == "noiseless":
        T = x.shape[1]
        any_t = imposed.any(axis=2)
        t_first = np.where(any_t.any(1), any_t.argmax(1), T)
        pred_nl = world.one_step_prediction(x_hat, noiseless_eps(eps, t_first))
        imposed = np.abs(x_hat - pred_nl) > tol
    return imposed


def do_complexity(
    world: TemporalSCM,
    x: np.ndarray,
    x_hat: np.ndarray,
    semantics: Semantics = "pearl",
    tol: float = DEFAULT_CHANGE_TOL,
) -> np.ndarray:
    """``D(x̂; x, F) = |{t : S_t ≠ ∅}|`` (draft Eq. 4)."""
    return minimal_imposed_mask(world, x, x_hat, semantics, tol).any(axis=2).sum(axis=1)


# --------------------------------------------------------------------------- graded diagnostics
def normalized_replay_error(
    x_hat: np.ndarray, x_replay: np.ndarray, scale: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Scale-normalized ``E_replay``: mean of ``|x̂ − x^A| / (IQR_i + ε)`` over all entries."""
    return (np.abs(x_hat - x_replay) / (np.asarray(scale)[None, None, :] + eps)).mean(axis=(1, 2))


def soft_faith(err_norm: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Graded mechanism consistency ``S_faith = exp(−E_replay_norm / σ)`` ∈ (0, 1]."""
    return np.exp(-np.asarray(err_norm) / sigma)


def effect_profile_correlation(
    x: np.ndarray, x_hat: np.ndarray, x_replay: np.ndarray, t_from: int = 0
) -> float:
    """Pearson correlation of proposed effect ``x̂ − x`` and replayed effect ``x^A − x`` over
    entries with ``t ≥ t_from``. NaN when either effect is constant (e.g. no edit)."""
    a = (x_hat - x)[t_from:].ravel()
    b = (x_replay - x)[t_from:].ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def support_overlap(
    x: np.ndarray, x_hat: np.ndarray, x_replay: np.ndarray, tol: float = DEFAULT_CHANGE_TOL
) -> float:
    """Jaccard overlap of the entries changed by the proposal and by the replayed action."""
    a, b = changed_mask(x, x_hat, tol), changed_mask(x, x_replay, tol)
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def retroactivity(
    x: np.ndarray, x_hat: np.ndarray, t_action: int | None, tol: float = DEFAULT_CHANGE_TOL
) -> float:
    """Fraction of the proposal's changed entries that lie *before* the action starts.

    Any positive value is a retroactive edit: no intervention at ``t_action`` can cause it.
    """
    m = changed_mask(x, x_hat, tol)
    if t_action is None or not m.any():
        return float("nan") if t_action is None else 0.0
    return float(m[:t_action].sum() / m.sum())


def prefix_preservation(
    x: np.ndarray, x_hat: np.ndarray, t_action: int | None, tol: float = DEFAULT_CHANGE_TOL
) -> float:
    """Fraction of pre-action entries (``t < t_action``) left unchanged; NaN if no prefix."""
    if t_action is None or t_action == 0:
        return float("nan")
    return float(1.0 - changed_mask(x[:t_action], x_hat[:t_action], tol).mean())


def causal_reachability(world: TemporalSCM, active: np.ndarray) -> np.ndarray:
    """Entries reachable from the intervened entries along the time-unrolled dependency graph."""
    G = world.dependency_graph()  # (k, k, L): G[i, j, l-1] ⇔ X^j_{t-l} → X^i_t
    reach = np.asarray(active, bool).copy()
    T = reach.shape[0]
    for t in range(T):
        for lag in range(1, world.L + 1):
            if t - lag >= 0 and reach[t - lag].any():
                reach[t] |= (G[:, :, lag - 1] & reach[t - lag][None, :]).any(axis=1)
    return reach


def causal_path_consistency(
    world: TemporalSCM,
    x: np.ndarray,
    x_hat: np.ndarray,
    active: np.ndarray,
    tol: float = DEFAULT_CHANGE_TOL,
) -> float:
    """Fraction of the proposal's changed entries that a causal path from the action can reach.

    Changes outside the reachable set cannot be consequences of the action under the true graph.
    NaN when the proposal changes nothing.
    """
    m = changed_mask(x, x_hat, tol)
    if not m.any():
        return float("nan")
    return float((m & causal_reachability(world, active)).sum() / m.sum())
