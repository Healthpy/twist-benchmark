"""Diagnostic readings that reconstruct an action from a trajectory-only proposal.

These adapters are the benchmark's *registered intervention readings*. Their output is always
marked ``native=False`` so a reconstructed action is never presented as the method's own claim.
``fit_error`` is filled in by the evaluator after replay. See
``docs/technical_notes/native_vs_reconstructed_actions.md``.

Every adapter has the signature ``adapter(x, x_hat, tol, world=None) -> ActionPlan | None``
(``None`` = no action; the unit abstains). Adapters in ``REQUIRES_WORLD`` read the proposal
through the mechanism and raise if ``world`` is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from twist.actions.interval import ParametricPulse
from twist.actions.observed_variable import PointIntervention, ScheduleIntervention
from twist.actions.schemas import ActionPlan

if TYPE_CHECKING:
    from twist.worlds.base_scm import TemporalSCM

DEFAULT_CHANGE_TOL = 1e-3


class Adapter(Protocol):
    def __call__(
        self,
        x: np.ndarray,
        x_hat: np.ndarray,
        tol: float = DEFAULT_CHANGE_TOL,
        world: TemporalSCM | None = None,
    ) -> ActionPlan | None: ...


def changed_mask(x: np.ndarray, x_hat: np.ndarray, tol: float = DEFAULT_CHANGE_TOL) -> np.ndarray:
    """Changed-element predicate ``|x̂ - x| > tol`` (one tolerance for t0, sparsity and S_t)."""
    return np.asarray(np.abs(np.asarray(x_hat) - np.asarray(x)) > tol)


def runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D boolean array as half-open ``(start, end)`` pairs."""
    padded = np.concatenate([[False], np.asarray(active, bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def _plan(action, semantics: str, adapter: str) -> ActionPlan:
    return ActionPlan(action, native=False, semantics=semantics, adapter=adapter)


def first_change_atomic(x, x_hat, tol=DEFAULT_CHANGE_TOL, world=None) -> ActionPlan | None:
    """Read the first edited slice as one ``do()`` on its changed coordinates."""
    m = changed_mask(x, x_hat, tol)
    ts = np.flatnonzero(np.any(m, axis=1))
    if ts.size == 0:
        return None
    t0 = int(ts[0])
    idx = np.flatnonzero(m[t0])
    action = PointIntervention(t0, idx.tolist(), x_hat[t0, idx].tolist())
    return _plan(action, "trajectory_inferred_first_slice", "first_change_atomic")


def first_changed_segment(x, x_hat, tol=DEFAULT_CHANGE_TOL, world=None) -> ActionPlan | None:
    """Read the first contiguous run of edited timesteps as a ``do()`` schedule."""
    m = changed_mask(x, x_hat, tol)
    segs = runs(np.any(m, axis=1))
    if not segs:
        return None
    t0, t1 = segs[0]
    seg = np.zeros_like(m)
    seg[t0:t1] = m[t0:t1]
    action = ScheduleIntervention.from_dense(seg, x_hat)
    return _plan(action, "trajectory_inferred_first_segment", "first_changed_segment")


def full_changed_schedule(x, x_hat, tol=DEFAULT_CHANGE_TOL, world=None) -> ActionPlan | None:
    """Read every edited entry as an imposed ``do()`` — reproduces the proposal by construction."""
    m = changed_mask(x, x_hat, tol)
    if not m.any():
        return None
    action = ScheduleIntervention.from_dense(m, x_hat)
    return _plan(action, "trajectory_inferred_full_schedule", "full_changed_schedule")


def sparse_segmentation(x, x_hat, tol=DEFAULT_CHANGE_TOL, world=None) -> ActionPlan | None:
    """Impose only the onset of every edited segment, per channel; let the mechanism do the rest.

    Each channel's edit is split into contiguous runs of changed timesteps and only the first entry
    of each run becomes a ``do()``. This generalizes the first-slice reading (one onset) toward the
    full schedule (every entry): a proposal made of several atomic actions, each followed by
    mechanism propagation, is reproduced exactly with one imposed entry per segment.
    """
    m = changed_mask(x, x_hat, tol)
    if not m.any():
        return None
    onsets = m & ~np.vstack([np.zeros((1, m.shape[1]), bool), m[:-1]])
    action = ScheduleIntervention.from_dense(onsets, x_hat)
    return _plan(action, "trajectory_inferred_segment_onsets", "sparse_segmentation")


def change_point_pulse_fit(x, x_hat, tol=DEFAULT_CHANGE_TOL, world=None) -> ActionPlan | None:
    """Fit one soft parametric pulse to the dominant change in the mechanism residual.

    The residual ``r = x̂ − (F(x̂ lags) + ε)`` is the additive shift the mechanism needs to produce
    the proposal. The run of timesteps with the largest residual energy defines the change point
    (onset, duration); channels whose residual exceeds ``tol`` in that run are the targets; a
    ``rect`` and a ``raised_cosine`` profile are fit by least squares and the better one is kept.
    """
    if world is None:
        raise ValueError(
            "change_point_pulse_fit reads the proposal through the mechanism; pass world"
        )
    from twist.replay.abduction import structural_residual

    r = structural_residual(x, x_hat, world)
    active = np.abs(r) > tol
    segs = runs(np.any(active, axis=1))
    if not segs:
        return None
    s, e = max(segs, key=lambda se: float((r[se[0] : se[1]] ** 2).sum()))
    targets = np.flatnonzero(np.any(active[s:e], axis=0))
    best: tuple[float, ParametricPulse] | None = None
    for shape in ("rect", "raised_cosine"):
        phi = ParametricPulse(s, e - s, [0], 1.0, shape).profile()
        amp = r[s:e, targets].T @ phi / (phi @ phi)
        err = float(((r[s:e, targets] - phi[:, None] * amp[None, :]) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, ParametricPulse(s, e - s, targets.tolist(), amp.tolist(), shape))
    assert best is not None
    return _plan(best[1], "trajectory_inferred_pulse", "change_point_pulse_fit")


ADAPTERS: dict[str, Adapter] = {
    "first_change_atomic": first_change_atomic,
    "first_changed_segment": first_changed_segment,
    "full_changed_schedule": full_changed_schedule,
    "sparse_segmentation": sparse_segmentation,
    "change_point_pulse_fit": change_point_pulse_fit,
}
REQUIRES_WORLD = frozenset({"change_point_pulse_fit"})
