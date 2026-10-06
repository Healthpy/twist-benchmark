"""Representation-level action costs.

These are *not* the do-complexity ``D`` of a proposal (see ``twist.metrics.mechanism``): ``D``
counts the observed-variable timesteps that must be imposed to reproduce a proposed trajectory,
whereas the costs here describe a declared action. Neither is an intrinsic real-world effort.
"""

from __future__ import annotations

import numpy as np

from twist.actions.schemas import DenseSchedule


def action_deltas(schedule: DenseSchedule, factual: np.ndarray) -> np.ndarray:
    """Per-entry action magnitude ``a_t``: ``value - factual`` for ``do()``, ``shift`` otherwise."""
    return np.where(schedule.mask, schedule.value - factual, schedule.shift)


def duration_cost(schedule: DenseSchedule) -> int:
    """``C_duration = Σ_t 1{a_t ≠ 0}``."""
    return schedule.n_timesteps


def magnitude_cost(schedule: DenseSchedule, factual: np.ndarray) -> float:
    """``C_magnitude = Σ_t ||a_t||_1``."""
    return float(np.abs(action_deltas(schedule, factual)).sum())


def smoothness_cost(schedule: DenseSchedule, factual: np.ndarray) -> float:
    """``C_smoothness = Σ_t ||a_t - a_{t-1}||_2^2`` (with ``a_{-1} = 0``)."""
    a = action_deltas(schedule, factual)
    a = np.vstack([np.zeros((1, a.shape[1])), a])
    return float((np.diff(a, axis=0) ** 2).sum())


def budget_cost(
    schedule: DenseSchedule, factual: np.ndarray, unit_costs: np.ndarray | None = None
) -> float:
    """``C_budget = Σ_t c(a_t)`` with a linear per-channel cost ``c(a) = Σ_i w_i |a_i|``."""
    a = np.abs(action_deltas(schedule, factual))
    w = np.ones(a.shape[1]) if unit_costs is None else np.asarray(unit_costs, float)
    return float((a * w[None, :]).sum())


def action_costs(
    schedule: DenseSchedule, factual: np.ndarray, unit_costs: np.ndarray | None = None
) -> dict[str, float]:
    return {
        "duration": float(duration_cost(schedule)),
        "magnitude": magnitude_cost(schedule, factual),
        "smoothness": smoothness_cost(schedule, factual),
        "budget": budget_cost(schedule, factual, unit_costs),
        "n_coordinates": float(schedule.active.sum()),
    }
