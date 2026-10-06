"""Unit-level Pearl replay (draft Eq. 3).

The oracle keeps the factual prefix, replaces intervened equations, and recursively evaluates the
rest with the *same* abducted innovations. One backend-agnostic recursion serves the NumPy oracle
and the differentiable torch path used by structural controls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from twist.actions.schemas import ActionPlan, ActionSpace, DenseSchedule, TemporalAction
from twist.replay.abduction import abduct_noise
from twist.replay.validation import validate_replay_inputs
from twist.utils import backend as bk
from twist.utils.validation import as_batch
from twist.worlds.base_scm import TemporalSCM

ActionLike = TemporalAction | ActionPlan | DenseSchedule | None


@dataclass
class ScheduleBatch:
    """Stacked dense schedules, each array ``(B, T, k)``."""

    mask: np.ndarray
    value: Any
    shift: Any

    @classmethod
    def from_actions(
        cls,
        actions: ActionLike | Sequence[ActionLike],
        B: int,
        T: int,
        k: int,
        space: ActionSpace | None = None,
    ) -> ScheduleBatch:
        if actions is None or isinstance(actions, (TemporalAction, ActionPlan, DenseSchedule)):
            actions = [actions] * B
        if len(actions) != B:
            raise ValueError(f"{len(actions)} actions for {B} trajectories")
        scheds = [to_dense(a, T, k, space) for a in actions]
        return cls(
            np.stack([s.mask for s in scheds]),
            np.stack([s.value for s in scheds]),
            np.stack([s.shift for s in scheds]),
        )

    def first_times(self) -> np.ndarray:
        """First active timestep per unit; ``T`` when the schedule is empty."""
        active = (self.mask | (np.asarray(self.shift) != 0)).any(axis=2)
        T = active.shape[1]
        return np.where(active.any(1), active.argmax(1), T)


def to_dense(action: ActionLike, T: int, k: int, space: ActionSpace | None = None) -> DenseSchedule:
    """Compile any action-like object to a dense schedule.

    ``space`` (``world.action_space()``) is required for actions that use world interfaces, such as
    high-level controls; observed-variable actions compile without it.
    """
    if action is None:
        return DenseSchedule.empty(T, k)
    if isinstance(action, ActionPlan):
        action = action.action
    if isinstance(action, DenseSchedule):
        return action
    return action.compile(space) if space is not None else action.to_schedule(T, k)


def replay(
    world: TemporalSCM,
    factual_eps: Any,
    mask: np.ndarray,
    value: Any,
    shift: Any,
) -> Any:
    """Core recursion over ``(B, T, k)`` arrays (NumPy or torch for ``factual_eps``/``value``/``shift``).

    ``x_t = value_t`` where ``mask_t``; otherwise ``F(x_{t-L:t}) + ε_t + shift_t`` (roots: ``ε_t``).
    """
    _, T, _ = mask.shape
    steps: list[Any] = []
    for t in range(T):
        if t < world.L:
            base = factual_eps[:, t]
        else:
            past = bk.stack(steps[t - world.L : t][::-1], axis=1)
            base = world.step(past, factual_eps[:, t])
        steps.append(bk.where(mask[:, t], value[:, t], base + shift[:, t]))
    return bk.stack(steps, axis=1)


def _prepare(factual: np.ndarray, actions, world: TemporalSCM):
    x, single = as_batch(np.asarray(factual, dtype=float))
    B, T, k = x.shape
    sb = ScheduleBatch.from_actions(actions, B, T, k, world.action_space())
    validate_replay_inputs(world, x, sb.mask, sb.value, sb.shift)
    return x, single, sb


def replay_pearl(
    factual: np.ndarray,
    actions: ActionLike | Sequence[ActionLike],
    world: TemporalSCM,
    eps: np.ndarray | None = None,
) -> np.ndarray:
    """Abduction–action–prediction with factual innovations retained (unit-level counterfactual)."""
    x, single, sb = _prepare(factual, actions, world)
    e = abduct_noise(x, world) if eps is None else as_batch(eps)[0]
    out = replay(world, e, sb.mask, sb.value, sb.shift)
    return out[0] if single else out


def replay_interventional_distribution(
    factual: np.ndarray,
    actions: ActionLike | Sequence[ActionLike],
    world: TemporalSCM,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fresh innovations after the first intervened timestep: samples of ``P(X | do(I), prefix)``.

    Returns ``(n_samples, B, T, k)`` (or ``(n_samples, T, k)`` for a single unit). This answers an
    interventional query and must never be averaged with the unit-level Pearl replay.
    """
    x, single, sb = _prepare(factual, actions, world)
    e = abduct_noise(x, world)
    t_first = sb.first_times()
    after = np.arange(world.T)[None, :, None] > t_first[:, None, None]
    out = []
    for _ in range(n_samples):
        fresh = world.noise.sample(rng, e.shape)
        out.append(replay(world, np.where(after, fresh, e), sb.mask, sb.value, sb.shift))
    res = np.stack(out)
    return res[:, 0] if single else res
