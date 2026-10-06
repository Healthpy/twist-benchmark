"""Noiseless (zero-innovation) rollout after the first intervened timestep.

This is the deterministic skeleton the draft's NoiselessSCMRecourse follows. It answers a
different question from Pearl replay; the two CF-faith semantics are never averaged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from twist.replay.abduction import abduct_noise
from twist.replay.pearl_replay import ActionLike, _prepare, replay
from twist.worlds.base_scm import TemporalSCM


def noiseless_eps(eps: np.ndarray, t_first: np.ndarray) -> np.ndarray:
    """Zero innovations strictly after each unit's first intervened timestep."""
    after = np.arange(eps.shape[1])[None, :, None] > np.asarray(t_first)[:, None, None]
    return np.where(after, 0.0, eps)


def replay_noiseless(
    factual: np.ndarray,
    actions: ActionLike | Sequence[ActionLike],
    world: TemporalSCM,
) -> np.ndarray:
    x, single, sb = _prepare(factual, actions, world)
    e = noiseless_eps(abduct_noise(x, world), sb.first_times())
    out = replay(world, e, sb.mask, sb.value, sb.shift)
    return out[0] if single else out
