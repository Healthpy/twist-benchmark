"""Input checks for replay (plan §Week 2, task 3: ``validate_replay_inputs``)."""

from __future__ import annotations

import numpy as np

from twist.utils.validation import TwistValidationError, check_trajectory
from twist.worlds.base_scm import TemporalSCM


def validate_replay_inputs(
    world: TemporalSCM, factual: np.ndarray, mask: np.ndarray, value: np.ndarray, shift: np.ndarray
) -> None:
    check_trajectory(factual, world.T, world.k)
    for name, arr in (("mask", mask), ("value", value), ("shift", shift)):
        if arr.shape != factual.shape:
            raise TwistValidationError(f"{name} shape {arr.shape} != factual shape {factual.shape}")
    if mask.dtype != bool:
        raise TwistValidationError("mask must be boolean")
    if not np.all(np.isfinite(value[mask])) or not np.all(np.isfinite(shift)):
        raise TwistValidationError("non-finite intervention values")
    if not world.assumptions.get("exact_replay", False):
        raise TwistValidationError(
            f"world {world.world_id} does not support exact unit-level replay; "
            "use twist.replay.posterior_replay"
        )
