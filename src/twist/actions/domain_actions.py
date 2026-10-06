"""High-level control actions ``u_t ∈ U`` acting through ``F(x_t, u_t, ε_t)``.

A world declares its control inputs as :class:`~twist.actions.schemas.ControlChannel` entries in
its action space. A control enters the structural equations additively through a fixed input map
(``loadings``) after ``delay`` steps, so replay keeps the mechanism and the factual innovations
and only adds the control's effect: it is a soft intervention, never a ``do()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from twist.actions.schemas import ActionSpace, DenseSchedule, TemporalAction


@dataclass
class HighLevelControl(TemporalAction):
    """Control ``control`` set to ``values[j]`` at times ``start + j`` for ``j < end - start``."""

    kind: ClassVar[str] = "control"
    control: str
    start: int
    end: int
    values: list[float]

    def __post_init__(self) -> None:
        self.values = [float(v) for v in self.values]

    @property
    def variables(self) -> list[int]:
        return []

    def to_schedule(self, T: int, k: int) -> DenseSchedule:
        raise NotImplementedError(
            "HighLevelControl needs the world's control interface; use action.compile(world.action_space())"
        )

    def compile(self, space: ActionSpace) -> DenseSchedule:
        ch = space.controls[self.control]
        if not 0 <= self.start < self.end <= space.T:
            raise IndexError(f"control window [{self.start}, {self.end}) invalid for T={space.T}")
        if len(self.values) != self.end - self.start:
            raise ValueError(f"{len(self.values)} values for window of {self.end - self.start}")
        load = np.asarray(ch.loadings, float)
        if load.shape != (space.k,):
            raise ValueError(f"control {ch.name!r} has {load.size} loadings for k={space.k}")
        s = DenseSchedule.empty(space.T, space.k)
        for j, t in enumerate(range(self.start + ch.delay, self.end + ch.delay)):
            if t < space.T:
                s.shift[t] += self.values[j] * load
        return s

    def validate(self, space: ActionSpace) -> list[str]:
        if self.control not in space.controls:
            return [f"unknown control {self.control!r}; available: {sorted(space.controls)}"]
        violations = super().validate(space)
        ch = space.controls[self.control]
        v = np.asarray(self.values)
        if np.any(v < ch.lower):
            violations.append(f"control {ch.name!r} below lower bound {ch.lower} ({v.min():.4g})")
        if np.any(v > ch.upper):
            violations.append(f"control {ch.name!r} above upper bound {ch.upper} ({v.max():.4g})")
        if self.start + ch.delay >= space.T:
            violations.append(
                f"control {ch.name!r} takes effect after the horizon (delay {ch.delay})"
            )
        return violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "control": self.control,
            "start": self.start,
            "end": self.end,
            "values": list(self.values),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HighLevelControl:
        return cls(control=d["control"], start=d["start"], end=d["end"], values=d["values"])
