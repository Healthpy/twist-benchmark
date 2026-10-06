"""Observed-variable ``do()`` interventions: point interventions and explicit schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from twist.actions.schemas import DenseSchedule, TemporalAction


@dataclass
class PointIntervention(TemporalAction):
    """``do(X^S_t = a_t)`` at a single timestep — the draft's atomic action."""

    kind: ClassVar[str] = "point"
    t: int
    targets: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        self.targets = [int(v) for v in self.targets]
        self.values = [float(v) for v in self.values]
        if len(self.targets) != len(self.values):
            raise ValueError("targets and values must have equal length")

    @property
    def variables(self) -> list[int]:
        return sorted(set(self.targets))

    def to_schedule(self, T: int, k: int) -> DenseSchedule:
        if not 0 <= self.t < T:
            raise IndexError(f"t={self.t} outside [0, {T})")
        s = DenseSchedule.empty(T, k)
        s.mask[self.t, self.targets] = True
        s.value[self.t, self.targets] = self.values
        return s

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "t": self.t, "targets": self.targets, "values": self.values}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PointIntervention:
        return cls(t=d["t"], targets=d["targets"], values=d["values"])


@dataclass
class ScheduleIntervention(TemporalAction):
    """An arbitrary observed-variable schedule ``I = {(t, S_t, a_t)}``.

    ``entries`` is a list of ``(t, targets, values)``. A full changed schedule of an edited
    trajectory is represented this way.
    """

    kind: ClassVar[str] = "schedule"
    entries: list[tuple[int, list[int], list[float]]]

    def __post_init__(self) -> None:
        self.entries = sorted(
            (int(t), [int(i) for i in s], [float(a) for a in v]) for t, s, v in self.entries
        )

    @classmethod
    def from_dense(cls, mask: np.ndarray, value: np.ndarray) -> ScheduleIntervention:
        entries = []
        for t in np.flatnonzero(mask.any(axis=1)):
            idx = np.flatnonzero(mask[t])
            entries.append((int(t), idx.tolist(), value[t, idx].tolist()))
        return cls(entries)

    @property
    def variables(self) -> list[int]:
        return sorted({i for _, s, _ in self.entries for i in s})

    def to_schedule(self, T: int, k: int) -> DenseSchedule:
        s = DenseSchedule.empty(T, k)
        for t, targets, values in self.entries:
            if not 0 <= t < T:
                raise IndexError(f"t={t} outside [0, {T})")
            s.mask[t, targets] = True
            s.value[t, targets] = values
        return s

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "entries": [list(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScheduleIntervention:
        return cls([tuple(e) for e in d["entries"]])
