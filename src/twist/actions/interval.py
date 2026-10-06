"""Finite-duration actions: interval ``do()`` interventions and parametric pulses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from twist.actions.schemas import DenseSchedule, TemporalAction


@dataclass
class IntervalIntervention(TemporalAction):
    """``do(X^S_{t:t+Δ} = a_{t:t+Δ})`` over ``[start, end)``.

    ``values`` is either one value per target (held constant) or a ``(Δ, |S|)`` nested list.
    """

    kind: ClassVar[str] = "interval"
    start: int
    end: int
    targets: list[int]
    values: list[float] | list[list[float]]

    @property
    def variables(self) -> list[int]:
        return sorted(set(self.targets))

    def _value_array(self) -> np.ndarray:
        v = np.asarray(self.values, dtype=float)
        n = self.end - self.start
        if v.ndim == 1:
            v = np.broadcast_to(v, (n, len(self.targets)))
        if v.shape != (n, len(self.targets)):
            raise ValueError(f"values shape {v.shape} != ({n}, {len(self.targets)})")
        return v

    def to_schedule(self, T: int, k: int) -> DenseSchedule:
        if not 0 <= self.start < self.end <= T:
            raise IndexError(f"interval [{self.start}, {self.end}) invalid for T={T}")
        s = DenseSchedule.empty(T, k)
        rows = np.arange(self.start, self.end)[:, None]
        s.mask[rows, self.targets] = True
        s.value[rows, self.targets] = self._value_array()
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "targets": list(self.targets),
            "values": np.asarray(self.values, dtype=float).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntervalIntervention:
        return cls(start=d["start"], end=d["end"], targets=d["targets"], values=d["values"])


@dataclass
class ParametricPulse(TemporalAction):
    """Soft additive pulse ``a_t = φ(t; α, τ, Δ)`` added to the structural equations of ``targets``.

    Unlike ``do()``, a pulse keeps the mechanism and the factual innovations in place and shifts
    them: ``x_t = F(Pa(x_t)) + ε_t + a_t``. ``shape`` is ``"rect"`` or ``"raised_cosine"``.
    """

    kind: ClassVar[str] = "pulse"
    start: int
    duration: int
    targets: list[int]
    amplitude: float | list[float]
    shape: str = "raised_cosine"

    @property
    def variables(self) -> list[int]:
        return sorted(set(self.targets))

    def profile(self) -> np.ndarray:
        n = self.duration
        if self.shape == "rect":
            return np.ones(n)
        if self.shape == "raised_cosine":
            return 0.5 * (1 - np.cos(2 * np.pi * (np.arange(n) + 1) / (n + 1)))
        raise ValueError(f"unknown pulse shape {self.shape!r}")

    def to_schedule(self, T: int, k: int) -> DenseSchedule:
        if self.duration < 1 or not 0 <= self.start < self.start + self.duration <= T:
            raise IndexError(f"pulse [{self.start}, {self.start + self.duration}) invalid")
        s = DenseSchedule.empty(T, k)
        amp = np.broadcast_to(np.asarray(self.amplitude, float), (len(self.targets),))
        rows = np.arange(self.start, self.start + self.duration)[:, None]
        s.shift[rows, self.targets] = self.profile()[:, None] * amp[None, :]
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start,
            "duration": self.duration,
            "targets": list(self.targets),
            "amplitude": self.amplitude,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParametricPulse:
        return cls(
            start=d["start"],
            duration=d["duration"],
            targets=d["targets"],
            amplitude=d["amplitude"],
            shape=d.get("shape", "raised_cosine"),
        )
