"""World label rules ``ℓ*`` (plan §Week 4, task 4).

| Type | ``ℓ*(x) = 1{…}`` | Typical use |
|---|---|---|
| ``threshold`` | ``x[t_label, j] > θ`` (terminal when ``t_label = −1``, interior otherwise) | draft labels |
| ``exposure`` | ``Σ_{t=t1}^{t2} w_t x[t, j] > θ`` (normalized weights) | cumulative dose / load |
| ``onset`` | ``max_{t1 ≤ t ≤ t2} x[t, j] > θ`` (threshold crossed before the horizon) | events, overload |
| ``multi`` | two linked conditions combined with ``and`` / ``or`` | composite outcomes |

Every label exposes ``score`` (continuous; label = ``score > 0`` after fitting), ``fit`` (threshold
from a population, chosen for class balance), ``site`` (last timestep it depends on, used for
label distance), and ``support`` (the ``(T, k)`` entries it reads, used by causal-route checks).
Times may be negative (counted from the end).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


class LabelFunction(ABC):
    type: ClassVar[str]
    _registry: ClassVar[dict[str, Any]] = {}  # name → LabelFunction subclass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "type" in cls.__dict__:
            LabelFunction._registry[cls.type] = cls

    @abstractmethod
    def margin(self, x: np.ndarray) -> np.ndarray:
        """Signed distance to the decision boundary; the label is ``margin > 0``."""

    @abstractmethod
    def fit(self, x: np.ndarray) -> LabelFunction: ...

    @abstractmethod
    def support(self, T: int, k: int) -> np.ndarray:
        """Boolean ``(T, k)`` mask of the entries the label reads."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @property
    @abstractmethod
    def fitted(self) -> bool: ...

    @property
    def channels(self) -> list[int]:
        return sorted({int(c) for c in self._channels()})

    @abstractmethod
    def _channels(self) -> list[int]: ...

    def site(self, T: int) -> int:
        """Last timestep the label depends on (label distance is measured to it)."""
        return int(np.flatnonzero(self.support(T, max(self.channels) + 1).any(axis=1)).max())

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.margin(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("label threshold not fit; call .fit(population) first")
        return (self.margin(x) > 0).astype(np.int64)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LabelFunction:
        d = dict(d)
        kind = d.pop("type", "threshold")
        return LabelFunction._registry[kind]._from_fields(d)

    @classmethod
    def _from_fields(cls, d: dict[str, Any]) -> LabelFunction:
        return cls(**d)


def _window(t1: int, t2: int, T: int) -> tuple[int, int]:
    a, b = t1 % T, t2 % T
    if a > b:
        raise ValueError(f"empty label window [{t1}, {t2}] for T={T}")
    return a, b


@dataclass
class ThresholdLabel(LabelFunction):
    """``1{x[t_label, channel] > threshold}``; threshold = population median when fit."""

    type: ClassVar[str] = "threshold"
    channel: int = 0
    t_label: int = -1
    threshold: float | None = None

    @property
    def fitted(self) -> bool:
        return self.threshold is not None

    def _channels(self) -> list[int]:
        return [self.channel]

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return x[..., self.t_label % x.shape[-2], self.channel]

    def margin(self, x: np.ndarray) -> np.ndarray:
        return self.raw(x) - (self.threshold if self.threshold is not None else 0.0)

    def score(self, x: np.ndarray) -> np.ndarray:  # raw value, kept for backward compatibility
        return self.raw(x)

    def fit(self, x: np.ndarray, quantile: float = 0.5) -> ThresholdLabel:
        self.threshold = float(np.quantile(self.raw(x), quantile))
        return self

    def site(self, T: int) -> int:
        return self.t_label % T

    def support(self, T: int, k: int) -> np.ndarray:
        m = np.zeros((T, k), bool)
        m[self.t_label % T, self.channel] = True
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "channel": self.channel,
            "t_label": self.t_label,
            "threshold": self.threshold,
        }


@dataclass
class ExposureLabel(LabelFunction):
    """Weighted mean of ``x[:, channel]`` over ``[t_start, t_end]`` above a threshold.

    ``weights``: ``uniform`` or ``ramp`` (linearly increasing, later exposure counts more).
    """

    type: ClassVar[str] = "exposure"
    channel: int = 0
    t_start: int = -10
    t_end: int = -1
    weights: str = "uniform"
    threshold: float | None = None

    @property
    def fitted(self) -> bool:
        return self.threshold is not None

    def _channels(self) -> list[int]:
        return [self.channel]

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        a, b = _window(self.t_start, self.t_end, x.shape[-2])
        seg = x[..., a : b + 1, self.channel]
        n = b - a + 1
        w = np.ones(n) if self.weights == "uniform" else np.arange(1, n + 1, dtype=float)
        return seg @ (w / w.sum())

    def margin(self, x: np.ndarray) -> np.ndarray:
        return self.raw(x) - (self.threshold if self.threshold is not None else 0.0)

    def fit(self, x: np.ndarray, quantile: float = 0.5) -> ExposureLabel:
        self.threshold = float(np.quantile(self.raw(x), quantile))
        return self

    def support(self, T: int, k: int) -> np.ndarray:
        a, b = _window(self.t_start, self.t_end, T)
        m = np.zeros((T, k), bool)
        m[a : b + 1, self.channel] = True
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "channel": self.channel,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "weights": self.weights,
            "threshold": self.threshold,
        }


@dataclass
class OnsetLabel(LabelFunction):
    """Event onset: ``x[t, channel]`` exceeds the threshold at some ``t ∈ [t_start, t_end]``."""

    type: ClassVar[str] = "onset"
    channel: int = 0
    t_start: int = 0
    t_end: int = -1
    threshold: float | None = None

    @property
    def fitted(self) -> bool:
        return self.threshold is not None

    def _channels(self) -> list[int]:
        return [self.channel]

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        a, b = _window(self.t_start, self.t_end, x.shape[-2])
        return x[..., a : b + 1, self.channel].max(axis=-1)

    def margin(self, x: np.ndarray) -> np.ndarray:
        return self.raw(x) - (self.threshold if self.threshold is not None else 0.0)

    def fit(self, x: np.ndarray, quantile: float = 0.5) -> OnsetLabel:
        self.threshold = float(np.quantile(self.raw(x), quantile))
        return self

    def onset_time(self, x: np.ndarray) -> np.ndarray:
        """First crossing time per unit (``-1`` if none)."""
        x = np.asarray(x)
        a, b = _window(self.t_start, self.t_end, x.shape[-2])
        above = x[..., a : b + 1, self.channel] > (self.threshold or 0.0)
        return np.where(above.any(-1), a + above.argmax(-1), -1)

    def support(self, T: int, k: int) -> np.ndarray:
        a, b = _window(self.t_start, self.t_end, T)
        m = np.zeros((T, k), bool)
        m[a : b + 1, self.channel] = True
        return m

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "channel": self.channel,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "threshold": self.threshold,
        }


@dataclass
class MultiOutputLabel(LabelFunction):
    """Two (or more) linked target conditions combined with ``and`` / ``or``.

    The margin is the min (``and``) or max (``or``) of the conditions' margins. ``fit`` places each
    condition's threshold at the quantile that balances the combined label if the conditions were
    independent (``1 − 0.5^{1/n}`` for ``and``, ``0.5^{1/n}`` for ``or``).
    """

    type: ClassVar[str] = "multi"
    conditions: list[LabelFunction] = field(default_factory=list)
    combine: str = "and"

    def __post_init__(self) -> None:
        self.conditions = [
            c if isinstance(c, LabelFunction) else LabelFunction.from_dict(c)
            for c in self.conditions
        ]
        if self.combine not in ("and", "or"):
            raise ValueError("combine must be 'and' or 'or'")
        if len(self.conditions) < 2:
            raise ValueError("a multi-output label needs at least two conditions")

    @property
    def fitted(self) -> bool:
        return all(c.fitted for c in self.conditions)

    def _channels(self) -> list[int]:
        return [ch for c in self.conditions for ch in c.channels]

    def margin(self, x: np.ndarray) -> np.ndarray:
        m = np.stack([c.margin(x) for c in self.conditions])
        return m.min(0) if self.combine == "and" else m.max(0)

    def fit(self, x: np.ndarray) -> MultiOutputLabel:
        n = len(self.conditions)
        q = 1 - 0.5 ** (1 / n) if self.combine == "and" else 0.5 ** (1 / n)
        for c in self.conditions:
            c.fit(x, quantile=q)  # type: ignore[call-arg]
        return self

    def site(self, T: int) -> int:
        return max(c.site(T) for c in self.conditions)

    def support(self, T: int, k: int) -> np.ndarray:
        return np.logical_or.reduce([c.support(T, k) for c in self.conditions])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "combine": self.combine,
            "conditions": [c.to_dict() for c in self.conditions],
        }


def make_label(spec: dict[str, Any] | None) -> LabelFunction:
    """Build a label from a config dict (``type`` defaults to ``threshold``)."""
    return LabelFunction.from_dict(spec or {"type": "threshold", "channel": 0, "t_label": -1})
