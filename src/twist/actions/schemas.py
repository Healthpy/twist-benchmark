"""Core action abstractions, the dense schedule representation, and serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


class ActionValidationError(ValueError):
    """Raised when an action is malformed or outside its action space."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("; ".join(violations))


@dataclass
class ControlChannel:
    """A high-level control input ``u`` and how it enters the world mechanism.

    The control enters additively through the structural equations with a fixed input map:
    ``x_{t+delay} = F(Pa) + ε + loadings · u_t`` — i.e. ``F(x_t, u_t, ε_t)`` with a linear input
    channel. ``lower``/``upper`` bound ``u_t``; ``unit``, ``description``, ``cost`` (per unit of
    ``|u|`` per step, in ``cost_unit``) and ``side_effects`` feed domain action cards.
    """

    name: str
    loadings: list[float]
    delay: int = 0
    lower: float = -np.inf
    upper: float = np.inf
    unit: str = ""
    description: str = ""
    cost: float = 0.0
    cost_unit: str = ""
    side_effects: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loadings": [float(v) for v in self.loadings],
            "delay": self.delay,
            "lower": float(self.lower),
            "upper": float(self.upper),
            "unit": self.unit,
            "description": self.description,
            "cost": float(self.cost),
            "cost_unit": self.cost_unit,
            "side_effects": self.side_effects,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ControlChannel:
        return cls(**d)


@dataclass
class ActionSpace:
    """Where and how a world may be intervened on.

    ``actionable`` lists observed channels whose structural equations may be replaced;
    ``lower``/``upper`` bound the *values* an intervention may set (per channel); ``controls``
    are the world's high-level control inputs.
    """

    T: int
    k: int
    variable_names: list[str]
    actionable: list[int]
    lower: list[float] | None = None
    upper: list[float] | None = None
    controls: dict[str, ControlChannel] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": self.T,
            "k": self.k,
            "variable_names": self.variable_names,
            "actionable": self.actionable,
            "lower": self.lower,
            "upper": self.upper,
            "controls": {n: c.to_dict() for n, c in self.controls.items()},
        }


@dataclass
class DenseSchedule:
    """Per-timestep intervention arrays for one unit, shape ``(T, k)`` each.

    Under replay, entry ``(t, i)`` evaluates to::

        value[t, i]                              if mask[t, i]      (hard do())
        F_i(Pa(x_t)) + eps[t, i] + shift[t, i]   otherwise          (shift = soft intervention)
    """

    mask: np.ndarray
    value: np.ndarray
    shift: np.ndarray

    @classmethod
    def empty(cls, T: int, k: int) -> DenseSchedule:
        return cls(np.zeros((T, k), bool), np.zeros((T, k)), np.zeros((T, k)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape

    @property
    def active(self) -> np.ndarray:
        """Boolean ``(T, k)`` array of entries touched by the action."""
        return self.mask | (self.shift != 0)

    @property
    def active_timesteps(self) -> np.ndarray:
        return np.flatnonzero(self.active.any(axis=1))

    @property
    def first_time(self) -> int | None:
        ts = self.active_timesteps
        return int(ts[0]) if ts.size else None

    @property
    def last_time(self) -> int | None:
        ts = self.active_timesteps
        return int(ts[-1]) if ts.size else None

    @property
    def is_empty(self) -> bool:
        return not self.active.any()

    @property
    def n_timesteps(self) -> int:
        """Observed-variable schedule complexity of this action (timesteps, not coordinates)."""
        return int(self.active_timesteps.size)


class TemporalAction(ABC):
    """Base interface for every action form (plan §Week 2, task 1)."""

    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type[TemporalAction]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "kind" in cls.__dict__:
            TemporalAction._registry[cls.kind] = cls

    @abstractmethod
    def to_schedule(self, T: int, k: int) -> DenseSchedule:
        """Compile to the dense schedule consumed by replay."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Stable JSON-serializable representation including ``kind``."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> TemporalAction: ...

    @property
    @abstractmethod
    def variables(self) -> list[int]:
        """Observed channels this action touches."""

    def compile(self, space: ActionSpace) -> DenseSchedule:
        """Compile against a world's action space (needed by actions that use world interfaces)."""
        return self.to_schedule(space.T, space.k)

    def active_at(self, t: int, T: int, k: int) -> bool:
        return bool(self.to_schedule(T, k).active[t].any())

    def apply(self, state: np.ndarray, t: int, T: int, k: int) -> np.ndarray:
        """Return the post-intervention value at ``t`` given the structural value ``state``."""
        s = self.to_schedule(T, k)
        return np.where(s.mask[t], s.value[t], state + s.shift[t])

    def validate(self, space: ActionSpace) -> list[str]:
        """Return a list of human-readable violations (empty if valid)."""
        violations: list[str] = []
        bad = [v for v in self.variables if not 0 <= v < space.k]
        if bad:
            violations.append(f"variables {bad} outside channel range [0, {space.k})")
            return violations
        try:
            s = self.compile(space)
        except (IndexError, ValueError, KeyError, NotImplementedError) as exc:
            return [f"cannot compile to schedule: {exc}"]
        immutable = sorted(set(self.variables) - set(space.actionable))
        if immutable:
            names = [space.variable_names[i] for i in immutable]
            violations.append(f"non-actionable variables intervened: {names}")
        if not np.all(np.isfinite(s.value[s.mask])) or not np.all(np.isfinite(s.shift)):
            violations.append("non-finite intervention values")
        if space.lower is not None:
            lo = np.broadcast_to(np.asarray(space.lower), (space.T, space.k))
            if np.any(s.value[s.mask] < lo[s.mask]):
                violations.append("intervention value below action-space lower bound")
        if space.upper is not None:
            hi = np.broadcast_to(np.asarray(space.upper), (space.T, space.k))
            if np.any(s.value[s.mask] > hi[s.mask]):
                violations.append("intervention value above action-space upper bound")
        return violations

    def check(self, space: ActionSpace) -> None:
        violations = self.validate(space)
        if violations:
            raise ActionValidationError(violations)

    def cost(self, factual: np.ndarray) -> dict[str, float]:
        from twist.actions.costs import action_costs

        return action_costs(self.to_schedule(*factual.shape), factual)


def action_from_dict(d: dict[str, Any]) -> TemporalAction:
    try:
        cls = TemporalAction._registry[d["kind"]]
    except KeyError as exc:
        raise ActionValidationError([f"unknown action kind: {d.get('kind')!r}"]) from exc
    return cls.from_dict(d)


@dataclass
class ActionPlan:
    """An action plus the metadata that makes it auditable.

    ``native=True`` means the explainer itself declared the action. ``native=False`` means the
    benchmark reconstructed it from a trajectory with ``adapter`` — a diagnostic reading, never a
    claim attributed to the method (plan §Week 2, task 2).
    """

    action: TemporalAction
    native: bool
    semantics: str
    constraints: list[str] = field(default_factory=list)
    adapter: str | None = None
    fit_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        a = self.action.to_dict()
        return {
            "semantics": self.semantics,
            "is_native_action": self.native,
            "adapter_name": self.adapter,
            "fit_error": self.fit_error,
            "target_variables": self.action.variables,
            "constraints": list(self.constraints),
            "action": a,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionPlan:
        return cls(
            action=action_from_dict(d["action"]),
            native=d["is_native_action"],
            semantics=d["semantics"],
            constraints=list(d.get("constraints", [])),
            adapter=d.get("adapter_name"),
            fit_error=d.get("fit_error"),
        )
