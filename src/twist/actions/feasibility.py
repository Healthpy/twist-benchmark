"""Generic feasibility constraints evaluated on (factual, action, replayed trajectory).

Every constraint returns a :class:`ConstraintResult` with a readable message, so invalid actions
are rejected with a violation report rather than silently dropped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from twist.actions.costs import action_deltas, budget_cost
from twist.actions.schemas import DenseSchedule

# Continuous bounds are checked with a small tolerance: actions projected exactly onto a bound
# must not fail on floating-point round-off (e.g. (x + θ) − x ≠ θ by ~1e-16).
FEAS_TOL = 1e-9


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    violation: float = 0.0
    message: str = ""


@dataclass
class FeasibilityReport:
    results: list[ConstraintResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def violations(self) -> list[ConstraintResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "results": [r.__dict__ for r in self.results],
        }


class Constraint(ABC):
    name: str = "constraint"

    @abstractmethod
    def evaluate(
        self, factual: np.ndarray, action: DenseSchedule, replayed: np.ndarray | None
    ) -> ConstraintResult: ...

    def _ok(self) -> ConstraintResult:
        return ConstraintResult(self.name, True)

    def _fail(self, violation: float, message: str) -> ConstraintResult:
        return ConstraintResult(self.name, False, float(violation), message)


@dataclass
class ImmutableVariables(Constraint):
    variables: list[int]
    name: str = "immutable_variables"

    def evaluate(self, factual, action, replayed):
        touched = sorted(set(np.flatnonzero(action.active.any(0))) & set(self.variables))
        if touched:
            return self._fail(len(touched), f"immutable variables intervened: {touched}")
        return self._ok()


@dataclass
class AmplitudeBounds(Constraint):
    """Bound the per-entry action magnitude ``|a_t|`` (deviation from factual / shift size)."""

    max_abs: float
    name: str = "amplitude_bounds"

    def evaluate(self, factual, action, replayed):
        a = np.abs(action_deltas(action, factual))[action.active]
        excess = float(np.max(a - self.max_abs, initial=0.0))
        if excess > FEAS_TOL:
            return self._fail(excess, f"action amplitude exceeds {self.max_abs} by {excess:.4g}")
        return self._ok()


@dataclass
class StateBounds(Constraint):
    """Replayed trajectory must stay within ``[lower, upper]``."""

    lower: float = -np.inf
    upper: float = np.inf
    name: str = "state_bounds"

    def evaluate(self, factual, action, replayed):
        if replayed is None:
            return self._fail(np.inf, "state bounds require a replayed trajectory")
        excess = float(max(np.max(replayed - self.upper), np.max(self.lower - replayed), 0.0))
        if excess > FEAS_TOL:
            return self._fail(excess, f"replayed state leaves [{self.lower}, {self.upper}]")
        return self._ok()


@dataclass
class MaxRateOfChange(Constraint):
    max_rate: float
    name: str = "max_rate_of_change"

    def evaluate(self, factual, action, replayed):
        if replayed is None:
            return self._fail(np.inf, "rate constraint requires a replayed trajectory")
        excess = float(np.max(np.abs(np.diff(replayed, axis=0)) - self.max_rate, initial=0.0))
        if excess > FEAS_TOL:
            return self._fail(excess, f"replayed rate of change exceeds {self.max_rate}")
        return self._ok()


@dataclass
class InterventionBudget(Constraint):
    budget: float
    unit_costs: list[float] | None = None
    name: str = "intervention_budget"

    def evaluate(self, factual, action, replayed):
        c = budget_cost(
            action, factual, None if self.unit_costs is None else np.array(self.unit_costs)
        )
        if c > self.budget + FEAS_TOL:
            return self._fail(c - self.budget, f"budget cost {c:.4g} > {self.budget}")
        return self._ok()


@dataclass
class MaxDuration(Constraint):
    max_timesteps: int
    name: str = "max_duration"

    def evaluate(self, factual, action, replayed):
        n = action.n_timesteps
        if n > self.max_timesteps:
            return self._fail(
                n - self.max_timesteps, f"{n} active timesteps > {self.max_timesteps}"
            )
        return self._ok()


@dataclass
class MinLeadTime(Constraint):
    """The action must start at least ``min_lead`` steps before the label site."""

    t_label: int
    min_lead: int
    name: str = "min_lead_time"

    def evaluate(self, factual, action, replayed):
        t0 = action.first_time
        if t0 is None:
            return self._ok()
        lead = self.t_label - t0
        if lead < self.min_lead:
            return self._fail(self.min_lead - lead, f"lead time {lead} < {self.min_lead}")
        return self._ok()


@dataclass
class ForbiddenCombination(Constraint):
    """These variables may not be intervened on together (at any time)."""

    variables: list[int]
    name: str = "forbidden_combination"

    def evaluate(self, factual, action, replayed):
        touched = set(np.flatnonzero(action.active.any(0)))
        if set(self.variables) <= touched:
            return self._fail(1.0, f"forbidden joint intervention on {sorted(self.variables)}")
        return self._ok()


CONSTRAINTS: dict[str, type[Constraint]] = {
    c.name: c
    for c in (
        ImmutableVariables,
        AmplitudeBounds,
        StateBounds,
        MaxRateOfChange,
        InterventionBudget,
        MaxDuration,
        MinLeadTime,
        ForbiddenCombination,
    )
}


def build_constraints(specs: list[dict[str, Any]]) -> list[Constraint]:
    """Instantiate constraints from config entries like ``{"name": "max_duration", "max_timesteps": 5}``."""
    out = []
    for spec in specs:
        spec = dict(spec)
        out.append(CONSTRAINTS[spec.pop("name")](**spec))
    return out


def check_feasibility(
    constraints: list[Constraint],
    factual: np.ndarray,
    action: DenseSchedule,
    replayed: np.ndarray | None = None,
) -> FeasibilityReport:
    return FeasibilityReport([c.evaluate(factual, action, replayed) for c in constraints])
