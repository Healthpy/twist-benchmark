"""The benchmark explanation object and the reproducibility/provenance schemas.

An explanation is ``e = (A, x̂_{0:T}, y*, C, r, u)``: declared action plan, proposed trajectory,
requested target, invoked constraints, mechanism/rollout model used by the explainer, and
uncertainty / failure report. Fields an explainer does not provide stay ``None``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from twist.actions.schemas import ActionPlan


class ExplanationStatus(StrEnum):
    OK = "ok"
    ABSTAIN = "abstain"  # no action / no edit
    NO_TARGET_FLIP = "no_target_flip"
    NO_VALID_ACTION = "no_valid_action"
    INFEASIBLE_ACTION = "infeasible_action"
    TIMEOUT = "timeout"
    NUMERICAL_INSTABILITY = "numerical_instability"
    INVALID_SERIALIZATION = "invalid_serialization"
    BUDGET_EXCEEDED = "budget_exceeded"  # model-query budget exhausted
    INVALID_OUTPUT = "invalid_output"  # wrong shape / unusable output from the method
    METHOD_ERROR = "method_error"  # the method raised; message kept in diagnostics


@dataclass
class Explanation:
    explainer: str
    factual_id: str
    target: int
    proposed_trajectory: np.ndarray | None
    action: ActionPlan | None = None  # 𝒜 — None for trajectory-only explainers
    declares_actions: bool = False  # explainer type declares actions (evaluated under 'native')
    constraints: list[str] = field(default_factory=list)  # 𝒞
    mechanism: str | None = None  # r — mechanism/rollout model used by the explainer
    mechanism_access_regime: str = "R0"
    uncertainty: dict[str, Any] = field(default_factory=dict)  # u
    status: ExplanationStatus = ExplanationStatus.OK
    selection_objective: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 1
    selection_rank: int | None = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    explainer_seed: int | None = None
    runtime_seconds: float = 0.0
    explanation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def native_action(self) -> bool:
        return self.action is not None and self.action.native

    def to_dict(self, include_trajectory: bool = False) -> dict[str, Any]:
        d = {
            "explanation_id": self.explanation_id,
            "explainer": self.explainer,
            "factual_id": self.factual_id,
            "target_outcome": self.target,
            "native_action": self.native_action,
            "action": None if self.action is None else self.action.to_dict(),
            "constraints": self.constraints,
            "mechanism": self.mechanism,
            "mechanism_access_regime": self.mechanism_access_regime,
            "uncertainty": self.uncertainty,
            "status": self.status.value,
            "selection_objective": self.selection_objective,
            "candidate_count": self.candidate_count,
            "selection_rank": self.selection_rank,
            "diagnostics": self.diagnostics,
            "explainer_seed": self.explainer_seed,
            "runtime_seconds": self.runtime_seconds,
        }
        if include_trajectory and self.proposed_trajectory is not None:
            d["proposed_trajectory"] = self.proposed_trajectory.tolist()
        return d


class RunManifest(BaseModel):
    """Reproducibility metadata attached to every experiment artifact (plan §2.3)."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    world_id: str
    world_family: str
    world_seed: int
    classifier_architecture: str
    classifier_seed: int
    explainer: str
    explainer_seed: int | None
    action_semantics: str
    evaluation_seed: int
    dataset_hash: str
    config_hash: str
    git_commit: str
    timestamp_utc: str
    hardware: str
    runtime_seconds: float
    package_version: str


class ProvenanceRecord(BaseModel):
    """Explanation Provenance Record (plan §Week 3, task 4)."""

    model_config = ConfigDict(extra="allow")

    explanation_id: str
    factual_id: str
    target_outcome: int
    native_action: bool
    action_semantics: str
    action: dict[str, Any] | None
    selection_objective: dict[str, Any]
    active_constraints: list[str]
    candidate_count: int
    selection_rank: int | None
    mechanism_access_regime: str
    proposed_metrics: dict[str, Any]
    replayed_metrics: dict[str, Any]
    uncertainty: dict[str, Any]
    known_failure_modes: list[str]
