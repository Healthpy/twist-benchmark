"""Explainer interface and the information context handed to explainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from twist.classifiers.models import TorchClassifier
from twist.evaluation.result_schema import Explanation
from twist.worlds.base_scm import TemporalSCM


@dataclass
class ExplainerContext:
    """What an explainer may see. Populated by ``mechanism_access.regimes.build_context`` so that
    a regime's information limits are enforced by construction rather than by convention."""

    classifier: TorchClassifier
    regime: str = "R0"
    train_x: np.ndarray | None = None  # R1+: observational trajectories
    train_y: np.ndarray | None = None  # R1+: their labels (reference data for retrieval methods)
    graph: np.ndarray | None = None  # R2+
    world: TemporalSCM | None = None  # R5+: oracle mechanism (incl. abduction)
    constraints: list[Any] = field(default_factory=list)  # R6: action model + constraints
    budget: dict[str, Any] = field(default_factory=dict)
    variable_names: list[str] = field(default_factory=list)  # feature metadata (always)
    outcome_channels: list[int] | None = None  # R2+: graph nodes the outcome is defined on
    mechanism: Any = None  # R3/R4: estimated (or misspecified) F̂; R5+: the oracle world
    mechanism_label: str = ""  # e.g. "oracle", "estimated:lasso", "misspecified:delay-0.5"
    actionable: list[int] | None = None  # mutability metadata (R6 only)


class Explainer(ABC):
    name: str = "explainer"
    native_actions: bool = False  # does this method declare actions itself?
    min_regime: str = "R0"
    is_control: bool = False  # structural controls are not competing explainers
    # compatibility-matrix metadata (docs/compatibility_matrix.md)
    role: str | None = None  # override for the matrix: "baseline" / "reference" / "control"
    source: str = "twist"  # "twist" (in-house) or "cfts@<commit>"
    upstream: str = ""  # upstream function, for third-party adapters
    uses_reference_data: bool = False
    uses_gradients: bool = False
    stochastic: bool = False
    exact_world_only: bool = False  # needs abduction from observations

    def __init__(self, seed: int = 0, **params: Any):
        self.seed = seed
        self.params = params

    @abstractmethod
    def explain(
        self,
        factuals: np.ndarray,
        factual_ids: list[str],
        targets: np.ndarray,
        ctx: ExplainerContext,
    ) -> list[Explanation]: ...
