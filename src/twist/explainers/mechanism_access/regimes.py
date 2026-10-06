"""Mechanism-access regimes and the context builder that enforces them.

| Regime | Explainer information |
|---|---|
| R0 | black-box classifier and factual input only |
| R1 | + observational trajectories (plausibility / dynamics learning) |
| R2 | + causal graph G, no transition functions |
| R3 | + transition family and graph, unknown parameters |
| R4 | + estimated mechanism (Ĝ, F̂) |
| R5 | + oracle mechanism M* |
| R6 | + oracle mechanism, high-level action model and constraints |
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twist.classifiers.models import TorchClassifier
from twist.explainers.base import ExplainerContext
from twist.worlds.base_scm import TemporalSCM

REGIMES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6")


def regime_rank(regime: str) -> int:
    return REGIMES.index(regime)


def build_context(
    regime: str,
    classifier: TorchClassifier,
    world: TemporalSCM,
    train_x: np.ndarray | None = None,
    train_y: np.ndarray | None = None,
    constraints: list[Any] | None = None,
    budget: dict[str, Any] | None = None,
    mechanism: Any = None,
    mechanism_label: str = "",
) -> ExplainerContext:
    """Expose exactly what ``regime`` allows.

    R3/R4 need ``mechanism`` (fitted or misspecified by the runner from the observational data;
    see ``twist.mechanisms``). At R3 the true graph is also given; at R4 the explainer sees only
    the estimate and its own graph ``Ĝ``. R5/R6 receive the oracle world as their mechanism.
    """
    r = regime_rank(regime)
    if regime in ("R3", "R4") and mechanism is None:
        raise ValueError(f"{regime} needs an estimated mechanism (configure `mechanism:`)")
    obs_index = list(getattr(world, "obs_index", range(world.k)))
    outcome = [obs_index.index(c) for c in world.label.channels if c in obs_index]
    if r >= 5:
        mech, label = world, "oracle"
    elif r >= 3:
        mech, label = mechanism, mechanism_label or "estimated"
    else:
        mech, label = None, ""
    if r == 4:
        graph = np.asarray(mechanism.graph, bool).copy()
    elif r >= 2:
        g = world.dependency_graph()
        graph = g[np.ix_(obs_index, obs_index)] if r < 5 else g.copy()
    else:
        graph = None
    return ExplainerContext(
        classifier=classifier,
        regime=regime,
        train_x=train_x if r >= 1 else None,
        train_y=train_y if r >= 1 else None,
        graph=graph,
        world=world if r >= 5 else None,
        constraints=list(constraints or []) if r >= 6 else [],
        budget=dict(budget or {}),
        variable_names=list(world.variable_names),
        actionable=list(world.action_space().actionable) if r >= 6 else None,
        outcome_channels=outcome if r >= 2 else None,
        mechanism=mech,
        mechanism_label=label,
    )
