"""Action-native recourse along the mechanism-access ladder (plan §Week 6, task 3).

One optimizer (``_SCMRecourse``: point / interval / pulse / control actions, Adam on
``CE(h(x̂), y*) + λ‖θ‖²``, candidate start times) is run with exactly the information a regime
allows. The declared action is native; the benchmark replays it in the true world.

| Explainer | Regime | Plans with | Acts on |
|---|---|---|---|
| ``GraphActionRecourse`` | R2 | no mechanism: the action is *imposed*, nothing propagates | ancestors of the outcome in ``G`` (∩ actionable when known) |
| ``EstimatedActionRecourse`` | R3 / R4 | the estimated (or misspecified) ``F̂`` from the context | all observed channels |
| ``OracleActionRecourse`` | R6 | the oracle ``F*`` and the action model: controls, bounds, constraints | the world's actionable set / controls |

Graph and estimated variants plan in *observed* coordinates; on partially observed worlds the
evaluator maps their actions to state channels. Their abduction uses ``F̂`` on the observations,
which is what a real analyst could do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twist.actions.feasibility import (
    AmplitudeBounds,
    ImmutableVariables,
    MaxDuration,
    MinLeadTime,
    check_feasibility,
)
from twist.actions.schemas import ActionSpace
from twist.explainers.base import ExplainerContext
from twist.explainers.structural_controls.scm_recourse import _SCMRecourse
from twist.worlds.integrity import ancestors


class GraphActionRecourse(_SCMRecourse):
    """R2: knows the causal graph but no transition functions. It intervenes only on ancestors of
    the outcome and must assume the edit is all that changes."""

    name = "GraphActionRecourse"
    min_regime = "R2"
    is_control = False
    role = "reference"
    exact_world_only = False
    coordinates = "observed"
    default_form = "interval"

    def _resolve_mechanism(self, ctx: ExplainerContext) -> tuple[Any, str]:
        if ctx.graph is None or ctx.outcome_channels is None:
            raise PermissionError(f"{self.name} needs the causal graph (R2+)")
        return None, "graph-only"

    def _action_space(self, ctx: ExplainerContext, T: int, k: int) -> ActionSpace:
        assert ctx.graph is not None
        anc = ancestors(ctx.graph, list(ctx.outcome_channels or [])) | set(
            ctx.outcome_channels or []
        )
        targets = sorted(a for a in anc if ctx.actionable is None or a in ctx.actionable)
        return ActionSpace(T, k, list(ctx.variable_names), targets or list(range(k)))


class EstimatedActionRecourse(_SCMRecourse):
    """R3/R4: plans with the estimated mechanism in the context. Under misspecification its
    proposals stop matching the true replay — the degradation the ladder measures."""

    name = "EstimatedActionRecourse"
    min_regime = "R3"
    is_control = False
    role = "reference"
    exact_world_only = False
    coordinates = "observed"
    default_form = "interval"

    def _resolve_mechanism(self, ctx: ExplainerContext) -> tuple[Any, str]:
        if ctx.mechanism is None:
            raise PermissionError(f"{self.name} needs an estimated mechanism (R3/R4)")
        if not ctx.mechanism.assumptions.get("exact_abduction", True):
            # at R5 the mechanism is the oracle world itself; on partially observed worlds it
            # cannot abduct from observations (estimates are fit in observation space and can)
            raise NotImplementedError(
                f"{self.name} abducts from observations with the oracle mechanism; partially "
                "observed worlds need posterior abduction (estimated-mechanism recourse runs there)"
            )
        return ctx.mechanism, ctx.mechanism_label or "estimated"

    def _action_space(self, ctx: ExplainerContext, T: int, k: int) -> ActionSpace:
        acts = list(range(k)) if ctx.actionable is None else list(ctx.actionable)
        return ActionSpace(T, k, list(ctx.variable_names), acts)


class OracleActionRecourse(_SCMRecourse):
    """R6 positive control: oracle mechanism plus the action model (controls, actionable set,
    bounds) and the feasibility constraints. It plans within the constraints it can project onto
    (amplitude, duration, lead time, immutability) and prefers candidates that pass all of them."""

    name = "OracleActionRecourse"
    min_regime = "R6"
    is_control = True
    default_form = "interval"

    def __init__(self, seed: int = 0, **params: Any):
        super().__init__(seed=seed, **params)

    def _limits(self, ctx: ExplainerContext) -> dict[str, Any]:
        lim: dict[str, Any] = {}
        for c in ctx.constraints:
            if isinstance(c, AmplitudeBounds):
                lim["max_abs"] = min(lim.get("max_abs", np.inf), c.max_abs)
            elif isinstance(c, MaxDuration):
                lim["max_duration"] = min(lim.get("max_duration", 10**9), c.max_timesteps)
            elif isinstance(c, MinLeadTime):
                lim["latest_start"] = min(lim.get("latest_start", 10**9), c.t_label - c.min_lead)
        return lim

    def _action_space(self, ctx: ExplainerContext, T: int, k: int) -> ActionSpace:
        assert ctx.world is not None
        space = ctx.world.action_space()
        immutable = {
            v for c in ctx.constraints if isinstance(c, ImmutableVariables) for v in c.variables
        }
        if immutable:
            space.actionable = [a for a in space.actionable if a not in immutable]
        return space

    def _feasible(
        self, ctx: ExplainerContext, x: np.ndarray, action: Any, x_hat: np.ndarray
    ) -> bool:
        if not ctx.constraints:
            return True
        assert ctx.world is not None
        sched = action.compile(ctx.world.action_space())
        return check_feasibility(ctx.constraints, x, sched, x_hat).passed

    def explain(self, factuals, factual_ids, targets, ctx):
        if (
            "action_form" not in self.params
            and ctx.world is not None
            and ctx.world.action_space().controls
        ):
            self.params["action_form"] = "control"
            self.name = f"{type(self).name}[control]"
        return super().explain(factuals, factual_ids, targets, ctx)
