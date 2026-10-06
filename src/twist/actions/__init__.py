"""Explicit intervention semantics (WP1).

An action is a declared intervention plan. Every observed-variable action compiles to a
:class:`~twist.actions.schemas.DenseSchedule` — per-timestep ``do()`` masks/values plus additive
soft-intervention shifts — which is the only representation the replay engine consumes.
"""

from twist.actions.domain_actions import HighLevelControl
from twist.actions.interval import IntervalIntervention, ParametricPulse
from twist.actions.observed_variable import PointIntervention, ScheduleIntervention
from twist.actions.schemas import (
    ActionPlan,
    ActionSpace,
    ActionValidationError,
    ControlChannel,
    DenseSchedule,
    TemporalAction,
    action_from_dict,
)

__all__ = [
    "ActionPlan",
    "ActionSpace",
    "ActionValidationError",
    "ControlChannel",
    "DenseSchedule",
    "HighLevelControl",
    "IntervalIntervention",
    "ParametricPulse",
    "PointIntervention",
    "ScheduleIntervention",
    "TemporalAction",
    "action_from_dict",
]
