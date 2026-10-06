"""Traceability checks on Explanation Provenance Records (plan §Week 3, task 4)."""

from __future__ import annotations

from typing import Any

from twist.actions.schemas import ActionPlan

REQUIRED_PROVENANCE_FIELDS = (
    "explanation_id",
    "factual_id",
    "target_outcome",
    "native_action",
    "action_semantics",
    "action",
    "selection_objective",
    "active_constraints",
    "candidate_count",
    "selection_rank",
    "mechanism_access_regime",
    "proposed_metrics",
    "replayed_metrics",
    "uncertainty",
    "known_failure_modes",
)


def provenance_completeness(record: dict[str, Any]) -> dict[str, Any]:
    """Fraction of required fields present (``None`` counts as present-but-empty, absent does not)."""
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in record]
    return {
        "completeness": 1.0 - len(missing) / len(REQUIRED_PROVENANCE_FIELDS),
        "missing_fields": missing,
        "has_objective": bool(record.get("selection_objective")),
        "constraints_logged": "active_constraints" in record,
    }


def action_roundtrip_ok(plan: ActionPlan | None) -> bool:
    """Serialization/deserialization consistency of the declared action."""
    if plan is None:
        return True
    return ActionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
