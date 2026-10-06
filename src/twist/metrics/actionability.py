"""Actionability: feasibility rate and action cost summaries over evaluated explanations."""

from __future__ import annotations

import numpy as np

from twist.actions.feasibility import FeasibilityReport


def feasible_rate(reports: list[FeasibilityReport | None]) -> float:
    """Fraction of explanations with a declared/inferred action that pass every constraint."""
    vals = [r.passed for r in reports if r is not None]
    return float(np.mean(vals)) if vals else float("nan")
