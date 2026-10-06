"""Plain-text/Markdown tables. Every outcome table carries readings, D and denominators."""

from __future__ import annotations

import math
from typing import Any

MAIN_COLUMNS = [
    ("explainer", "method"),
    ("reading", "reading"),
    ("validity", "V"),
    ("cf_faith_noiseless", "R-faith"),
    ("cf_faith_pearl", "P-faith"),
    ("D_pearl", "D"),
    ("replay_error_pearl", "E_rep"),
    ("proposal_model_success", "A"),
    ("replay_model_success", "B"),
    ("replay_world_success", "C"),
    ("delta_traj", "Δtraj"),
    ("delta_outcome", "Δout"),
    ("delta_total", "Δtotal"),
    ("feasible_rate", "feas"),
    ("median_label_distance", "t_label−t0"),
    ("n_scorable", "n_s"),
    ("n_abstain", "abst"),
    ("n_invalid_action", "inval"),
]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def markdown_table(
    rows: list[dict[str, Any]], columns: list[tuple[str, str]] = MAIN_COLUMNS
) -> str:
    head = "| " + " | ".join(h for _, h in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(key, "")) for key, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])
