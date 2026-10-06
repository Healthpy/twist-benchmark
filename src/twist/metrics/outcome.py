"""World-relative success: the A/B/C decomposition (draft Eqs. 5–6) and action-grounded success.

    A = Pr[h(x̂) = y*]            model on the proposal
    B = Pr[h(x^{A,*}) = y*]      model on the replayed trajectory
    C = Pr[ℓ*(x^{A,*}) = y*]     benchmark-world outcome
    A − C = (A − B) + (B − C)

All rates are over the scorable population (flip candidates with a non-empty action). No-action
proposals abstain and are reported with the remaining denominator; they are never scored as
correct zero-effect interventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def abc_decomposition(
    h_proposal: np.ndarray,
    h_replay: np.ndarray,
    world_replay: np.ndarray,
    target: np.ndarray | int,
    scorable: np.ndarray,
    feasible: np.ndarray | None = None,
    replay_faithful: np.ndarray | None = None,
    invalid: np.ndarray | None = None,
) -> dict[str, Any]:
    """``invalid`` marks units whose action was rejected by validation; they are excluded from the
    scorable population and counted separately from abstentions (no action)."""
    target = np.broadcast_to(np.asarray(target), np.shape(scorable))
    s = np.asarray(scorable, bool)
    n_cand, n_s = int(s.size), int(s.sum())
    n_invalid = int(np.sum(invalid)) if invalid is not None else 0

    def rate(hit: np.ndarray) -> float:
        return float(np.mean(hit[s])) if n_s else float("nan")

    a_hit = np.asarray(h_proposal) == target
    b_hit = np.asarray(h_replay) == target
    c_hit = np.asarray(world_replay) == target
    A, B, C = rate(a_hit), rate(b_hit), rate(c_hit)
    out: dict[str, Any] = {
        "proposal_model_success": A,
        "replay_model_success": B,
        "replay_world_success": C,
        "delta_traj": A - B,
        "delta_outcome": B - C,
        "delta_total": A - C,
        "n_flip_candidates": n_cand,
        "n_scorable": n_s,
        "n_abstain": n_cand - n_s - n_invalid,
        "n_invalid_action": n_invalid,
        "frac_abstain": (n_cand - n_s - n_invalid) / n_cand if n_cand else float("nan"),
    }
    if feasible is not None:
        out["feasible_rate"] = rate(np.asarray(feasible, bool))
        out["safe_world_success"] = rate(np.asarray(feasible, bool) & c_hit)
    if replay_faithful is not None:
        # the proposal is what its action generates AND that action achieves the world target
        out["action_grounded_success"] = rate(np.asarray(replay_faithful, bool) & c_hit)
    return out
