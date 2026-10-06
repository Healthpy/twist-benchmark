"""Tolerance sensitivity (plan §Week 7, task 3).

The change predicate ``|x̂ − x| > τ`` defines ``t0``, sparsity, the action support and ``D``.
``tolerance_profile`` recomputes them from saved proposals at ``τ ∈ taus × channel scale``
(per-channel std of the evaluated population) and reports hard Pearl CF-faith, ``D`` and the
unchanged-entry fraction per method. ``ranking_stability`` asks whether methods keep their order
by ``D`` across tolerances (Kendall τ against the reference tolerance).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from twist.metrics.conventional import sparsity
from twist.metrics.mechanism import cf_faith, do_complexity

TAUS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
REFERENCE_TAU = 1e-3


def tolerance_profile(
    world: Any,
    factuals: np.ndarray,
    proposals: dict[str, np.ndarray],
    taus: tuple[float, ...] = TAUS,
) -> dict[str, dict[float, dict[str, float]]]:
    """``{method: {τ: {cf_faith, D, unchanged_entries, n_scorable}}}`` for exact worlds."""
    if not world.assumptions.get("exact_abduction", True):
        return {}
    scale = factuals.reshape(-1, factuals.shape[-1]).std(0) + 1e-12
    out: dict[str, dict[float, dict[str, float]]] = {}
    for name, xh in proposals.items():
        ok = np.all(np.isfinite(xh), axis=(1, 2))
        if not ok.any():
            continue
        x, p = factuals[ok], xh[ok]
        prof = {}
        for tau in taus:
            tol = tau * scale
            f = cf_faith(world, x, p, "pearl", tol=float(tau * scale.mean()), change_tol=tol)
            hard = f["hard"][f["scorable"]]
            prof[tau] = {
                "cf_faith": float(np.mean(hard)) if hard.size else float("nan"),
                "D": float(np.mean(do_complexity(world, x, p, "pearl", tol))),
                "unchanged_entries": float(np.mean(sparsity(x, p, tol)["unchanged_entries"])),
                "n_scorable": float(f["scorable"].sum()),
            }
        out[name] = prof
    return out


def ranking_stability(
    profile: dict[str, dict[float, dict[str, float]]],
    key: str = "D",
    reference: float = REFERENCE_TAU,
) -> dict[float, float]:
    """Kendall τ between the method ranking by ``key`` at each tolerance and at ``reference``."""
    from scipy.stats import kendalltau

    methods = sorted(profile)
    if len(methods) < 3:
        return {}
    ref = [profile[m][reference][key] for m in methods]
    out = {}
    for tau in next(iter(profile.values())):
        cur = [profile[m][tau][key] for m in methods]
        out[tau] = float(kendalltau(ref, cur).statistic)
    return out
