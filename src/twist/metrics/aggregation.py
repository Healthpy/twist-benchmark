"""Aggregation with uncertainty: percentile bootstrap and a nested (hierarchical) bootstrap over
world → … → unit (Week 7; Week 8 adds classifier/seed levels and paired comparisons)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile CI over finite values; NaNs are dropped (and must be reported)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def hierarchical_bootstrap(
    records: list[dict],
    stat: Callable[[list[dict]], float],
    levels: tuple[str, ...] = ("world_id", "unit"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``stat`` under a nested bootstrap.

    Resamples the first level (e.g. worlds) with replacement, then, within each drawn cluster,
    the next level (e.g. factual units), and so on. Records sharing the innermost key stay
    together (e.g. one unit's records). Week 8 extends the level list (classifiers, seeds) and adds
    paired differences.
    """
    point = stat(records)
    if not records or not np.isfinite(point):
        return float(point), float("nan"), float("nan")
    rng = np.random.default_rng(seed)

    def group(rs: list[dict], key: str) -> dict[Any, list[dict]]:
        out: dict[Any, list[dict]] = {}
        for r in rs:
            out.setdefault(r[key], []).append(r)
        return out

    def draw(rs: list[dict], depth: int) -> list[dict]:
        if depth == len(levels):
            return rs
        groups = list(group(rs, levels[depth]).values())
        picks = rng.integers(0, len(groups), len(groups))
        out: list[dict] = []
        for i in picks:
            out.extend(draw(groups[i], depth + 1))
        return out

    boots = np.array([stat(draw(records, 0)) for _ in range(n_boot)], float)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
