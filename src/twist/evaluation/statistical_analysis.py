"""Statistical analysis (plan §Week 8, task 3; protocol v1.0 §8).

* **Statistical units.** world → classifier → factual unit → explainer seed. Records carry
  ``world_id``, ``architecture``, ``unit`` (unique within world × classifier) and
  ``explainer_seed``. ``estimate`` bootstraps all four levels hierarchically.
* **Paired comparisons.** Methods are compared on the *same* flip candidates (all methods of a run
  see the same units). Per unit the metric is averaged over seeds, and the difference A − B is
  bootstrapped hierarchically over worlds → classifiers → units. Reported: mean difference with a
  95% CI, paired standardized effect size ``d_z = mean(d)/sd(d)``, and the probability of
  superiority ``P(d > 0) + ½·P(d = 0)`` over units.
* **Intention-to-treat success.** For comparisons, a unit without a scorable action counts as a
  failure (``world_success_itt``). Denominators are therefore identical across methods.
* **Unpaired populations** (base vs shifted worlds, whose units differ) use a world-blocked
  bootstrap: worlds are resampled jointly, units independently within each population.
* **Pareto frontier.** Methods are compared on world success (higher) and imposed timesteps or cost
  (lower); a method is non-dominated if no other is at least as good on both and better on one.
  There is no scalar leaderboard.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from twist.metrics.aggregation import hierarchical_bootstrap

LEVELS = ("world_id", "architecture", "unit", "explainer_seed")
UNIT_LEVELS = ("world_id", "architecture", "unit")


def world_success_itt(r: dict[str, Any]) -> float:
    """1 if a scorable action reaches the world target, else 0 (no action counts as failure)."""
    return float(bool(r["scorable"]) and r["world_replay"] == r["target"])


def proposal_success_itt(r: dict[str, Any]) -> float:
    return float(bool(r["has_proposal"]) and r["h_proposal"] == r["target"])


def mean_of(fn: Callable[[dict], float]) -> Callable[[list[dict]], float]:
    def stat(rs: list[dict]) -> float:
        v = np.array([fn(r) for r in rs], float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    return stat


def estimate(
    records: list[dict],
    stat: Callable[[list[dict]], float],
    levels: tuple[str, ...] = LEVELS,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and 95% CI of ``stat`` under the full hierarchical bootstrap."""
    lv = tuple(lv for lv in levels if records and lv in records[0])
    return hierarchical_bootstrap(records, stat, lv, n_boot=n_boot, seed=seed)


def unit_values(records: list[dict], value: Callable[[dict], float]) -> dict[tuple, float]:
    """(world, classifier, unit) → value averaged over explainer seeds (NaNs ignored)."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        acc[tuple(r[k] for k in UNIT_LEVELS)].append(value(r))
    out = {}
    for k, v in acc.items():
        a = np.array(v, float)
        a = a[np.isfinite(a)]
        if a.size:
            out[k] = float(a.mean())
    return out


def paired_difference(
    records_a: list[dict],
    records_b: list[dict],
    value: Callable[[dict], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison A − B on the units both methods have a value for."""
    va, vb = unit_values(records_a, value), unit_values(records_b, value)
    keys = sorted(set(va) & set(vb))
    if not keys:
        return {
            "n_units": 0,
            "mean_diff": float("nan"),
            "ci": (float("nan"), float("nan")),
            "effect_size_dz": float("nan"),
            "prob_superiority": float("nan"),
            "n_worlds": 0,
        }
    rows = [
        {"world_id": k[0], "architecture": k[1], "unit": k[2], "d": va[k] - vb[k]} for k in keys
    ]
    d = np.array([r["d"] for r in rows])
    point, lo, hi = hierarchical_bootstrap(
        rows, lambda rs: float(np.mean([r["d"] for r in rs])), UNIT_LEVELS, n_boot=n_boot, seed=seed
    )
    sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    return {
        "n_units": len(keys),
        "n_worlds": len({k[0] for k in keys}),
        "mean_diff": point,
        "ci": (lo, hi),
        "effect_size_dz": point / sd if sd and np.isfinite(sd) and sd > 0 else float("nan"),
        "prob_superiority": float((d > 0).mean() + 0.5 * (d == 0).mean()),
    }


def blocked_difference(
    records_a: list[dict],
    records_b: list[dict],
    stat: Callable[[list[dict]], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """stat(A) − stat(B) for populations that share worlds but not units (world-blocked
    bootstrap: worlds jointly, units independently within each population)."""
    worlds = sorted({r["world_id"] for r in records_a} & {r["world_id"] for r in records_b})
    if not worlds:
        return float("nan"), float("nan"), float("nan")

    def by(rs: list[dict]) -> dict[str, dict[Any, list[dict]]]:
        out: dict[str, dict[Any, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rs:
            out[r["world_id"]][r["unit"]].append(r)
        return out

    ga, gb = by(records_a), by(records_b)
    point = stat([r for w in worlds for u in ga[w].values() for r in u]) - stat(
        [r for w in worlds for u in gb[w].values() for r in u]
    )
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ws = rng.choice(worlds, len(worlds))
        sa: list[dict] = []
        sb: list[dict] = []
        for w in ws:
            for g, sink in ((ga[w], sa), (gb[w], sb)):
                units = list(g.values())
                for i in rng.integers(0, len(units), len(units)):
                    sink.extend(units[i])
        boots.append(stat(sa) - stat(sb))
    b = np.array(boots, float)
    b = b[np.isfinite(b)]
    if not b.size:
        return float(point), float("nan"), float("nan")
    lo, hi = np.quantile(b, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Non-dominated methods for (higher-is-better y, lower-is-better x) points ``{m: (x, y)}``."""
    front = []
    for m, (x, y) in points.items():
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        dominated = any(
            (x2 <= x and y2 >= y) and (x2 < x or y2 > y)
            for m2, (x2, y2) in points.items()
            if m2 != m and np.isfinite(x2) and np.isfinite(y2)
        )
        if not dominated:
            front.append(m)
    return sorted(front)


def bootstrap_draws(
    records: list[dict], levels: tuple[str, ...] = LEVELS, n_boot: int = 1000, seed: int = 0
) -> list[np.ndarray]:
    """Record-index arrays of ``n_boot`` nested bootstrap draws (same scheme as
    ``hierarchical_bootstrap``). Draw once, evaluate many vectorized statistics on the indices.

    The hierarchy is flattened into contiguous node ranges per level, so each draw samples a whole
    level at once with numpy (milliseconds for ~10⁴ records and four levels).
    """
    lv = tuple(k for k in levels if records and k in records[0])
    n = len(records)
    if n == 0 or not lv:
        return [np.arange(n) for _ in range(n_boot)]
    codes = np.empty((n, len(lv)), int)  # per level, value → int in first-appearance order
    for d, key in enumerate(lv):
        ids: dict[Any, int] = {}
        codes[:, d] = [ids.setdefault(r[key], len(ids)) for r in records]
    order = np.lexsort(codes.T[::-1])
    sc = codes[order]
    # node id of every (sorted) record at each depth: a new node starts where its prefix changes
    change = np.vstack([np.ones(len(lv), bool), np.diff(sc, axis=0) != 0])
    new_node = np.logical_or.accumulate(change, axis=1)
    node = np.cumsum(new_node, axis=0) - 1  # (n, L)
    n_nodes = node[-1] + 1
    starts, counts = [], []  # children of each depth-d node: contiguous depth-(d+1) nodes / records
    for d in range(len(lv)):
        child = node[:, d + 1] if d + 1 < len(lv) else np.arange(n)
        first = np.r_[True, np.diff(child) != 0]
        parent_of_child = node[first, d]
        cnt = np.bincount(parent_of_child, minlength=n_nodes[d])
        starts.append(np.cumsum(cnt) - cnt)
        counts.append(cnt)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sel = rng.integers(0, n_nodes[0], n_nodes[0])
        for d in range(len(lv)):
            c = counts[d][sel]
            if d + 1 < len(lv):  # resample children with replacement
                sel = np.repeat(starts[d][sel], c) + (
                    rng.random(int(c.sum())) * np.repeat(c, c)
                ).astype(int)
            else:  # leaves: keep every record of each drawn leaf
                within = np.arange(int(c.sum())) - np.repeat(np.cumsum(c) - c, c)
                draws.append(order[np.repeat(starts[d][sel], c) + within])
    return draws


def column_estimates(
    columns: dict[str, tuple[np.ndarray, str]], draws: list[np.ndarray], alpha: float = 0.05
) -> dict[str, tuple[float, float, float]]:
    """Point estimate and percentile CI of each column's ``mean`` or ``median`` (NaN = not in the
    column's population, e.g. non-scorable units for conditional metrics)."""
    out = {}
    for name, (v, how) in columns.items():
        f = np.nanmedian if how == "median" else np.nanmean

        def stat(x: np.ndarray, f: Callable = f) -> float:
            return float(f(x)) if np.isfinite(x).any() else float("nan")

        point = stat(v)
        b = np.array([stat(v[d]) for d in draws], float)
        b = b[np.isfinite(b)]
        if not np.isfinite(point) or not b.size:
            out[name] = (point, float("nan"), float("nan"))
            continue
        lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
        out[name] = (point, float(lo), float(hi))
    return out
