"""Causal-discovery study (plan §Week 9 task 0, protocol v1.1 §4): discovered graphs as a mechanism
source for action-native recourse, scored against the known true graph.

Rows are ``EstimatedActionRecourse`` instances that differ only in the graph their mechanism is
fitted on (all with the masked *linear* fitter, so the fitter is held fixed):

* **R3 masked-linear** — the true graph (reference);
* **R4 discovery** — PCMCI at several significance levels, VAR-LiNGAM, and the Lasso VAR;
* **R4 corrupted** — the true graph with a controlled fraction of edges deleted / added
  (benchmark-side dose–response, like the misspecification sweep);
* **R5 oracle** — the true mechanism (upper reference).

Reported per row: graph recovery against the truth (summary precision / recall / F1 / SHD, lagged
F1, lag accuracy; mean over worlds with a world-bootstrap CI), mechanism replay error, and recourse
outcomes with four-level CIs. The **cost of not knowing the graph** is the paired ITT world-success
difference R3 masked-linear − R4 row on identical units.

Pre-declared acceptance checks:
1. every discovery/corruption mechanism has a scored graph (finite F1 and lagged F1) on every
   world it ran on;
2. every R4 row has a paired comparison with the R3 reference over shared units with a finite CI;
3. the corruption sweep spans graph error: mean summary F1 at the strongest corruption ≤ 0.6.

Pre-declared hypotheses (reported with CIs, not acceptance criteria):
* H1 — recourse degrades with graph error: Spearman ρ(graph F1, ITT world success) over
  (world, R4 row) points is > 0 (world-blocked bootstrap CI);
* H2 — not knowing the graph costs world success: R3 − R4 discovery > 0 for PCMCI at α = 0.01.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from twist.evaluation.multiworld_runner import MultiWorldConfig, run_multiworld
from twist.evaluation.statistical_analysis import paired_difference, world_success_itt
from twist.evaluation.study import load_study_records, method_table
from twist.utils.logging import get_logger
from twist.utils.seeds import derive_seed
from twist.utils.serialization import utc_timestamp, write_json

log = get_logger(__name__)
GRAPH_KEYS = (
    "graph_precision",
    "graph_recall",
    "graph_f1",
    "graph_shd",
    "graph_lag_f1",
    "graph_lag_accuracy",
)
MAX_F1_AT_STRONGEST = 0.6


class DiscoveryConfig(MultiWorldConfig):
    n_boot: int = 1000
    reference: str = "ActionRecourse[R3 masked-linear]"
    hypothesis_row: str = "ActionRecourse[R4 pcmci a=0.01]"


def _mech_label(r: dict) -> str:
    parts = (r.get("mechanism") or "").split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else ""


def _world_ci(vals: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not v.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def build_discovery_report(cfg: DiscoveryConfig, run_dirs: list[Path]) -> Path:
    records, infos = load_study_records(run_dirs)
    records = [r for r in records if r["method_reading"]]
    qual = {
        (i["world_id"], label): q for i in infos for label, q in (i.get("mechanisms") or {}).items()
    }
    table = method_table(records, cfg.n_boot)
    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_m[r["explainer"]].append(r)
    rows = []
    for m, rs in sorted(by_m.items()):
        regime = rs[0]["mechanism_access_regime"]
        label = _mech_label(rs[0])
        worlds = sorted({r["world_id"] for r in rs})
        qs = [q for w in worlds if (q := qual.get((w, label)))]
        row: dict[str, Any] = {
            "method": m,
            "regime": regime,
            "mechanism": label or "oracle",
            "n_worlds": len(worlds),
            "n_scored_worlds": len(qs),
        }
        for k in (*GRAPH_KEYS, "replay_error"):
            row[k] = _world_ci([q.get(k) for q in qs], cfg.n_boot, derive_seed(0, m, k))
        disc = [q.get("discovery") or {} for q in qs]
        row["seconds"] = (
            float(np.mean([d.get("seconds", np.nan) for d in disc])) if disc else float("nan")
        )
        row["n_instantaneous"] = [d.get("n_instantaneous") for d in disc]
        for k in ("world_success_itt", "C", "E_replay_median", "feasible", "action_timesteps"):
            row[k] = table[m][k]
        if m != cfg.reference and cfg.reference in by_m:
            row["cost_vs_reference"] = paired_difference(
                by_m[cfg.reference], rs, world_success_itt, cfg.n_boot, seed=derive_seed(0, m)
            )
        rows.append(row)
    order = {"R3": 0, "R4": 1, "R5": 2, "R6": 3}
    rows.sort(key=lambda r: (order.get(r["regime"], 9), "corrupted" in r["mechanism"], r["method"]))

    # dose–response points: (world, R4 row) → (summary F1, ITT world success)
    pts = []
    for row in rows:
        if row["regime"] != "R4":
            continue
        for w in sorted({r["world_id"] for r in by_m[row["method"]]}):
            q = qual.get((w, row["mechanism"]))
            if not q:
                continue
            ws = [world_success_itt(r) for r in by_m[row["method"]] if r["world_id"] == w]
            pts.append(
                {
                    "world_id": w,
                    "method": row["method"],
                    "f1": q["graph_f1"],
                    "lag_f1": q.get("graph_lag_f1"),
                    "success": float(np.mean(ws)),
                    "corrupted": "corrupted" in row["mechanism"],
                }
            )
    rho = _spearman(np.array([p["f1"] for p in pts]), np.array([p["success"] for p in pts]))
    worlds = sorted({p["world_id"] for p in pts})
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(cfg.n_boot):
        drawn = list(rng.choice(worlds, len(worlds))) if worlds else []
        sel = [p for w in drawn for p in pts if p["world_id"] == w]
        boots.append(
            _spearman(np.array([p["f1"] for p in sel]), np.array([p["success"] for p in sel]))
        )
    b = np.array(boots, float)
    b = b[np.isfinite(b)]
    rho_ci = (rho, *(np.quantile(b, [0.025, 0.975]).tolist() if b.size else [np.nan, np.nan]))

    # checks
    disc_rows = [r for r in rows if r["regime"] == "R4" and r["mechanism"] != "estimated:lasso"]
    unscored = [
        r["method"]
        for r in disc_rows
        if r["n_scored_worlds"] < r["n_worlds"]
        or not np.isfinite(r["graph_f1"][0])
        or not np.isfinite(r["graph_lag_f1"][0])
    ]
    r4 = [r for r in rows if r["regime"] == "R4"]
    unpaired = [
        r["method"]
        for r in r4
        if not r.get("cost_vs_reference") or not np.isfinite(r["cost_vs_reference"]["ci"][0])
    ]
    corr = [r for r in rows if "corrupted" in r["mechanism"]]
    strongest = min(corr, key=lambda r: r["graph_f1"][0]) if corr else None
    checks = [
        {
            "check": "every_discovery_graph_scored",
            "passed": bool(disc_rows) and not unscored,
            "detail": f"{len(disc_rows)} discovery/corruption rows; unscored: {unscored or 'none'}",
        },
        {
            "check": "cost_of_graph_is_paired",
            "passed": bool(r4) and not unpaired,
            "detail": f"{len(r4)} R4 rows vs `{cfg.reference}`; unpaired: {unpaired or 'none'}",
        },
        {
            "check": "corruption_spans_graph_error",
            "passed": strongest is not None and strongest["graph_f1"][0] <= MAX_F1_AT_STRONGEST,
            "detail": (
                f"lowest mean F1 {strongest['graph_f1'][0]:.2f} ({strongest['method']})"
                if strongest
                else "no corruption rows"
            ),
        },
    ]
    hyp_row = next((r for r in rows if r["method"] == cfg.hypothesis_row), None)
    hyps = [
        {
            "hypothesis": "H1: recourse degrades with graph error (Spearman ρ(F1, success) > 0)",
            "estimate": rho_ci,
            "supported": bool(np.isfinite(rho_ci[1]) and rho_ci[1] > 0),
        },
        {
            "hypothesis": f"H2: not knowing the graph costs world success (R3 − {cfg.hypothesis_row} > 0)",
            "estimate": (
                (hyp_row["cost_vs_reference"]["mean_diff"], *hyp_row["cost_vs_reference"]["ci"])
                if hyp_row and hyp_row.get("cost_vs_reference")
                else (np.nan, np.nan, np.nan)
            ),
            "supported": bool(
                hyp_row
                and hyp_row.get("cost_vs_reference")
                and hyp_row["cost_vs_reference"]["ci"][0] > 0
            ),
        },
    ]
    out = Path(cfg.output_dir) / cfg.name / "report"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    figs = [_scatter(pts, rows, out / f"{stem}__graph_vs_success.png")]
    path = out / f"{stem}.md"
    path.write_text(_markdown(cfg, infos, rows, checks, hyps, figs), encoding="utf-8")
    write_json(
        out / f"{stem}.json",
        {
            "checks": checks,
            "hypotheses": hyps,
            "rows": rows,
            "points": pts,
            "runs": [str(d) for d in run_dirs],
        },
    )
    return path


def _ci(t: Any, fmt: str = ".2f") -> str:
    if not isinstance(t, tuple | list) or t[0] is None or not np.isfinite(t[0]):
        return "–"
    p, lo, hi = t
    return f"{p:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]" if np.isfinite(lo) else f"{p:{fmt}}"


def _markdown(cfg, infos, rows, checks, hyps, figs) -> str:
    worlds = sorted({i["world_id"] for i in infos})
    L = [
        f"# Causal discovery as a mechanism source — `{cfg.name}`",
        "",
        f"{utc_timestamp()} · worlds {worlds} · classifiers "
        f"{sorted({i['architecture'] for i in infos})} · {cfg.n_factuals} units. Graph metrics: mean "
        "over worlds with a world-bootstrap 95% CI; outcomes: four-level hierarchical bootstrap. "
        "Every recourse row uses the masked linear fitter on its graph (R5: the true mechanism).",
        "",
        "## Acceptance checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    L += [
        f"| {c['check']} | {'**PASS**' if c['passed'] else '**FAIL**'} | {c['detail']} |"
        for c in checks
    ]
    L += [
        "",
        "## Pre-declared hypotheses",
        "",
        "| hypothesis | estimate [95% CI] | supported |",
        "|---|---|---|",
    ]
    L += [
        f"| {h['hypothesis']} | {_ci(tuple(h['estimate']), '+.3f')} | {'yes' if h['supported'] else 'no'} |"
        for h in hyps
    ]
    L += [
        "",
        "## Graph recovery against the true graph",
        "",
        "| regime | method | mechanism | precision | recall | F1 | SHD | lagged F1 | lag accuracy | mech. replay err | discovery s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["regime"] in ("R3", "R4"):
            secs = f"{r['seconds']:.1f}" if np.isfinite(r["seconds"]) else "–"
            L.append(
                f"| {r['regime']} | {r['method']} | {r['mechanism']} | {_ci(r['graph_precision'])} | "
                f"{_ci(r['graph_recall'])} | {_ci(r['graph_f1'])} | {_ci(r['graph_shd'], '.1f')} | "
                f"{_ci(r['graph_lag_f1'])} | {_ci(r['graph_lag_accuracy'])} | "
                f"{_ci(r['replay_error'], '.3f')} | {secs} |"
            )
    L += [
        "",
        "## Recourse outcomes and the cost of not knowing the graph",
        "",
        f"Cost = paired ITT world-success difference `{cfg.reference}` − row, on identical units.",
        "",
        "| regime | method | world success (ITT) | C | E_replay (median) | feasible | cost vs R3 [95% CI] | d_z |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r.get("cost_vs_reference")
        if c:
            dz_v = c["effect_size_dz"]
            cost = _ci((c["mean_diff"], *c["ci"]), "+.3f")
            dz = f"{dz_v:.2f}" if np.isfinite(dz_v) else "–"
        else:
            cost, dz = ("reference" if r["method"] == cfg.reference else "–"), "–"
        L.append(
            f"| {r['regime']} | {r['method']} | {_ci(r['world_success_itt'])} | {_ci(r['C'])} | "
            f"{_ci(r['E_replay_median'], '.2g')} | {_ci(r['feasible'])} | {cost} | {dz} |"
        )
    L += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs if p]
    return "\n".join(L) + "\n"


def _scatter(pts, rows, path: Path) -> Path | None:
    if not pts:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    methods = sorted({p["method"] for p in pts})
    for j, m in enumerate(methods):
        ps = [p for p in pts if p["method"] == m]
        ax.scatter(
            [p["f1"] for p in ps],
            [p["success"] for p in ps],
            s=14,
            marker="x" if ps[0]["corrupted"] else "o",
            color=f"C{j % 10}",
            label=m,
        )
    for r in rows:
        if r["regime"] in ("R3", "R5"):
            ax.axhline(r["world_success_itt"][0], ls="--", lw=0.8, color="grey")
            ax.annotate(
                r["method"],
                (0.01, r["world_success_itt"][0]),
                fontsize=6,
                xycoords=("axes fraction", "data"),
                va="bottom",
            )
    ax.set_xlabel("summary-graph F1 of the mechanism's graph (vs true)")
    ax.set_ylabel("world success (ITT), per world")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_discovery(cfg: DiscoveryConfig) -> Path:
    return build_discovery_report(cfg, run_multiworld(cfg))


def load_discovery_config(path: str | Path) -> DiscoveryConfig:
    from twist.config import load_yaml

    return DiscoveryConfig.model_validate(load_yaml(path))
