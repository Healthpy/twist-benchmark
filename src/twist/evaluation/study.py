"""Pilot multi-world study (plan §Week 8, tasks 3–5; protocol v1.0 §8–9).

Runs worlds × classifiers × explainers × seeds through the multi-world runner and reports, with
**no scalar leaderboard**:

* a primary metric table: every headline metric with mean (or median), 95% CI from the four-level
  hierarchical bootstrap (worlds → classifiers → units → seeds) and its population;
* paired method comparisons on identical units (``paired_difference``; ITT world success; mean
  difference, CI, d_z, probability of superiority) and a forest plot;
* a Pareto frontier (world success ↑ vs action magnitude ↓);
* per-world distributions and a failure-mode taxonomy;
* pre-declared acceptance checks:
  1. positive controls meet their intended conditions (95% CI bound on the right side);
  2. negative controls fail at least one relevant criterion (CI bound on the failing side);
  3. method conclusions are paired: every pair of non-control methods has a paired comparison on
     ≥ 1 shared unit with a finite CI, and every headline cell has a CI;
  4. a draft finding replicates across independent worlds: the model–world gap Δtotal = A − C of
     the trajectory baselines (first-slice reading), paired within units, has a 95% CI above 0 in
     at least two worlds from at least two families.

Methods are read under their own semantics: action-native methods by the declared action,
trajectory methods by the first-slice reading (``first_change_atomic``); the full changed schedule
is reported as a sensitivity reading. Control conditions are checked on exact (observed-state)
worlds, where replay error and CF-faith are defined.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from twist.evaluation.multiworld_runner import MultiWorldConfig, run_multiworld
from twist.evaluation.statistical_analysis import (
    LEVELS,
    bootstrap_draws,
    column_estimates,
    paired_difference,
    pareto_front,
    world_success_itt,
)
from twist.utils.logging import get_logger
from twist.utils.seeds import derive_seed
from twist.utils.serialization import read_json, utc_timestamp, write_json

log = get_logger(__name__)
NATIVE, FIRST, FULL = "native", "first_change_atomic", "full_changed_schedule"
EXACT_FAMILIES = ("linear", "nonlinear", "spring", "regime")
KEEP = (
    "explanation_id",
    "explainer",
    "reading",
    "factual_id",
    "target",
    "status",
    "has_proposal",
    "scorable",
    "native_action",
    "h_proposal",
    "h_replay",
    "world_replay",
    "valid",
    "feasible",
    "replay_error_pearl",
    "cf_faith_pearl_hard",
    "cf_faith_scorable",
    "D_pearl",
    "action_timesteps",
    "ood_proposal",
    "ood_factual",
    "attribution",
    "explainer_seed",
    "runtime_seconds",
    "d_norm",
    "mechanism",
    "mechanism_access_regime",
)


class StudyConfig(MultiWorldConfig):
    n_boot: int = 1000
    reference: str = "NUN"  # paired comparisons: every method vs this one, plus all method pairs
    positive_controls: dict[str, list[list[Any]]] = Field(default_factory=dict)
    negative_controls: dict[str, list[list[Any]]] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------- loading
def load_study_records(run_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    """Slim records (method reading + full-schedule sensitivity) with the four statistical levels."""
    from twist.explainers.registry import EXPLAINERS

    records, infos = [], []
    for d in run_dirs:
        d = Path(d)
        if not (d / "run_info.json").exists():
            continue
        info = read_json(d / "run_info.json")
        cfg = read_json(d / "config.resolved.json")
        info.update(
            world_id=cfg["world"]["world_id"],
            family=cfg["world"]["family"],
            architecture=cfg["classifier"]["architecture"],
            n_factuals=cfg["n_factuals"],
        )
        infos.append(info)
        T = int(cfg["world"]["params"].get("T", 50))
        for r in read_json(d / "records.json"):
            method_reading = NATIVE if (r["native_action"] or r["reading"] == NATIVE) else FIRST
            if r["reading"] not in (method_reading, FULL):
                continue
            s = {k: r.get(k) for k in KEEP}
            s["cost"] = (r.get("costs") or {}).get("magnitude")
            s["constraints_failed"] = [
                c["name"]
                for c in (r.get("feasibility") or {}).get("results", [])
                if not c["passed"]
            ]
            s["replay_hash"] = r.get("replay_hash")
            s["action"] = r.get("action")
            s["T"] = T
            s["action_semantics"] = r.get("action_semantics")
            cls = EXPLAINERS.get(r["explainer"].split("[")[0])
            s.update(
                world_id=info["world_id"],
                family=info["family"],
                architecture=info["architecture"],
                unit=r["factual_id"].rsplit("/", 1)[1],
                explainer_seed=r.get("explainer_seed") or 0,
                run_id=info["run_id"],
                method_reading=r["reading"] == method_reading,
                is_control=bool(getattr(cls, "is_control", False)),
            )
            records.append(s)
    return records, infos


# ------------------------------------------------------------------------------------- metrics
def _f(r: dict, key: str) -> float:
    v = r.get(key)
    return float("nan") if v is None else float(v)


def _cols(rs: list[dict]) -> dict[str, tuple[np.ndarray, str]]:
    sc = np.array([bool(r["scorable"]) for r in rs])
    nan = np.where(sc, 0.0, np.nan)  # NaN outside the scorable population

    def scor(vals) -> np.ndarray:
        return np.asarray(vals, float) + nan

    return {
        "world_success_itt": (np.array([world_success_itt(r) for r in rs], float), "mean"),
        "validity_itt": (np.array([float(bool(r["valid"])) for r in rs], float), "mean"),
        "A": (scor([r["h_proposal"] == r["target"] for r in rs]), "mean"),
        "B": (scor([r["h_replay"] == r["target"] for r in rs]), "mean"),
        "C": (scor([r["world_replay"] == r["target"] for r in rs]), "mean"),
        "delta_total": (
            scor(
                [
                    float(r["h_proposal"] == r["target"]) - float(r["world_replay"] == r["target"])
                    for r in rs
                ]
            ),
            "mean",
        ),
        "feasible": (scor([float(bool(r["feasible"])) for r in rs]), "mean"),
        "safe_world_success": (
            np.array(
                [
                    float(
                        bool(r["scorable"])
                        and bool(r["feasible"])
                        and r["world_replay"] == r["target"]
                    )
                    for r in rs
                ]
            ),
            "mean",
        ),
        "E_replay_median": (scor([_f(r, "replay_error_pearl") for r in rs]), "median"),
        "cf_faith_pearl": (
            np.array(
                [_f(r, "cf_faith_pearl_hard") if r["cf_faith_scorable"] else np.nan for r in rs]
            ),
            "mean",
        ),
        "D_pearl": (np.array([_f(r, "D_pearl") for r in rs]), "mean"),
        "action_timesteps": (scor([r["action_timesteps"] for r in rs]), "mean"),
        "action_fraction_median": (scor([r["action_timesteps"] / r["T"] for r in rs]), "median"),
        "cost": (scor([_f(r, "cost") for r in rs]), "mean"),
        "ood_delta": (np.array([_f(r, "ood_proposal") - _f(r, "ood_factual") for r in rs]), "mean"),
        "failure_rate": (np.array([float(r["status"] != "ok") for r in rs]), "mean"),
    }


HEADLINE_COLUMNS = (
    "world_success_itt",
    "validity_itt",
    "A",
    "B",
    "C",
    "delta_total",
    "feasible",
    "safe_world_success",
    "E_replay_median",
    "cf_faith_pearl",
    "D_pearl",
    "action_timesteps",
    "action_fraction_median",
    "cost",
    "ood_delta",
    "failure_rate",
)


def method_table(records: list[dict], n_boot: int, reading: str = "method") -> dict[str, dict]:
    """method → {metric: (point, lo, hi)}, plus population sizes."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["method_reading"] if reading == "method" else r["reading"] == reading:
            by[r["explainer"]].append(r)
    out: dict[str, dict[str, Any]] = {}
    for m, rs in sorted(by.items()):
        draws = bootstrap_draws(rs, LEVELS, n_boot, seed=derive_seed(0, m, reading))
        est: dict[str, Any] = dict(column_estimates(_cols(rs), draws))
        est.update(
            n=len(rs),
            n_scorable=sum(bool(r["scorable"]) for r in rs),
            n_units=len({(r["world_id"], r["architecture"], r["unit"]) for r in rs}),
            n_worlds=len({r["world_id"] for r in rs}),
            is_control=rs[0]["is_control"],
            native=bool(rs[0]["native_action"]),
        )
        out[m] = est
    return out


# --------------------------------------------------------------------------- pre-declared checks
DEFAULT_POSITIVE = {
    "PearlSCMRecourse": [["E_replay_median", "<=", 1e-9], ["cf_faith_pearl", ">=", 0.95]],
    "FullScheduleOracle": [["E_replay_median", "<=", 1e-9]],
    "OracleActionRecourse": [["C", ">=", 0.8], ["feasible", ">=", 0.9]],
}
DEFAULT_NEGATIVE = {  # each must fail at least one of these relevant criteria
    "NoActionAbstain": [["world_success_itt", "<=", 0.0]],
    "AdversarialTrajectoryEdit": [["cf_faith_pearl", "<=", 0.1], ["C", "<=", 0.5]],
    "PlausibleButMechanismWrongEdit": [
        ["E_replay_median", ">=", 1e-2],
        ["cf_faith_pearl", "<=", 0.1],
    ],
    "FullScheduleOracle": [["action_fraction_median", ">=", 0.9]],  # fails compactness (≥ 0.9·T)
}


def _meets(est: tuple[float, float, float], op: str, thr: float) -> bool:
    """CI-aware: the whole 95% interval is on the required side (point estimate if no CI)."""
    p, lo, hi = est
    if op == "<=":
        b = hi if np.isfinite(hi) else p
        return bool(np.isfinite(b) and b <= thr)
    b = lo if np.isfinite(lo) else p
    return bool(np.isfinite(b) and b >= thr)


def _conditions(est: dict[str, Any] | None, conds: list[list[Any]]) -> list[tuple]:
    """(metric, op, threshold, estimate, met) per condition; nothing is met without an estimate."""
    out = []
    for k, op, thr in conds:
        e = est.get(k) if est else None
        out.append((k, op, thr, e, e is not None and _meets(e, str(op), float(thr))))
    return out


def control_checks(exact_table: dict[str, dict], cfg: StudyConfig) -> list[dict[str, Any]]:
    pos = cfg.positive_controls or DEFAULT_POSITIVE
    neg = cfg.negative_controls or DEFAULT_NEGATIVE
    rows = []
    for m, conds in pos.items():
        res = _conditions(exact_table.get(m), conds)
        rows.append(
            {
                "control": m,
                "kind": "positive",
                "conditions": res,
                "passed": m in exact_table and all(r[4] for r in res),
            }
        )
    for m, conds in neg.items():  # "met" = fails the criterion as intended
        res = _conditions(exact_table.get(m), conds)
        rows.append(
            {
                "control": m,
                "kind": "negative",
                "conditions": res,
                "passed": m in exact_table and any(r[4] for r in res),
            }
        )
    return rows


def paired_comparisons(
    records: list[dict], methods: list[str], reference: str, n_boot: int
) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in records:
        if r["method_reading"]:
            by[r["explainer"]].append(r)
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1 :]]
    pairs.sort(key=lambda p: (reference not in p, p))
    out = []
    for a, b in pairs:
        d = paired_difference(by[a], by[b], world_success_itt, n_boot, seed=derive_seed(0, a, b))
        out.append({"a": a, "b": b, "metric": "world_success_itt", **d})
    return out


def draft_replication(records: list[dict], baselines: list[str], n_boot: int) -> dict[str, Any]:
    """Per world: paired within-unit Δtotal = 1{h(x̂)=y*} − 1{ℓ*(x^A)=y*} of the trajectory
    baselines (first-slice reading, scorable), averaged per unit over methods and seeds."""
    from twist.metrics.aggregation import hierarchical_bootstrap

    rs = [
        r
        for r in records
        if r["explainer"] in baselines and r["reading"] == FIRST and r["scorable"]
    ]
    per_world = {}
    for w in sorted({r["world_id"] for r in rs}):
        units: dict[tuple, list[float]] = defaultdict(list)
        fam = None
        for r in rs:
            if r["world_id"] == w:
                fam = r["family"]
                units[(r["architecture"], r["unit"])].append(
                    float(r["h_proposal"] == r["target"]) - float(r["world_replay"] == r["target"])
                )
        rows = [
            {"architecture": k[0], "unit": k[1], "d": float(np.mean(v))} for k, v in units.items()
        ]
        p, lo, hi = hierarchical_bootstrap(
            rows,
            lambda xs: float(np.mean([x["d"] for x in xs])),
            ("architecture", "unit"),
            n_boot,
            seed=derive_seed(0, w),
        )
        per_world[w] = {
            "family": fam,
            "delta_total": (p, lo, hi),
            "n_units": len(rows),
            "replicated": bool(np.isfinite(lo) and lo > 0),
        }
    rep = [w for w, v in per_world.items() if v["replicated"]]
    fams = {per_world[w]["family"] for w in rep}
    return {
        "per_world": per_world,
        "replicated_worlds": rep,
        "families": sorted(fams),
        "passed": len(rep) >= 2 and len(fams) >= 2,
    }


def failure_taxonomy(records: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r["method_reading"]:
            out[r["explainer"]][r.get("attribution") or "unknown"] += 1
            if r["status"] != "ok":
                out[r["explainer"]][f"status:{r['status']}"] += 1
    return {
        m: {
            k: v / sum(v2 for k2, v2 in c.items() if not k2.startswith("status:"))
            for k, v in sorted(c.items())
        }
        for m, c in out.items()
    }


def per_world_success(records: list[dict]) -> dict[str, dict[str, float]]:
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        if r["method_reading"]:
            acc[(r["explainer"], r["world_id"])].append(world_success_itt(r))
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (m, w), v in acc.items():
        out[m][w] = float(np.mean(v))
    return dict(out)


# -------------------------------------------------------------------------------------- report
def build_study_report(cfg: StudyConfig, run_dirs: list[Path]) -> Path:
    records, infos = load_study_records(run_dirs)
    log.info("study report: %d records from %d runs", len(records), len(infos))
    table = method_table(records, cfg.n_boot)
    full = method_table([r for r in records if not r["native_action"]], cfg.n_boot, reading=FULL)
    exact = [r for r in records if r["family"] in EXACT_FAMILIES]
    exact_table = method_table(exact, cfg.n_boot)
    methods = sorted(m for m, t in table.items() if not t["is_control"])
    baselines = sorted(m for m in methods if not table[m]["native"])
    comps = paired_comparisons(records, methods, cfg.reference, cfg.n_boot)
    replication = draft_replication(records, baselines, cfg.n_boot)
    controls = control_checks(exact_table, cfg)
    taxonomy = failure_taxonomy(records)
    pw = per_world_success(records)
    pareto_pts = {m: (table[m]["cost"][0], table[m]["world_success_itt"][0]) for m in methods}
    front = pareto_front(pareto_pts)

    missing_ci = [
        f"{m}/{k}"
        for m, t in table.items()
        for k in ("world_success_itt", "A", "C")
        if np.isfinite(t[k][0]) and not np.isfinite(t[k][1])
    ]
    unpaired = [
        f"{c['a']} vs {c['b']}" for c in comps if c["n_units"] == 0 or not np.isfinite(c["ci"][0])
    ]
    checks = [
        {
            "check": "positive_controls_pass",
            "passed": all(c["passed"] for c in controls if c["kind"] == "positive"),
            "detail": "; ".join(
                f"{c['control']}: {'pass' if c['passed'] else 'FAIL'}"
                for c in controls
                if c["kind"] == "positive"
            ),
        },
        {
            "check": "negative_controls_fail_a_relevant_criterion",
            "passed": all(c["passed"] for c in controls if c["kind"] == "negative"),
            "detail": "; ".join(
                f"{c['control']}: {'fails as intended' if c['passed'] else 'DOES NOT FAIL'}"
                for c in controls
                if c["kind"] == "negative"
            ),
        },
        {
            "check": "conclusions_are_paired",
            "passed": not unpaired
            and not missing_ci
            and len(comps) == len(methods) * (len(methods) - 1) // 2,
            "detail": f"{len(comps)} paired comparisons over shared units; unpaired/undefined: "
            f"{unpaired or 'none'}; headline cells without CI: {missing_ci or 'none'}",
        },
        {
            "check": "draft_finding_replicated_across_worlds",
            "passed": replication["passed"],
            "detail": f"Δtotal = A − C > 0 (95% CI) in {len(replication['replicated_worlds'])}/"
            f"{len(replication['per_world'])} worlds, families {replication['families']}",
        },
    ]
    out = Path(cfg.output_dir) / cfg.name / "report"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    figs = [
        _forest(comps, out / f"{stem}__paired.png"),
        _pareto(table, pareto_pts, front, out / f"{stem}__pareto.png"),
        _per_world(pw, table, infos, out / f"{stem}__per_world.png"),
        _replication_plot(replication, out / f"{stem}__replication.png"),
    ]
    md = _markdown(
        cfg,
        infos,
        table,
        full,
        exact_table,
        comps,
        replication,
        controls,
        taxonomy,
        front,
        checks,
        figs,
    )
    path = out / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    write_json(
        out / f"{stem}.json",
        {
            "checks": checks,
            "table": table,
            "full_schedule_table": full,
            "exact_table": exact_table,
            "paired": comps,
            "replication": replication,
            "controls": controls,
            "taxonomy": taxonomy,
            "per_world": pw,
            "pareto_front": front,
            "runs": [str(d) for d in run_dirs],
        },
    )
    return path


def _ci(t: Any, fmt: str = ".2f") -> str:
    if not isinstance(t, tuple | list) or not np.isfinite(t[0]):
        return "–"
    p, lo, hi = t
    return f"{p:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]" if np.isfinite(lo) else f"{p:{fmt}}"


def _markdown(
    cfg,
    infos,
    table,
    full,
    exact_table,
    comps,
    replication,
    controls,
    taxonomy,
    front,
    checks,
    figs,
) -> str:
    worlds = sorted({i["world_id"] for i in infos})
    archs = sorted({i["architecture"] for i in infos})
    fams = Counter(i["family"] for i in infos if i["architecture"] == archs[0]) if archs else {}
    L = [
        f"# Pilot multi-world study — `{cfg.name}`",
        "",
        f"{utc_timestamp()} · {len(worlds)} worlds ({dict(fams)}) × classifiers {archs} × "
        f"{cfg.n_factuals} units × seeds {cfg.explainer_seeds} · {len(infos)} runs. "
        f"95% CIs: hierarchical bootstrap over worlds → classifiers → units → seeds ({cfg.n_boot} draws). "
        "Trajectory methods are read by their first edited slice, action-native methods by their "
        "declared action. ‡ = control (never ranked). There is no aggregate score.",
        "",
        "## Week-8 acceptance checks",
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
        "## Primary metric table",
        "",
        "Means with 95% CIs; *ITT* = all flip candidates (no action = failure); A, B, C, Δtotal, "
        "feasible, E_replay, action size and cost are over scorable units; E_replay is a median.",
        "",
        "| method | units | world success (ITT) | validity (ITT) | A | B | C | Δtotal = A−C | feasible | "
        "safe-world success | E_replay (median) | CF-faith (Pearl) | D | action timesteps | cost | ΔOOD | failure rate |",
        "|---|" + "---|" * 16,
    ]
    order = sorted(table, key=lambda m: (table[m]["is_control"], table[m]["native"], m))
    for m in order:
        t = table[m]
        L.append(
            f"| {m}{' ‡' if t['is_control'] else ''} | {t['n_units']} | {_ci(t['world_success_itt'])} | "
            f"{_ci(t['validity_itt'])} | {_ci(t['A'])} | {_ci(t['B'])} | {_ci(t['C'])} | "
            f"{_ci(t['delta_total'])} | {_ci(t['feasible'])} | {_ci(t['safe_world_success'])} | "
            f"{_ci(t['E_replay_median'], '.2g')} | {_ci(t['cf_faith_pearl'])} | {_ci(t['D_pearl'], '.1f')} | "
            f"{_ci(t['action_timesteps'], '.1f')} | {_ci(t['cost'], '.2f')} | {_ci(t['ood_delta'], '.3f')} | "
            f"{_ci(t['failure_rate'])} |"
        )
    L += [
        "",
        "### Reading sensitivity (trajectory methods, full changed schedule)",
        "",
        "| method | world success (ITT) | A | C | Δtotal | action timesteps | E_replay (median) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in sorted(full):
        t = full[m]
        L.append(
            f"| {m}{' ‡' if t['is_control'] else ''} | {_ci(t['world_success_itt'])} | {_ci(t['A'])} | "
            f"{_ci(t['C'])} | {_ci(t['delta_total'])} | {_ci(t['action_timesteps'], '.1f')} | "
            f"{_ci(t['E_replay_median'], '.2g')} |"
        )
    L += [
        "",
        "## Paired method comparisons",
        "",
        "World success (ITT) of A − B on the same flip candidates; d_z = paired standardized effect; "
        "P(sup) = probability that A beats B on a random unit (ties ½).",
        "",
        "| A | B | units | worlds | Δ world success [95% CI] | d_z | P(sup) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in comps:
        L.append(
            f"| {c['a']} | {c['b']} | {c['n_units']} | {c['n_worlds']} | "
            f"{_ci((c['mean_diff'], *c['ci']), '+.3f')} | {c['effect_size_dz']:.2f} | {c['prob_superiority']:.2f} |"
        )
    L += [
        "",
        "## Pareto frontier",
        "",
        f"World success (ITT, ↑) against mean action magnitude (↓), non-control methods. "
        f"Non-dominated: **{', '.join(front) or '–'}**.",
        "",
        "## Draft replication: model–world gap across independent worlds",
        "",
        "Per world, Δtotal = 1{h(x̂)=y*} − 1{ℓ*(x^A)=y*} paired within each scorable unit, pooled over "
        "trajectory baselines and seeds (first-slice reading); CI over classifiers → units.",
        "",
        "| world | family | units | Δtotal [95% CI] | replicated |",
        "|---|---|---|---|---|",
    ]
    for w, v in replication["per_world"].items():
        L.append(
            f"| {w} | {v['family']} | {v['n_units']} | {_ci(v['delta_total'])} | "
            f"{'yes' if v['replicated'] else 'no'} |"
        )
    L += [
        "",
        "## Controls (exact worlds)",
        "",
        "| control | kind | condition | estimate [95% CI] | met |",
        "|---|---|---|---|---|",
    ]
    for c in controls:
        for k, op, thr, est, ok in c["conditions"]:
            L.append(
                f"| {c['control']} | {c['kind']} | {k} {op} {thr:g} | {_ci(est, '.3g')} | "
                f"{'yes' if ok else 'no'} |"
            )
    cats = sorted({k for v in taxonomy.values() for k in v})
    L += [
        "",
        "## Failure-mode taxonomy",
        "",
        "Fraction of method-reading records per attribution category (status rows: fraction with that "
        "explainer status).",
        "",
        "| method | " + " | ".join(cats) + " |",
        "|---|" + "---|" * len(cats),
    ]
    for m in order:
        if m in taxonomy:
            L.append(f"| {m} | " + " | ".join(f"{taxonomy[m].get(k, 0):.2f}" for k in cats) + " |")
    L += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs if p]
    return "\n".join(L) + "\n"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _forest(comps, path: Path) -> Path | None:
    if not comps:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(comps) + 1.2))
    for i, c in enumerate(comps):
        lo, hi = c["ci"]
        ax.plot([lo, hi], [i, i], color="k", lw=1)
        ax.plot(c["mean_diff"], i, "o", color="C0" if lo > 0 or hi < 0 else "grey", ms=4)
    ax.axvline(0, color="r", lw=0.8, ls="--")
    ax.set_yticks(range(len(comps)), [f"{c['a']} − {c['b']}" for c in comps], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Δ world success (ITT), paired, 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _pareto(table, pts, front, path: Path) -> Path | None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m, t in table.items():
        x, y = t["cost"][0], t["world_success_itt"][0]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ctrl = t["is_control"]
        ax.errorbar(
            x,
            y,
            yerr=[[y - t["world_success_itt"][1]], [t["world_success_itt"][2] - y]],
            fmt="s" if ctrl else "o",
            color="grey" if ctrl else ("C3" if m in front else "C0"),
            ms=5,
            capsize=2,
            alpha=0.6 if ctrl else 1,
        )
        ax.annotate(
            m + (" ‡" if ctrl else ""),
            (x, y),
            fontsize=6,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fr = sorted((pts[m] for m in front), key=lambda p: p[0])
    if fr:
        ax.plot(*zip(*fr, strict=True), "C3--", lw=0.8)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("mean action magnitude (scorable)")
    ax.set_ylabel("world success (ITT)")
    ax.set_title("Pareto frontier (red: non-dominated methods)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _per_world(pw, table, infos, path: Path) -> Path | None:
    if not pw:
        return None
    plt = _plt()
    fam = {i["world_id"]: i["family"] for i in infos}
    fams = sorted(set(fam.values()))
    order = sorted(pw, key=lambda m: (table[m]["is_control"], table[m]["native"], m))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(order)), 3.6))
    for i, m in enumerate(order):
        vals = list(pw[m].items())
        xs = i + np.linspace(-0.25, 0.25, len(vals)) if len(vals) > 1 else [i]
        for x, (w, v) in zip(xs, vals, strict=True):
            ax.plot(x, v, "o", ms=3, color=f"C{fams.index(fam[w]) % 10}")
        ax.boxplot([v for _, v in vals], positions=[i], widths=0.6, showfliers=False)
    ax.set_xticks(
        range(len(order)),
        [m + (" ‡" if table[m]["is_control"] else "") for m in order],
        rotation=40,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("world success (ITT), per world")
    for j, f in enumerate(fams):
        ax.plot([], [], "o", color=f"C{j % 10}", label=f)
    ax.legend(fontsize=6, frameon=False, ncol=len(fams))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _replication_plot(rep, path: Path) -> Path | None:
    pw = rep["per_world"]
    if not pw:
        return None
    plt = _plt()
    ws = list(pw)
    fig, ax = plt.subplots(figsize=(6, 0.25 * len(ws) + 1.2))
    for i, w in enumerate(ws):
        p, lo, hi = pw[w]["delta_total"]
        ax.plot([lo, hi], [i, i], color="k", lw=1)
        ax.plot(p, i, "o", ms=4, color="C0" if pw[w]["replicated"] else "grey")
    ax.axvline(0, color="r", lw=0.8, ls="--")
    ax.set_yticks(range(len(ws)), ws, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Δtotal = A − C (paired within units), 95% CI")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_study(cfg: StudyConfig) -> Path:
    return build_study_report(cfg, run_multiworld(cfg))


def load_study_config(path: str | Path) -> StudyConfig:
    from twist.config import load_yaml

    return StudyConfig.model_validate(load_yaml(path))
