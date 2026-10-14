"""Energy domain bridge study (plan §Week 9, tasks 1–4; protocol v1.2 §10).

Semi-synthetic energy worlds (``twist.worlds.semi_synthetic``) with explicit action variables are
run through the multi-world runner. Rows are evaluated twice from the same records: under the
*generic* constraints only, and under generic + *domain* constraints (battery capacity, battery
C-rate, demand-response energy cap; ids listed in ``domain_constraints``). The difference shows
which explanations a domain constraint makes unacceptable.

An explanation is **acceptable** when its action is scorable, reaches the world target under
oracle replay, and passes the constraint set (ITT: every factual unit counts, no action = not
acceptable).

Pre-declared acceptance checks:
1. ``native_control_actions_scored`` — every control row (rows planning a world control) has
   scorable native control actions on every world: the bridge has explicit action variables, not
   only editable observations;
2. ``control_actions_replayable`` — every scorable control action, rebuilt from its serialized
   form, replays on the reloaded world to the stored replay hash (documented, replayable
   action-to-state mechanism);
3. ``domain_constraint_changes_acceptability`` — among the constraint-blind control row's
   explanations that are acceptable under the generic constraints, the share a domain constraint
   rejects has a world-bootstrap 95% CI with lower bound > 0.

Pre-declared hypotheses (reported with CIs, not acceptance criteria):
* HE1 — constraint-aware planning (R6) raises domain acceptability over the same optimizer without
  constraint knowledge (R5), for battery dispatch: paired ITT difference > 0.
* HE2 — an editable observation overstates achievable recourse: the abstract loading set-point
  ``do(grid)`` (R5) is acceptable more often than constraint-aware battery dispatch (R6), paired
  ITT difference > 0 (the realizability gap of observation-level actions).
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
)
from twist.evaluation.study import load_study_records
from twist.utils.logging import get_logger
from twist.utils.seeds import derive_seed
from twist.utils.serialization import hash_arrays, read_json, utc_timestamp, write_json

log = get_logger(__name__)

CLAIM_BOUNDARY = (
    "The energy worlds are semi-synthetic. Weather and demand covariates are real (OPSD, "
    "Netherlands 2016–2019), but the transformer, battery, policy and demand-response dynamics are "
    "an explicit simulator inspired by distribution-grid physics (a simplified IEC 60076-7 thermal "
    "model). Causal truth holds by construction for the simulator only. No result here is evidence "
    "about a real feeder, a real battery or real customers; the domain constraints are "
    "illustrative operating limits, not a grid code; and the overload label is a class-balanced "
    "threshold on the simulated hot spot, not a protection setting. The bridge tests whether "
    "TWIST's evaluation logic carries over to explicit action variables and domain feasibility."
)


class BridgeConfig(MultiWorldConfig):
    n_boot: int = 1000
    domain_constraints: list[str] = Field(
        default_factory=lambda: ["soc_capacity", "battery_c_rate", "dr_energy_cap"]
    )
    constraint_blind: str = "Dispatch[R5 constraint-blind]"
    constraint_aware: str = "Dispatch[R6 constraint-aware]"
    abstract_setpoint: str = "SetPoint[R5 grid]"
    demo_worlds: int = 2  # demonstration examples: one unit per world for the first n worlds


# ------------------------------------------------------------------------------------ helpers
def _is_control(r: dict) -> bool:
    return str(r.get("action_semantics") or "").startswith("native_control")


def _accept(r: dict, domain: set[str], with_domain: bool) -> float:
    if not r["scorable"] or r["world_replay"] != r["target"]:
        return 0.0
    failed = set(r.get("constraints_failed") or [])
    if not with_domain:
        failed -= domain
    return float(not failed)


def _world_share(
    rs: list[tuple[str, float]], n_boot: int, seed: int
) -> tuple[float, float, float, int]:
    """Pooled share of ``flag`` among records, world-bootstrap CI; ``rs`` = (world_id, flag)."""
    worlds = sorted({w for w, _ in rs})
    if not rs:
        return float("nan"), float("nan"), float("nan"), 0
    by = {w: np.array([f for ww, f in rs if ww == w], float) for w in worlds}
    point = float(np.mean([f for _, f in rs]))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        drawn = rng.choice(worlds, len(worlds))
        v = np.concatenate([by[w] for w in drawn])
        b.append(v.mean())
    lo, hi = np.quantile(b, [0.025, 0.975])
    return point, float(lo), float(hi), len(rs)


def _cols(rs: list[dict], domain: set[str]) -> dict[str, tuple[np.ndarray, str]]:
    sc = np.array([bool(r["scorable"]) for r in rs])
    nan = np.where(sc, 0.0, np.nan)
    generic_ok = [not (set(r.get("constraints_failed") or []) - domain) for r in rs]
    all_ok = [not r.get("constraints_failed") for r in rs]
    return {
        "world_success_itt": (
            np.array([float(bool(r["scorable"]) and r["world_replay"] == r["target"]) for r in rs]),
            "mean",
        ),
        "C": (np.array([float(r["world_replay"] == r["target"]) for r in rs]) + nan, "mean"),
        "feasible_generic": (np.array(generic_ok, float) + nan, "mean"),
        "feasible_domain": (np.array(all_ok, float) + nan, "mean"),
        "accept_generic": (np.array([_accept(r, domain, False) for r in rs]), "mean"),
        "accept_domain": (np.array([_accept(r, domain, True) for r in rs]), "mean"),
        "invalid_itt": (
            np.array([float(r.get("attribution") == "invalid_action") for r in rs]),
            "mean",
        ),
        "cost": (np.array([float(r.get("cost") or 0.0) for r in rs]) + nan, "mean"),
    }


# ------------------------------------------------------------------------------------- report
def build_bridge_report(cfg: BridgeConfig, run_dirs: list[Path]) -> Path:
    from twist.reporting.benchmark_cards import domain_action_card

    records, infos = load_study_records(run_dirs)
    records = [r for r in records if r["method_reading"]]
    domain = set(cfg.domain_constraints)
    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_m[r["explainer"]].append(r)

    rows = []
    for m, rs in sorted(by_m.items()):
        draws = bootstrap_draws(rs, LEVELS, cfg.n_boot, seed=derive_seed(0, "bridge", m))
        est = column_estimates(_cols(rs, domain), draws)
        acc_generic = [r for r in rs if _accept(r, domain, False)]
        flips = Counter(c for r in acc_generic for c in r["constraints_failed"] if c in domain)
        violations = Counter(c for r in rs if r["scorable"] for c in r["constraints_failed"])
        rows.append(
            {
                "method": m,
                "regime": rs[0]["mechanism_access_regime"],
                "action": "control"
                if any(_is_control(r) for r in rs)
                else ("native" if rs[0]["native_action"] else "reading"),
                "n": len(rs),
                "n_scorable": sum(bool(r["scorable"]) for r in rs),
                "n_worlds": len({r["world_id"] for r in rs}),
                **est,
                "rejected_by_domain": _world_share(
                    [
                        (r["world_id"], float(bool(set(r["constraints_failed"]) & domain)))
                        for r in acc_generic
                    ],
                    cfg.n_boot,
                    derive_seed(0, "flip", m),
                ),
                "flips_by_constraint": dict(flips),
                "violations": dict(violations),
            }
        )
    order = {"R0": 0, "R1": 1, "R5": 2, "R6": 3}
    rows.sort(key=lambda r: (r["action"] != "reading", order.get(r["regime"], 9), r["method"]))

    # ---- checks
    worlds = sorted({i["world_id"] for i in infos})
    control_rows = [r for r in rows if r["action"] == "control"]
    missing = [
        f"{r['method']}@{w}"
        for r in control_rows
        for w in worlds
        if not any(x["scorable"] and x["world_id"] == w for x in by_m[r["method"]])
    ]
    replay = _check_replay(run_dirs, by_m, control_rows)
    blind = next((r for r in rows if r["method"] == cfg.constraint_blind), None)
    flip = blind["rejected_by_domain"] if blind else (np.nan, np.nan, np.nan, 0)
    checks = [
        {
            "check": "native_control_actions_scored",
            "passed": bool(control_rows) and not missing,
            "detail": f"{len(control_rows)} control rows × {len(worlds)} worlds; "
            f"missing: {missing or 'none'}",
        },
        {
            "check": "control_actions_replayable",
            "passed": replay["n"] > 0 and replay["mismatch"] == 0,
            "detail": f"{replay['n']} scorable control actions re-replayed from their serialized "
            f"form; {replay['mismatch']} hash mismatches",
        },
        {
            "check": "domain_constraint_changes_acceptability",
            "passed": bool(np.isfinite(flip[1]) and flip[1] > 0),
            "detail": f"`{cfg.constraint_blind}`: {flip[3]} explanations acceptable under generic "
            f"constraints; share rejected by a domain constraint {_ci(flip[:3])}",
        },
    ]

    def paired(a: str, b: str, key: str) -> dict[str, Any] | None:
        if not by_m.get(a) or not by_m.get(b):
            return None
        return paired_difference(
            by_m[a],
            by_m[b],
            lambda r: _accept(r, domain, True),
            cfg.n_boot,
            seed=derive_seed(0, key),
        )

    hyps = []
    for key, a, b, text in (
        (
            "HE1",
            cfg.constraint_aware,
            cfg.constraint_blind,
            "constraint-aware planning raises domain acceptability",
        ),
        (
            "HE2",
            cfg.abstract_setpoint,
            cfg.constraint_aware,
            "an editable observation overstates achievable recourse",
        ),
    ):
        d = paired(a, b, key)
        hyps.append(
            {
                "hypothesis": f"{key}: {text} ({a} − {b} > 0)",
                "estimate": (d["mean_diff"], *d["ci"]) if d else (np.nan, np.nan, np.nan),
                "n_units": d["n_units"] if d else 0,
                "supported": bool(d and d["ci"][0] > 0),
            }
        )

    out = Path(cfg.output_dir) / cfg.name / "report"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    cards: list[tuple[str, str]] = []
    demos: list[dict[str, Any]] = []
    figs: list[Path] = []
    for run_dir in run_dirs:
        world = _load_world(run_dir)
        if world is not None and world.world_id not in {c[0] for c in cards}:
            cards.append((world.world_id, domain_action_card(world)))
    for w in worlds[: cfg.demo_worlds]:
        demo = _demonstration(cfg, run_dirs, by_m, w, domain, out / f"{stem}__demo_{w}.png")
        if demo:
            demos.append(demo)
            figs.append(demo["figure"])
    figs.append(_acceptance_plot(rows, out / f"{stem}__acceptance.png"))
    path = out / f"{stem}.md"
    path.write_text(_markdown(cfg, infos, rows, checks, hyps, cards, demos, figs), encoding="utf-8")
    write_json(
        out / f"{stem}.json",
        {
            "checks": checks,
            "hypotheses": hyps,
            "rows": rows,
            "demonstrations": [{k: v for k, v in d.items() if k != "figure"} for d in demos],
            "claim_boundary": CLAIM_BOUNDARY,
            "runs": [str(d) for d in run_dirs],
        },
    )
    return path


# ----------------------------------------------------------------------------- replay checks
def _run_world(run_dir: Path) -> tuple[Any, dict[str, Any]] | None:
    from twist.worlds.base_scm import TemporalSCM
    from twist.worlds.io import load_world_data

    cfg = read_json(Path(run_dir) / "config.resolved.json")
    wdir = Path(cfg["output_dir"]) / "worlds" / cfg["world"]["world_id"]
    if not (wdir / "world_card.json").exists():
        return None
    return TemporalSCM.load(wdir), load_world_data(wdir)  # the card carries the fitted label


def _load_world(run_dir: Path) -> Any:
    got = _run_world(run_dir)
    return got[0] if got else None


def _unit_exo(data: dict[str, Any], factual_id: str) -> dict[str, np.ndarray]:
    i = data["n_train"] + int(factual_id.rsplit("/", 1)[1])
    return {k: v[i : i + 1] for k, v in data["exo"].items()}


def _check_replay(run_dirs, by_m, control_rows) -> dict[str, int]:
    from twist.actions.schemas import ActionPlan

    n = mismatch = 0
    methods = {r["method"] for r in control_rows}
    for d in run_dirs:
        got = _run_world(d)
        if got is None:
            continue
        world, data = got
        for m in methods:
            for r in by_m[m]:
                if r["world_id"] != world.world_id or not r["scorable"] or not r.get("action"):
                    continue
                plan = ActionPlan.from_dict(r["action"])
                x_rep, _ = world.replay_units(_unit_exo(data, r["factual_id"]), [plan])
                n += 1
                mismatch += int(hash_arrays(x_rep[0]) != r["replay_hash"])
    return {"n": n, "mismatch": mismatch}


# ---------------------------------------------------------------------------- demonstrations
def _demonstration(cfg, run_dirs, by_m, world_id, domain, path: Path) -> dict[str, Any] | None:
    """One unit where the constraint-blind dispatch is acceptable under generic constraints but
    rejected by a domain constraint, shown next to the constraint-aware dispatch on that unit."""
    from twist.actions.schemas import ActionPlan

    blind = [
        r
        for r in by_m.get(cfg.constraint_blind, [])
        if r["world_id"] == world_id
        and _accept(r, domain, False)
        and set(r["constraints_failed"]) & domain
    ]
    if not blind:
        return None
    rb = blind[0]
    ra = next(
        (
            r
            for r in by_m.get(cfg.constraint_aware, [])
            if r["world_id"] == world_id and r["factual_id"] == rb["factual_id"]
        ),
        None,
    )
    run_dir = next(d for d in run_dirs if rb["run_id"] in str(d))
    world, data = _run_world(run_dir)  # type: ignore[misc]
    exo = _unit_exo(data, rb["factual_id"])
    trajs = {"factual": exo["state"][0]}
    summary = {"world_id": world_id, "unit": rb["factual_id"], "actions": {}}
    for label, r in (("constraint-blind (R5)", rb), ("constraint-aware (R6)", ra)):
        if r is None or not r.get("action"):
            continue
        plan = ActionPlan.from_dict(r["action"])
        _, s = world.replay_units(exo, [plan])
        trajs[label] = s[0]
        a = r["action"]["action"]
        summary["actions"][label] = {
            "control": a.get("control"),
            "window": [a.get("start"), a.get("end")],
            "values_MW": [round(10.0 * v, 2) for v in a.get("values", [])],
            "world_success": bool(r["world_replay"] == r["target"]),
            "failed_constraints": r["constraints_failed"],
            "acceptable": bool(_accept(r, domain, True)),
        }
    summary["figure"] = _demo_plot(world, trajs, path)
    return summary


def _demo_plot(world, trajs: dict[str, np.ndarray], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from twist.worlds.semi_synthetic import GRID, HOT_SPOT, HS_REF, HS_SCALE, SOC

    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    panels = (
        (GRID, "transformer loading (p.u.)", 1.0),
        (SOC, "battery state of charge", 1.0),
        (HOT_SPOT, "hot spot (°C)", None),
    )
    for ax, (ch, lab, _) in zip(axes, panels, strict=True):
        for j, (name, z) in enumerate(trajs.items()):
            y = z[:, ch] * HS_SCALE + HS_REF if ch == HOT_SPOT else z[:, ch]
            ax.plot(y, lw=1.4 if j == 0 else 1.1, ls="-" if j == 0 else "--", label=name)
        ax.set_ylabel(lab, fontsize=8)
    axes[1].axhspan(-0.5, 0.0, color="red", alpha=0.08)
    axes[1].axhline(0.0, color="red", lw=0.8)
    thr = getattr(world.label, "threshold", None)
    if thr is not None:
        axes[2].axhline(thr * HS_SCALE + HS_REF, color="red", lw=0.8, label="label threshold")
    axes[0].legend(fontsize=7, frameon=False)
    axes[2].legend(fontsize=7, frameon=False, loc="upper left")
    axes[2].set_xlabel("hour (window starts 00:00 UTC)")
    fig.suptitle(world.world_id, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _acceptance_plot(rows: list[dict], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1))
    y = np.arange(len(rows))
    for off, key, lab in ((-0.18, "accept_generic", "generic"), (0.18, "accept_domain", "domain")):
        p = np.array([r[key][0] for r in rows])
        lo = np.array([r[key][1] for r in rows])
        hi = np.array([r[key][2] for r in rows])
        ax.errorbar(p, y + off, xerr=[p - lo, hi - p], fmt="o", ms=4, capsize=2, label=lab)
    ax.set_yticks(y, [r["method"] for r in rows], fontsize=7)
    ax.set_xlabel("acceptable explanations (ITT): world target reached ∧ constraints pass")
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------ markdown
def _ci(t: Any, fmt: str = ".2f") -> str:
    if not isinstance(t, tuple | list) or t[0] is None or not np.isfinite(t[0]):
        return "–"
    p, lo, hi = t[:3]
    return f"{p:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]" if np.isfinite(lo) else f"{p:{fmt}}"


def _markdown(cfg, infos, rows, checks, hyps, cards, demos, figs) -> str:
    worlds = sorted({i["world_id"] for i in infos})
    L = [
        f"# Energy domain bridge — `{cfg.name}`",
        "",
        f"{utc_timestamp()} · worlds {worlds} · classifiers "
        f"{sorted({i['architecture'] for i in infos})} · {cfg.n_factuals} units per world. "
        "Outcomes: four-level hierarchical bootstrap (worlds → classifiers → units → seeds); "
        "shares of flipped explanations: world bootstrap. Domain constraints: "
        f"{sorted(cfg.domain_constraints)}.",
        "",
        "## Claim boundary",
        "",
        CLAIM_BOUNDARY,
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
        "| hypothesis | estimate [95% CI] | units | supported |",
        "|---|---|---|---|",
    ]
    L += [
        f"| {h['hypothesis']} | {_ci(tuple(h['estimate']), '+.3f')} | {h['n_units']} | "
        f"{'yes' if h['supported'] else 'no'} |"
        for h in hyps
    ]
    L += [
        "",
        "## Acceptability under generic vs domain constraints",
        "",
        "`action`: **control** = native world control (explicit action variable); **native** = "
        "native observed-variable action (abstract loading set-point); **reading** = trajectory "
        "explanation read as an action by the first-slice adapter (rejected as invalid when it "
        "edits non-actionable channels such as weather or demand forecasts). Acceptable = "
        "scorable action, world target (no overload) reached under oracle replay, constraints "
        "pass (ITT: every factual unit counts).",
        "",
        "| action | regime | method | invalid action | world success | feasible (generic) | "
        "feasible (domain) | acceptable (generic) | acceptable (domain) | rejected by domain | "
        "cost (Σ|a|) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['action']} | {r['regime']} | {r['method']} | {_ci(r['invalid_itt'])} | "
            f"{_ci(r['world_success_itt'])} | "
            f"{_ci(r['feasible_generic'])} | {_ci(r['feasible_domain'])} | "
            f"{_ci(r['accept_generic'])} | {_ci(r['accept_domain'])} | "
            f"{_ci(r['rejected_by_domain'][:3])} | {_ci(r['cost'])} |"
        )
    L += [
        "",
        "Domain constraints that turned a generic-acceptable explanation unacceptable, and all "
        "constraint violations among scorable actions:",
        "",
        "| method | flips by constraint | violations (scorable) |",
        "|---|---|---|",
    ]
    L += [
        f"| {r['method']} | {r['flips_by_constraint'] or '–'} | {r['violations'] or '–'} |"
        for r in rows
    ]
    L += ["", "## Demonstration examples", ""]
    for d in demos:
        L.append(f"### {d['world_id']} — unit `{d['unit']}`")
        L.append("")
        L.append(
            "| planner | control | window [start, end) | values (MW) | world target | failed constraints | acceptable |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for lab, a in d["actions"].items():
            L.append(
                f"| {lab} | `{a['control']}` | {a['window']} | {a['values_MW']} | "
                f"{'reached' if a['world_success'] else 'missed'} | "
                f"{a['failed_constraints'] or 'none'} | {'yes' if a['acceptable'] else 'no'} |"
            )
        L += ["", f"![demo]({Path(d['figure']).name})", ""]
    if not demos:
        L.append("No unit where a domain constraint rejected a generic-acceptable dispatch.")
    L += ["", "## Domain action cards", ""]
    for _, card in cards:
        L += [card, ""]
    L += ["## Figures", ""] + [f"![{Path(p).stem}]({Path(p).name})" for p in figs if p]
    return "\n".join(L) + "\n"


def run_bridge(cfg: BridgeConfig) -> Path:
    return build_bridge_report(cfg, run_multiworld(cfg))


def load_bridge_config(path: str | Path) -> BridgeConfig:
    from twist.config import load_yaml

    return BridgeConfig.model_validate(load_yaml(path))
