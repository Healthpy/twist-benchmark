"""Mechanism-access ladder report (plan §Week 6 deliverable: first access-ladder plots).

Rows are explainers at their regime. Trajectory-only baselines are read with the first-slice
reading; action-native explainers with their declared action. Per row: the mechanism used, its
quality against the true world (benchmark-side), A/B/C, replay error, feasibility, and where the
failures arise (``attribution``). Misspecified-mechanism rows (``misspecified:<kind>-<level>``)
also feed the degradation sweep. Pre-declared acceptance checks (thresholds fixed before running):

* the oracle action control reaches C ≥ 0.8 with feasible rate ≥ 0.95;
* every misspecification kind degrades: median replay error at the largest level exceeds level 0
  by ≥ 1e-3;
* every unit of every method is attributed (success / outcome_disagreement / replay_mismatch /
  optimization / no_action / invalid_action / no_proposal).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from twist.evaluation.pilot_report import load_runs
from twist.evaluation.protocol import ATTRIBUTIONS, summarize
from twist.explainers.mechanism_access.regimes import REGIMES
from twist.utils.serialization import utc_timestamp, write_json

FIRST_SLICE = "first_change_atomic"
ORACLE_C_MIN, ORACLE_FEAS_MIN, DEGRADATION_MIN = 0.8, 0.95, 1e-3
MISSPEC = re.compile(r"misspecified:([a-z_]+)-([0-9.]+)")


def ladder_rows(records: list[dict[str, Any]], infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality: dict[str, list] = defaultdict(list)
    for i in infos:
        for label, q in (i.get("mechanisms") or {}).items():
            quality[label].append(q)
    rows = []
    for row in summarize(records):
        if row["reading"] not in (FIRST_SLICE, "native"):
            continue
        rs = [
            r
            for r in records
            if r["explainer"] == row["explainer"] and r["reading"] == row["reading"]
        ]
        regime = rs[0]["mechanism_access_regime"]
        mech = (rs[0].get("mechanism") or "").split(":")
        # rows pool worlds: oracle labels carry a world id, estimates carry their estimator
        label = "oracle" if mech[0] == "oracle" else (":".join(mech[:2]) if len(mech) >= 2 else "")
        qs = quality.get(label, [])
        n = len(rs)
        att = Counter(r["attribution"] for r in rs)
        rows.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "explainer",
                        "reading",
                        "validity",
                        "proposal_model_success",
                        "replay_model_success",
                        "replay_world_success",
                        "replay_error_pearl",
                        "n_scorable",
                        "is_control",
                    )
                },
                "feasible_rate": row.get("feasible_rate", float("nan")),
                "regime": regime,
                "mechanism": label or ("oracle" if regime in ("R5", "R6") else "—"),
                "mech_replay_error": float(np.mean([q["replay_error"] for q in qs]))
                if qs
                else float("nan"),
                "mech_graph_f1": float(np.mean([q["graph_f1"] for q in qs]))
                if qs
                else float("nan"),
                "attribution": {k: att.get(k, 0) / n for k in ATTRIBUTIONS},
                "attributed_all": sum(att.values()) == n and set(att) <= set(ATTRIBUTIONS),
            }
        )
    rows.sort(key=lambda r: (REGIMES.index(r["regime"]), r["explainer"]))
    return rows


def misspec_sweep(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float, float]]]:
    """kind → sorted [(level, median replay error, C)]."""
    out: dict[str, list] = defaultdict(list)
    anchor = None
    for r in rows:
        m = MISSPEC.fullmatch(r["mechanism"])
        if m:
            pt = (float(m.group(2)), r["replay_error_pearl"], r["replay_world_success"])
            if pt[0] == 0:
                anchor = pt  # level 0 is the exact mechanism, shared by every kind
            out[m.group(1)].append(pt)
    if anchor is not None:
        for _kind, pts in out.items():
            if not any(p[0] == 0 for p in pts):
                pts.append(anchor)
    return {k: sorted(v) for k, v in out.items() if any(p[0] > 0 for p in v)}


def ladder_checks(rows: list[dict[str, Any]], sweep: dict[str, list]) -> list[dict[str, Any]]:
    checks = []
    oracle = [r for r in rows if r["explainer"].startswith("OracleActionRecourse")]
    if oracle:
        c = float(np.mean([r["replay_world_success"] for r in oracle]))
        f = float(np.mean([r["feasible_rate"] for r in oracle]))
        checks.append(
            {
                "check": "oracle_action_high_world_success",
                "passed": c >= ORACLE_C_MIN and f >= ORACLE_FEAS_MIN,
                "detail": f"C = {c:.2f} (≥ {ORACLE_C_MIN}), feasible = {f:.2f} (≥ {ORACLE_FEAS_MIN})",
            }
        )
    deg = {}
    for kind, pts in sweep.items():
        base = [e for lv, e, _ in pts if lv == 0]
        top = pts[-1]
        if base and top[0] > 0:
            deg[kind] = (top[1] - base[0], top[2] - next(c for lv, _, c in pts if lv == 0))
    checks.append(
        {
            "check": "misspecification_degrades_estimated_recourse",
            "passed": bool(deg) and all(d >= DEGRADATION_MIN for d, _ in deg.values()),
            "detail": "; ".join(
                f"{k}: ΔE_replay {d:+.3f}, ΔC {dc:+.2f}" for k, (d, dc) in deg.items()
            )
            or "no sweep",
        }
    )
    checks.append(
        {
            "check": "failures_attributed",
            "passed": all(r["attributed_all"] for r in rows),
            "detail": f"{len(rows)} method rows, categories {list(ATTRIBUTIONS)}",
        }
    )
    return checks


def _f(v: Any, fmt: str = ".2f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return format(v, fmt)


def plot_ladder(rows: list[dict[str, Any]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r for r in rows if not MISSPEC.fullmatch(r["mechanism"]) or r["mechanism"].endswith("-0")
    ]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(rows) + 2), 3.6))
    ax.bar(x - 0.2, [r["proposal_model_success"] for r in rows], 0.4, label="A: model on proposal")
    ax.bar(x + 0.2, [r["replay_world_success"] for r in rows], 0.4, label="C: world on replay")
    ax.set_xticks(
        x, [f"{r['regime']}\n{r['explainer']}" for r in rows], rotation=35, ha="right", fontsize=7
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (scorable units)")
    ax.set_title("Mechanism-access ladder", fontsize=9, pad=30)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_sweep(sweep: dict[str, list], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for kind, pts in sorted(sweep.items()):
        lv = [p[0] for p in pts]
        a1.plot(lv, [p[1] for p in pts], marker="o", label=kind)
        a2.plot(lv, [p[2] for p in pts], marker="o", label=kind)
    a1.set_xlabel("misspecification level")
    a1.set_ylabel("median replay error")
    a2.set_xlabel("misspecification level")
    a2.set_ylabel("C: world success")
    a2.set_ylim(0, 1.05)
    a1.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Estimated-mechanism action recourse under controlled misspecification", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_ladder_report(name: str, run_dirs: list[Path], out_dir: str | Path) -> Path:
    records, infos = load_runs(run_dirs)
    rows = ladder_rows(records, infos)
    sweep = misspec_sweep(rows)
    checks = ladder_checks(rows, sweep)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    figs = [plot_ladder(rows, out / f"{stem}__ladder.png")]
    if sweep:
        figs.append(plot_sweep(sweep, out / f"{stem}__misspecification.png"))
    worlds = sorted({i["world_id"] for i in infos})
    L = [
        f"# Mechanism-access ladder — `{name}`",
        "",
        f"{utc_timestamp()} · worlds {worlds} · pooled point estimates (uncertainty: Week 8). "
        "Trajectory baselines: first-slice reading; action-native explainers: declared action.",
        "",
        "## Week-6 acceptance checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        *[
            f"| {c['check']} | {'**PASS**' if c['passed'] else '**FAIL**'} | {c['detail']} |"
            for c in checks
        ],
        "",
        "## Ladder",
        "",
        "| regime | method | mechanism | mech. replay err | Ĝ F1 | V | A | B | C | E_rep | feasible | success | outcome dis. | replay mism. | optimization | no action |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a = r["attribution"]
        L.append(
            f"| {r['regime']} | {r['explainer']}{' ‡' if r['is_control'] else ''} | {r['mechanism']} | "
            f"{_f(r['mech_replay_error'], '.4f')} | {_f(r['mech_graph_f1'])} | {_f(r['validity'])} | "
            f"{_f(r['proposal_model_success'])} | {_f(r['replay_model_success'])} | {_f(r['replay_world_success'])} | "
            f"{_f(r['replay_error_pearl'], '.3g')} | {_f(r['feasible_rate'])} | {_f(a['success'])} | "
            f"{_f(a['outcome_disagreement'])} | {_f(a['replay_mismatch'])} | {_f(a['optimization'])} | "
            f"{_f(a['no_action'] + a['no_proposal'] + a['invalid_action'])} |"
        )
    if sweep:
        L += [
            "",
            "## Misspecification sweep (EstimatedActionRecourse)",
            "",
            "| kind | level | median replay error | C |",
            "|---|---|---|---|",
        ]
        for kind, pts in sorted(sweep.items()):
            L += [f"| {kind} | {lv:g} | {_f(e, '.4f')} | {_f(c)} |" for lv, e, c in pts]
    L += [
        "",
        "Attribution columns are fractions of all flip candidates: *success* (world target "
        "reached), *outcome dis.* (replay keeps the classifier's decision, world label differs), "
        "*replay mism.* (the true world does not reproduce the proposal closely enough to keep "
        "the decision: mechanism error for estimated mechanisms, missing action grounding for "
        "trajectory readings), *optimization* (target not reached under the explainer's own model).",
        "",
        "## Figures",
        "",
    ] + [f"![{p.stem}]({p.name})" for p in figs]
    path = out / f"{stem}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(
        out / f"{stem}.json",
        {
            "name": name,
            "checks": checks,
            "rows": rows,
            "sweep": sweep,
            "runs": [str(d) for d in run_dirs],
        },
    )
    return path
