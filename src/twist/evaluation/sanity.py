"""Synthetic metric sanity report (Week 3 deliverable).

Runs the positive/negative control suite on several world families and evaluates pre-declared
checks taken from the plan's Week-3 acceptance criteria. Thresholds are fixed here, before any
run, and are not tuned to results; a failing check is reported, not hidden.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twist.config import load_experiment, load_world_config, load_yaml
from twist.evaluation.audit import audit_run
from twist.evaluation.paired_runner import run_paired_experiment
from twist.utils.serialization import read_json, utc_timestamp, write_json

Row = dict[str, Any]
Lookup = Callable[[str, str], Row]

OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "==": operator.eq,
}


@dataclass(frozen=True)
class Check:
    id: str
    criterion: str
    subject: str  # human-readable "explainer / reading"
    op: str
    threshold: float
    observe: Callable[[Lookup, dict[str, Any]], float]


def _metric(explainer: str, reading: str, key: str) -> Callable[[Lookup, dict[str, Any]], float]:
    return lambda get, ctx: float(get(explainer, reading)[key])


def _frac_D1(get: Lookup, ctx: dict[str, Any]) -> float:
    rs = [
        r
        for r in ctx["records"]
        if r["explainer"] == "PearlSCMRecourse"
        and r["reading"] == "native"
        and r["cf_faith_scorable"]
    ]
    return float(np.mean([r["D_pearl"] == 1 for r in rs])) if rs else float("nan")


def _oracle_schedule_fraction(get: Lookup, ctx: dict[str, Any]) -> float:
    return float(get("FullScheduleOracle", "native")["mean_action_timesteps"]) / ctx["T"]


def _soft_faith_gap(get: Lookup, ctx: dict[str, Any]) -> float:
    pos = get("PearlSCMRecourse", "native")["soft_faith_replay"]
    neg = get("AdversarialTrajectoryEdit", "first_change_atomic")["soft_faith_replay"]
    return float(pos - neg)


def _audit(get: Lookup, ctx: dict[str, Any]) -> float:
    return float(ctx["audit"]["passed"])


FAITH_TOL = 1e-3
CHECKS: tuple[Check, ...] = (
    Check(
        "pearl_faith",
        "Pearl control gets near-perfect Pearl CF-faith",
        "PearlSCMRecourse / native",
        ">=",
        0.95,
        _metric("PearlSCMRecourse", "native", "cf_faith_pearl"),
    ),
    Check(
        "pearl_D1",
        "Pearl control has D = 1 for scorable cases",
        "PearlSCMRecourse / native",
        ">=",
        1.0,
        _frac_D1,
    ),
    Check(
        "noiseless_faith",
        "Noiseless control gets near-perfect deterministic-rollout CF-faith",
        "NoiselessSCMRecourse / native",
        ">=",
        0.95,
        _metric("NoiselessSCMRecourse", "native", "cf_faith_noiseless"),
    ),
    Check(
        "oracle_replay",
        "Full-schedule oracle has near-zero replay error",
        "FullScheduleOracle / native",
        "<=",
        1e-9,
        _metric("FullScheduleOracle", "native", "replay_error_pearl"),
    ),
    Check(
        "oracle_complexity",
        "…but high schedule complexity (imposed fraction of T)",
        "FullScheduleOracle / native",
        ">=",
        0.9,
        _oracle_schedule_fraction,
    ),
    Check(
        "adv_validity",
        "Adversarial edit achieves high model validity",
        "AdversarialTrajectoryEdit / first_change_atomic",
        ">=",
        0.8,
        _metric("AdversarialTrajectoryEdit", "first_change_atomic", "validity"),
    ),
    Check(
        "adv_pearl_faith",
        "…but low mechanism consistency (Pearl CF-faith)",
        "AdversarialTrajectoryEdit / first_change_atomic",
        "<=",
        0.1,
        _metric("AdversarialTrajectoryEdit", "first_change_atomic", "cf_faith_pearl"),
    ),
    Check(
        "adv_noiseless_faith",
        "…and low noiseless CF-faith",
        "AdversarialTrajectoryEdit / first_change_atomic",
        "<=",
        0.1,
        _metric("AdversarialTrajectoryEdit", "first_change_atomic", "cf_faith_noiseless"),
    ),
    Check(
        "soft_faith_separates",
        "Graded faith ranks the positive control above the adversarial edit",
        "Pearl native − Adversarial first-slice",
        ">",
        0.0,
        _soft_faith_gap,
    ),
    Check(
        "wrong_replay",
        "Mechanism-wrong edit does not replay (median E_replay ≥ 10× faith tol)",
        "PlausibleButMechanismWrongEdit / native",
        ">=",
        10 * FAITH_TOL,
        _metric("PlausibleButMechanismWrongEdit", "native", "replay_error_pearl"),
    ),
    Check(
        "wrong_pearl_faith",
        "…and fails Pearl CF-faith",
        "PlausibleButMechanismWrongEdit / native",
        "<=",
        0.1,
        _metric("PlausibleButMechanismWrongEdit", "native", "cf_faith_pearl"),
    ),
    Check(
        "wrong_plausible",
        "…while staying plausible (OOD score drop vs factual ≤ 0.05)",
        "PlausibleButMechanismWrongEdit / native",
        ">=",
        -0.05,
        _metric("PlausibleButMechanismWrongEdit", "native", "ood_delta"),
    ),
    Check(
        "abstain_denominator",
        "Abstentions shrink the denominator instead of scoring",
        "NoActionAbstain / native",
        "==",
        0.0,
        _metric("NoActionAbstain", "native", "n_scorable"),
    ),
    Check(
        "audit",
        "Traceability audit reproduces every certificate",
        "all certificates",
        "==",
        1.0,
        _audit,
    ),
)


def evaluate_checks(
    summary: list[Row], records: list[Row], T: int, audit: dict[str, Any]
) -> list[dict[str, Any]]:
    index = {(r["explainer"], r["reading"]): r for r in summary}

    def get(explainer: str, reading: str) -> Row:
        return index[(explainer, reading)]

    ctx = {"records": records, "T": T, "audit": audit}
    out = []
    for c in CHECKS:
        try:
            value = c.observe(get, ctx)
            passed = bool(np.isfinite(value) and OPS[c.op](value, c.threshold))
            note = ""
        except KeyError as exc:
            value, passed, note = float("nan"), False, f"missing row {exc}"
        out.append(
            {
                "id": c.id,
                "criterion": c.criterion,
                "subject": c.subject,
                "op": c.op,
                "threshold": c.threshold,
                "observed": value,
                "passed": passed,
                "note": note,
            }
        )
    return out


def run_sanity(config: str | Path, output_dir: str | None = None) -> Path:
    scfg = load_yaml(config)
    base = load_experiment(scfg["base"])
    if output_dir:
        base.output_dir = output_dir
    per_world = []
    for wpath in scfg["worlds"]:
        cfg = base.model_copy(deep=True)
        cfg.world = load_world_config(wpath)
        cfg.name = f"{scfg['name']}__{cfg.world.world_id}"
        run_dir = run_paired_experiment(cfg)
        audit = audit_run(run_dir)
        summary = read_json(run_dir / "summary.json")
        records = read_json(run_dir / "records.json")
        T = int(cfg.world.params.get("T", 100))
        per_world.append(
            {
                "world_id": cfg.world.world_id,
                "family": cfg.world.family,
                "run_dir": str(run_dir),
                "checks": evaluate_checks(summary, records, T, audit),
                "summary": summary,
            }
        )
    report = {
        "name": scfg["name"],
        "timestamp_utc": utc_timestamp(),
        "worlds": per_world,
        "passed": all(c["passed"] for w in per_world for c in w["checks"]),
    }
    out = Path(base.output_dir) / "reports"
    stem = f"{scfg['name']}__{report['timestamp_utc'].replace(':', '').replace('-', '')}"
    write_json(out / f"{stem}.json", report)
    path = out / f"{stem}.md"
    path.write_text(sanity_markdown(report), encoding="utf-8")
    print(sanity_markdown(report))
    return path


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{v:.3g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"


SUMMARY_COLS = [
    ("validity", "V"),
    ("cf_faith_noiseless", "R-faith"),
    ("cf_faith_pearl", "P-faith"),
    ("soft_faith_replay", "S-faith"),
    ("D_pearl", "D"),
    ("mean_action_timesteps", "|𝒜|"),
    ("replay_error_pearl", "E_rep"),
    ("causal_path_consistency", "path"),
    ("ood_delta", "ΔOOD"),
    ("proposal_model_success", "A"),
    ("replay_model_success", "B"),
    ("replay_world_success", "C"),
    ("n_scorable", "n_s"),
]


def sanity_markdown(report: dict[str, Any]) -> str:
    worlds = report["worlds"]
    lines = [
        f"# Metric sanity report — {report['name']}",
        "",
        f"{report['timestamp_utc']} · overall **{'PASS' if report['passed'] else 'FAIL'}** · "
        "thresholds pre-declared in `twist.evaluation.sanity.CHECKS`",
        "",
        "| check | criterion | subject | expected | "
        + " | ".join(w["family"] for w in worlds)
        + " |",
        "|---|---|---|---|" + "---|" * len(worlds),
    ]
    for i, c in enumerate(worlds[0]["checks"]):
        cells = []
        for w in worlds:
            r = w["checks"][i]
            cells.append(f"{_fmt(r['observed'])} {'✓' if r['passed'] else '✗ ' + r['note']}")
        lines.append(
            f"| `{c['id']}` | {c['criterion']} | {c['subject']} | {c['op']} {_fmt(c['threshold'])} | "
            + " | ".join(cells)
            + " |"
        )
    for w in worlds:
        lines += [
            "",
            f"## {w['family']} — `{w['world_id']}`",
            "",
            f"run: `{w['run_dir']}`",
            "",
            "| method | reading | " + " | ".join(h for _, h in SUMMARY_COLS) + " |",
            "|---|---|" + "---|" * len(SUMMARY_COLS),
        ]
        for r in w["summary"]:
            vals = [_fmt(float(r[k])) if r[k] is not None else "–" for k, _ in SUMMARY_COLS]
            lines.append(f"| {r['explainer']} | {r['reading']} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"
