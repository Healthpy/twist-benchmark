"""Pilot dashboard and Week-5 acceptance checks over a set of multi-world run directories.

Pooled tables are descriptive pilot numbers (point estimates over worlds × classifiers × seeds).
Uncertainty and hierarchical aggregation arrive in Week 8. No method ranking is implied.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from twist.evaluation.protocol import summarize
from twist.evaluation.result_schema import ExplanationStatus
from twist.utils.serialization import read_json, utc_timestamp, write_json

FIRST_SLICE = "first_change_atomic"
OK_STATUSES = {s.value for s in ExplanationStatus}


def load_runs(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, infos = [], []
    for d in run_dirs:
        if not (Path(d) / "run_info.json").exists():  # incomplete or skipped task
            continue
        info = read_json(d / "run_info.json")
        cfg = read_json(d / "config.resolved.json")
        info["world_id"] = cfg["world"]["world_id"]
        info["family"] = cfg["world"]["family"]
        info["architecture"] = cfg["classifier"]["architecture"]
        info["n_factuals"] = cfg["n_factuals"]
        infos.append(info)
        for r in read_json(d / "records.json"):
            r.update(
                world_id=info["world_id"],
                family=info["family"],
                architecture=info["architecture"],
                run_id=info["run_id"],
            )
            records.append(r)
    return records, infos


def _explanations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per explanation (records repeat once per reading)."""
    seen, out = set(), []
    for r in records:
        if r["explanation_id"] not in seen:
            seen.add(r["explanation_id"])
            out.append(r)
    return out


def acceptance_checks(
    records: list[dict[str, Any]], infos: list[dict[str, Any]], summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    checks = []
    # 1. every method yields a standardized output or a structured failure record for every unit
    counts: Counter = Counter()
    for r in _explanations(records):
        counts[(r["run_id"], r["explainer"], r["explainer_seed"])] += 1
    n_expected = {i["run_id"]: i["n_factuals"] for i in infos}
    missing = [k for k, v in counts.items() if v != n_expected[k[0]]]
    bad_status = sorted({r["status"] for r in records} - OK_STATUSES)
    skipped = [
        (i["world_id"], s["explainer"]) for i in infos for s in i.get("skipped_explainers", [])
    ]
    checks.append(
        {
            "check": "standardized_output_or_failure_record",
            "passed": not missing and not bad_status,
            "detail": f"{len(counts)} (run, method, seed) groups; incomplete {len(missing)}; unknown "
            f"statuses {bad_status or 'none'}; recorded skips {len(skipped)}",
        }
    )
    # 2. qualitative model-vs-world gap (draft): A > C under the first-slice reading
    rows = {(r["explainer"], r["reading"]): r for r in summary}
    baselines = sorted(
        {r["explainer"] for r in records if not r.get("is_control") and not r["native_action"]}
    )
    gaps = {b: rows[(b, FIRST_SLICE)]["delta_total"] for b in baselines if (b, FIRST_SLICE) in rows}
    finite = {b: g for b, g in gaps.items() if np.isfinite(g)}
    checks.append(
        {
            "check": "model_vs_world_gap_reproduced",
            "passed": bool(finite) and all(g > 0 for g in finite.values()),
            "detail": "Δtotal (A−C) under first-slice reading: "
            + ", ".join(f"{b} {g:+.2f}" for b, g in finite.items()),
        }
    )

    # 3. at least one method passes each positive-control condition
    def get(name: str, key: str) -> float:
        row = rows.get((name, "native"))
        return float(row[key]) if row and row[key] is not None else float("nan")

    pearl_d1 = [
        r["D_pearl"] == 1
        for r in records
        if r["explainer"] == "PearlSCMRecourse"
        and r["reading"] == "native"
        and r["cf_faith_scorable"]
    ]
    conds = {
        "Pearl CF-faith ≥ 0.95 (PearlSCMRecourse)": get("PearlSCMRecourse", "cf_faith_pearl")
        >= 0.95,
        "D = 1 on scorable units (PearlSCMRecourse)": bool(pearl_d1) and all(pearl_d1),
        "noiseless CF-faith ≥ 0.95 (NoiselessSCMRecourse)": get(
            "NoiselessSCMRecourse", "cf_faith_noiseless"
        )
        >= 0.95,
        "replay error ≤ 1e-9 (FullScheduleOracle)": get("FullScheduleOracle", "replay_error_pearl")
        <= 1e-9,
    }
    checks.append(
        {
            "check": "positive_controls_pass",
            "passed": all(conds.values()),
            "detail": "; ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in conds.items()),
        }
    )
    return checks


def seed_stability(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """Across-seed std of validity and C per (method, reading), averaged over (world, classifier)."""
    cells: dict[tuple, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        cells[(r["explainer"], r["reading"], r["run_id"])][r["explainer_seed"]].append(r)
    out: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {"validity": [], "C": []})
    for (exp, reading, _), by_seed in cells.items():
        if len(by_seed) < 2:
            continue
        v = [np.mean([x["valid"] for x in rs]) for rs in by_seed.values()]
        c = [
            np.mean([x["world_replay"] == x["target"] for x in rs if x["scorable"]] or [np.nan])
            for rs in by_seed.values()
        ]
        out[(exp, reading)]["validity"].append(float(np.std(v)))
        out[(exp, reading)]["C"].append(float(np.nanstd(c)))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in out.items()}


def _f(v: Any, fmt: str = ".2f") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return format(v, fmt) if isinstance(v, (float, int, np.floating)) else str(v)


def pilot_markdown(name: str, records, infos, summary, checks, stability, figures) -> str:
    worlds = sorted({(i["world_id"], i["family"]) for i in infos})
    archs = sorted({i["architecture"] for i in infos})
    n_expl = len(_explanations(records))
    L = [
        f"# Pilot dashboard — `{name}`",
        "",
        f"{utc_timestamp()} · {len(worlds)} worlds × {len(archs)} classifiers · {n_expl} "
        "explanations. Pooled point estimates, no uncertainty yet (Week 8). Controls (‡) are "
        "not competing explainers.",
        "",
        "## Week-5 acceptance checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        *[
            f"| {c['check']} | {'**PASS**' if c['passed'] else '**FAIL**'} | {c['detail']} |"
            for c in checks
        ],
        "",
        "## Classifiers",
        "",
        "| world | family | classifier | test acc | ECE | Brier | acc under uniform-innovation shift |",
        "|---|---|---|---|---|---|---|",
    ]
    for i in sorted(infos, key=lambda i: (i["world_id"], i["architecture"])):
        c = i["classifier"]
        L.append(
            f"| {i['world_id']} | {i['family']} | {i['architecture']} | {_f(c['test']['accuracy'])} | "
            f"{_f(c['test']['ece'], '.3f')} | {_f(c['test']['brier'], '.3f')} | "
            f"{_f(c['shift_uniform_innovations']['accuracy'])} |"
        )
    L += [
        "",
        "## Pooled outcomes by method and reading",
        "",
        "| method | reading | V | P-faith | R-faith | D | E_rep | A | B | C | Δtraj | Δout | Δtotal | n_s | abst | inval |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        summary, key=lambda r: (r.get("is_control", False), r["explainer"], r["reading"])
    ):
        tag = " ‡" if r.get("is_control") else ""
        L.append(
            f"| {r['explainer']}{tag} | {r['reading']} | {_f(r['validity'])} | {_f(r['cf_faith_pearl'])} | "
            f"{_f(r['cf_faith_noiseless'])} | {_f(r['D_pearl'], '.1f')} | {_f(r['replay_error_pearl'], '.2g')} | "
            f"{_f(r['proposal_model_success'])} | {_f(r['replay_model_success'])} | {_f(r['replay_world_success'])} | "
            f"{_f(r['delta_traj'], '+.2f')} | {_f(r['delta_outcome'], '+.2f')} | {_f(r['delta_total'], '+.2f')} | "
            f"{r['n_scorable']} | {r['n_abstain']} | {r['n_invalid_action']} |"
        )
    # failure taxonomy and cost
    L += [
        "",
        "## Failure taxonomy and cost (one row per method; each explanation counted once)",
        "",
        "| method | explanations | status counts | median runtime s | p90 runtime s | median queries |",
        "|---|---|---|---|---|---|",
    ]
    by_m: dict[str, list] = defaultdict(list)
    for r in _explanations(records):
        by_m[r["explainer"]].append(r)
    for m, rs in sorted(by_m.items()):
        st = Counter(r["status"] for r in rs)
        rt = np.array([r["runtime_seconds"] for r in rs])
        q = np.array([r["queries"] for r in rs if r.get("queries") is not None], float)
        L.append(
            f"| {m} | {len(rs)} | {dict(st)} | {_f(float(np.median(rt)), '.3f')} | "
            f"{_f(float(np.quantile(rt, 0.9)), '.3f')} | {_f(float(np.median(q)) if q.size else None, '.0f')} |"
        )
    errors = Counter(
        (r["explainer"], (r.get("error") or "")[:120])
        for r in _explanations(records)
        if r.get("error")
    )
    if errors:
        L += ["", "Most frequent recorded errors:", ""]
        L += [f"- {m}: `{e}` ×{n}" for (m, e), n in errors.most_common(8)]
    skipped = [
        (i["world_id"], i["architecture"], s["explainer"], s["reason"][:90])
        for i in infos
        for s in i.get("skipped_explainers", [])
    ]
    if skipped:
        L += ["", "Skipped (method cannot run on the world/regime):", ""]
        L += [f"- {w} / {a}: {e} — {why}" for w, a, e, why in skipped]
    L += [
        "",
        "## Seed stability (mean across-seed std over world × classifier cells)",
        "",
        "| method | reading | std(validity) | std(C) |",
        "|---|---|---|---|",
    ]
    for (m, rd), s in sorted(stability.items()):
        L.append(f"| {m} | {rd} | {_f(s['validity'], '.3f')} | {_f(s['C'], '.3f')} |")
    L += [
        "",
        "## Per world and classifier (first-slice reading for trajectory methods, native for "
        "action-declaring ones)",
        "",
        "| world | classifier | method | V | A | B | C | Δtotal | D |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_run: dict[str, list] = defaultdict(list)
    for r in records:
        by_run[r["run_id"]].append(r)
    for i in sorted(infos, key=lambda i: (i["world_id"], i["architecture"])):
        for row in summarize(by_run[i["run_id"]]):
            if row["reading"] not in (FIRST_SLICE, "native"):
                continue
            L.append(
                f"| {i['world_id']} | {i['architecture']} | {row['explainer']} | {_f(row['validity'])} | "
                f"{_f(row['proposal_model_success'])} | {_f(row['replay_model_success'])} | "
                f"{_f(row['replay_world_success'])} | {_f(row['delta_total'], '+.2f')} | {_f(row['D_pearl'], '.1f')} |"
            )
    if figures:
        L += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figures]
    return "\n".join(L) + "\n"


def plot_abc(summary: list[dict[str, Any]], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in summary
        if r["reading"] in (FIRST_SLICE, "native")
        and np.isfinite(
            r["proposal_model_success"] if r["proposal_model_success"] is not None else np.nan
        )
    ]
    rows.sort(key=lambda r: (r.get("is_control", False), r["explainer"]))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(rows) + 2), 3.4))
    for j, (key, label) in enumerate(
        [
            ("proposal_model_success", "A: model on proposal"),
            ("replay_model_success", "B: model on replay"),
            ("replay_world_success", "C: world on replay"),
        ]
    ):
        ax.bar(x + (j - 1) * 0.26, [r[key] for r in rows], 0.26, label=label)
    ax.set_xticks(
        x,
        [r["explainer"] + (" ‡" if r.get("is_control") else "") for r in rows],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (scorable units)")
    ax.set_title(
        "Model-vs-world: A → B → C (first-slice reading / native action)", fontsize=9, pad=30
    )
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_pilot_report(name: str, run_dirs: list[Path], out_dir: str | Path) -> Path:
    records, infos = load_runs(run_dirs)
    summary = summarize(records)
    checks = acceptance_checks(records, infos, summary)
    stability = seed_stability(records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    figs = [plot_abc(summary, out / f"{stem}__abc.png")]
    md = pilot_markdown(name, records, infos, summary, checks, stability, figs)
    path = out / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    write_json(
        out / f"{stem}.json",
        {
            "name": name,
            "runs": [str(d) for d in run_dirs],
            "checks": checks,
            "summary": summary,
            "seed_stability": {f"{k[0]}|{k[1]}": v for k, v in stability.items()},
            "classifiers": [
                {k: i[k] for k in ("world_id", "architecture", "classifier")} for i in infos
            ],
        },
    )
    return path
