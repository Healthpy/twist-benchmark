"""Robustness, uncertainty and action-sensitivity study (plan §Week 7).

Per world × classifier, three kinds of run share one code path (``run_paired_experiment``):

* **base**: every explainer and seed, factual units from the base test split;
* **shift** (one per pre-declared spec): the classifier, reference data, estimated mechanisms and
  explainer knowledge stay those of the base world; factual units come from the shifted deployment
  world, which is also the replay oracle. Retraining shifts change the evaluation classifier;
* **perturb**: the base units with innovations perturbed by ``perturb_scale`` × std (same units),
  for explanation stability.

The report (``build_robustness_report``) computes, per method:
- hierarchical-bootstrap CIs (worlds → units) for every headline metric, base and per shift;
- stability under perturbation;
- ``p_robust`` over an ensemble of plausible mechanisms;
- calibration of reported target probabilities;
- explanation-set variability across seeds;
- a tolerance-sensitivity profile;
- a brittleness verdict.

Pre-declared acceptance checks:

* every headline metric of every (method, condition) row with scorable units has a finite CI;
* every method evaluated on exact worlds has a tolerance profile;
* the brittleness verdict ("model-valid but brittle under mechanism uncertainty": A ≥ 0.8 and the
  95% interval of some world-success drop — under a mechanism shift (world-blocked bootstrap) or
  the ensemble (``C − p_robust``, paired) — lies entirely above 0.10) is computed for every method
  and discriminates, i.e. flags at least one method and clears at least one. (Week 7 used the
  point estimate; the CI-aware rule replaced it in Week 8, as decided before Week 8.)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from twist.config import ExperimentConfig
from twist.evaluation.multiworld_runner import MultiWorldConfig, make_tasks
from twist.evaluation.pilot_report import load_runs
from twist.evaluation.statistical_analysis import blocked_difference
from twist.metrics.aggregation import hierarchical_bootstrap
from twist.metrics.conventional import diversity
from twist.metrics.robustness import (
    ensemble_outcomes,
    mechanism_ensemble,
    probability_calibration,
    stability_pairs,
    unit_index,
)
from twist.metrics.sensitivity import TAUS, ranking_stability, tolerance_profile
from twist.utils.logging import get_logger
from twist.utils.seeds import derive_seed
from twist.utils.serialization import utc_timestamp, write_json
from twist.worlds.shifts import shift_id

log = get_logger(__name__)
NATIVE, FIRST = "native", "first_change_atomic"
MODEL_VALID, BRITTLE_DROP = 0.8, 0.10
SENSITIVITY_BANDS = (0.01, 0.05)  # replay-error increase: low < 0.01 ≤ moderate < 0.05 ≤ high


class RobustnessConfig(MultiWorldConfig):
    shifts: list[dict[str, Any]] = Field(default_factory=list)
    shift_factuals: int = 30
    perturb_scale: float = 0.05
    ensemble_size: int = 8
    ensemble_rel_scale: float = 0.25
    taus: list[float] = Field(default_factory=lambda: list(TAUS))
    n_boot: int = 1000


def make_robustness_tasks(cfg: RobustnessConfig) -> list[tuple[ExperimentConfig, str]]:
    tasks = []
    for exp, h in make_tasks(cfg):  # base runs, all seeds
        tasks.append((exp, h))
        first_seed = exp.model_copy(
            update={
                "explainers": [
                    e
                    for e in exp.explainers
                    if e.seed == cfg.explainer_seeds[0]
                    or not any(
                        o.name == e.name and o.seed == cfg.explainer_seeds[0]
                        for o in exp.explainers
                    )
                ]
            }
        )
        tasks.append(
            (
                first_seed.model_copy(
                    update={"name": f"{exp.name}__perturb", "perturb": {"scale": cfg.perturb_scale}}
                ),
                h,
            )
        )
        for spec in cfg.shifts:
            tasks.append(
                (
                    first_seed.model_copy(
                        update={
                            "name": f"{exp.name}__shift-{shift_id(spec)}",
                            "shift": spec,
                            "n_factuals": cfg.shift_factuals,
                        }
                    ),
                    h,
                )
            )
    return tasks


def run_robustness(cfg: RobustnessConfig) -> list[Path]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from twist.evaluation.multiworld_runner import _run_task

    payloads = [(t.model_dump(), h) for t, h in make_robustness_tasks(cfg)]
    log.info("%d robustness tasks, %d jobs", len(payloads), cfg.jobs)
    if cfg.jobs <= 1:
        results = [_run_task(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=cfg.jobs, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_run_task, payloads))
    return [Path(r) for r in results if r]


# ----------------------------------------------------------------------------------- analysis
def _condition(info: dict[str, Any]) -> str:
    if info.get("shift"):
        return shift_id(info["shift"])
    return "perturb" if info.get("perturb") else "base"


def _method_reading(r: dict[str, Any]) -> bool:
    return r["reading"] == (NATIVE if r["native_action"] or r["reading"] == NATIVE else FIRST)


def _rate(key: str, value: Any = None):
    def stat(rs: list[dict]) -> float:
        s = [r for r in rs if r["scorable"]]
        if not s:
            return float("nan")
        if value == "target":
            return float(np.mean([r[key] == r["target"] for r in s]))
        return (
            float(np.nanmedian([r[key] for r in s]))
            if key == "replay_error_pearl"
            else float(np.mean([bool(r[key]) for r in s]))
        )

    return stat


HEADLINE = {
    "A": _rate("h_proposal", "target"),
    "B": _rate("h_replay", "target"),
    "C": _rate("world_replay", "target"),
    "feasible": _rate("feasible"),
    "E_replay": _rate("replay_error_pearl"),
}


def _ci_rows(records: list[dict], n_boot: int) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        if _method_reading(r):
            groups[(r["explainer"], r["condition"])].append(r)
    for key, rs in groups.items():
        row: dict[str, Any] = {"n_scorable": sum(r["scorable"] for r in rs), "n": len(rs)}
        for name, fn in HEADLINE.items():
            row[name] = hierarchical_bootstrap(
                rs, fn, ("world_id", "unit"), n_boot=n_boot, seed=derive_seed(0, *key, name)
            )
        rows[key] = row
    return rows


def _p_robust(
    base_infos: list[dict], records: list[dict], cfg: RobustnessConfig
) -> dict[str, float]:
    """explanation_id → p_robust over the mechanism ensemble (base runs, first seed)."""
    from twist.actions.schemas import ActionPlan
    from twist.worlds.base_scm import TemporalSCM
    from twist.worlds.io import load_world_data

    out: dict[str, float] = {}
    for info in base_infos:
        world = TemporalSCM.load(info["world_dir"])
        data = load_world_data(info["world_dir"])
        ens = mechanism_ensemble(world, cfg.ensemble_size, cfg.ensemble_rel_scale, seed=1)
        rs = [
            r
            for r in records
            if r["run_id"] == info["run_id"]
            and _method_reading(r)
            and r["scorable"]
            and r["action"]
        ]
        if not rs:
            continue
        states = np.stack(
            [data["exo"]["state"][data["n_train"] + unit_index(r["factual_id"])] for r in rs]
        )
        plans = [
            world.to_state_plan(ActionPlan.from_dict(r["action"]))
            if r["action"]["semantics"].endswith("@observed")
            else ActionPlan.from_dict(r["action"])
            for r in rs
        ]
        y = ensemble_outcomes(ens, states, plans)
        targets = np.array([r["target"] for r in rs])
        for r, p in zip(rs, (y == targets[None]).mean(axis=0), strict=True):
            out[r["explanation_id"]] = float(p)
    return out


def _proposals(info: dict[str, Any], records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """explainer → {explanation_id: proposal} from the saved arrays of one run."""
    out: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    rs = {
        r["explanation_id"]: r
        for r in records
        if r["run_id"] == info["run_id"] and _method_reading(r)
    }
    for f in {r["arrays_file"] for r in rs.values()}:
        a = np.load(f)
        for eid, p in zip(a["explanation_ids"], a["proposals"], strict=True):
            if str(eid) in rs:  # method name from the record, never parsed from file names
                out[rs[str(eid)]["explainer"]][str(eid)] = p
    return out


def build_robustness_report(cfg: RobustnessConfig, run_dirs: list[Path]) -> Path:
    from twist.worlds.base_scm import TemporalSCM
    from twist.worlds.io import load_world_data

    records, infos = load_runs(run_dirs)
    cond_of = {i["run_id"]: _condition(i) for i in infos}
    for r in records:
        r["condition"] = cond_of[r["run_id"]]
        r["unit"] = f"{r['world_id']}/{unit_index(r['factual_id'])}"
    base_infos = [i for i in infos if cond_of[i["run_id"]] == "base"]
    seed0 = cfg.explainer_seeds[0]
    rows = _ci_rows(records, cfg.n_boot)
    methods = sorted({k[0] for k in rows})
    conditions = ["base"] + [shift_id(s) for s in cfg.shifts]

    # stability (base seed0 vs perturbed)
    base0 = [
        r
        for r in records
        if r["condition"] == "base"
        and _method_reading(r)
        and r.get("explainer_seed") in (seed0, None)
    ]
    pert = [r for r in records if r["condition"] == "perturb" and _method_reading(r)]
    stab = {k[0]: v for k, v in stability_pairs(base0, pert).items()}
    # p_robust over the mechanism ensemble
    pr = _p_robust(base_infos, base0, cfg)
    for r in base0:
        r["p_robust"] = pr.get(r["explanation_id"], float("nan"))
    # tolerance profiles (pooled over exact base worlds) and seed variability
    profiles: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    variability: dict[str, list[float]] = defaultdict(list)
    for info in base_infos:
        world = TemporalSCM.load(info["world_dir"])
        data = load_world_data(info["world_dir"])
        props = _proposals(info, records)
        run_recs = {
            r["explanation_id"]: r
            for r in records
            if r["run_id"] == info["run_id"] and _method_reading(r)
        }
        scale = data["x"].reshape(-1, data["x"].shape[-1]).std(0)
        for m, pmap in props.items():
            recs_m = [run_recs[e] for e in pmap if e in run_recs]
            first = [r for r in recs_m if r.get("explainer_seed") in (seed0, None)]
            if first:  # tolerance profile on the first-seed proposals
                P = np.stack([pmap[r["explanation_id"]] for r in first])
                X = np.stack(
                    [data["x"][data["n_train"] + unit_index(r["factual_id"])] for r in first]
                )
                for tau, v in (
                    tolerance_profile(world, X, {m: P}, tuple(cfg.taus)).get(m, {}).items()
                ):
                    profiles[m][tau].append(v)
            by_unit: dict[int, list[np.ndarray]] = defaultdict(list)  # seed variability
            for r in recs_m:
                p = pmap[r["explanation_id"]]
                if np.all(np.isfinite(p)):
                    by_unit[unit_index(r["factual_id"])].append(p)
            variability[m] += [
                diversity(np.stack(v), scale) for v in by_unit.values() if len(v) > 1
            ]
    pooled = {
        m: {
            tau: {k: float(np.nanmean([v[k] for v in vs])) for k in vs[0]} for tau, vs in tp.items()
        }
        for m, tp in profiles.items()
    }
    rank_stab = ranking_stability(pooled) if pooled else {}

    # per-method summary, verdicts
    cards = []
    mech_conds = [shift_id(s) for s in cfg.shifts if s["level"] == "mechanism"]
    for m in methods:
        base = rows.get((m, "base"))
        if base is None:
            continue
        rs = [r for r in base0 if r["explainer"] == m and r["scorable"]]
        p_rob = hierarchical_bootstrap(
            rs,
            lambda xs: float(np.nanmean([x["p_robust"] for x in xs])) if xs else float("nan"),
            ("world_id", "unit"),
            cfg.n_boot,
        )
        # world-success drops with uncertainty (Week-8 rule): mechanism shifts use a world-blocked
        # bootstrap (different units); the ensemble drop C − p_robust is paired per unit
        base_rs = [
            r
            for r in records
            if r["explainer"] == m and r["condition"] == "base" and _method_reading(r)
        ]
        drop_cis: dict[str, tuple[float, float, float]] = {}
        for c in mech_conds:
            sh = [
                r
                for r in records
                if r["explainer"] == m and r["condition"] == c and _method_reading(r)
            ]
            if sh:
                drop_cis[c] = blocked_difference(
                    base_rs, sh, HEADLINE["C"], cfg.n_boot, seed=derive_seed(0, m, c)
                )
        ens = [
            {
                "world_id": r["world_id"],
                "unit": r["unit"],
                "d": float(r["world_replay"] == r["target"]) - r["p_robust"],
            }
            for r in rs
            if np.isfinite(r.get("p_robust", np.nan))
        ]
        if ens:
            drop_cis["ensemble"] = hierarchical_bootstrap(
                ens,
                lambda xs: float(np.mean([x["d"] for x in xs])),
                ("world_id", "unit"),
                cfg.n_boot,
            )
        drops = [v[0] for v in drop_cis.values() if np.isfinite(v[0])]
        worst = max(
            drop_cis,
            key=lambda k: drop_cis[k][0] if np.isfinite(drop_cis[k][0]) else -np.inf,
            default=None,
        )
        model_valid = base["A"][0] >= MODEL_VALID
        # brittle only when some drop's whole 95% interval lies above the threshold
        brittle = model_valid and any(
            np.isfinite(v[1]) and v[1] >= BRITTLE_DROP for v in drop_cis.values()
        )
        e_base = base["E_replay"][0]
        e_mech = [rows[(m, c)]["E_replay"][0] for c in mech_conds if (m, c) in rows]
        d_e = max([e - e_base for e in e_mech if np.isfinite(e)], default=float("nan"))
        sens = (
            "–"
            if not np.isfinite(d_e)
            else "low"
            if d_e < SENSITIVITY_BANDS[0]
            else "moderate"
            if d_e < SENSITIVITY_BANDS[1]
            else "high"
        )
        rep = [r for r in base0 if r["explainer"] == m]
        cal = probability_calibration(
            np.array(
                [
                    r.get("p_target_reported") if r.get("p_target_reported") is not None else np.nan
                    for r in rep
                ],
                float,
            ),
            np.array([r["world_replay"] == r["target"] for r in rep]),
            np.array([r.get("p_world_interventional", np.nan) for r in rep], float),
        )
        cards.append(
            {
                "method": m,
                "base": base,
                "by_condition": {c: rows.get((m, c)) for c in conditions},
                "p_robust": p_rob,
                "stability": stab.get(m, {}),
                "calibration": cal,
                "tolerance_profile": pooled.get(m, {}),
                "seed_variability": float(np.mean(variability[m]))
                if variability.get(m)
                else float("nan"),
                "replay_error_sensitivity": sens,
                "replay_error_increase": d_e,
                "model_valid": model_valid,
                "brittle": brittle,
                "max_world_success_drop": max(drops) if drops else float("nan"),
                "max_drop_ci": drop_cis.get(worst) if worst else None,
                "max_drop_condition": worst,
                "drop_cis": drop_cis,
            }
        )

    checks = _checks(rows, cards)
    out = Path(cfg.output_dir) / cfg.name / "report"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    figs = [
        _heatmap(cards, conditions, out / f"{stem}__shift_heatmap.png"),
        _tolerance_plot(pooled, out / f"{stem}__tolerance.png"),
    ]
    md = _markdown(cfg, infos, cards, conditions, checks, rank_stab, figs)
    path = out / f"{stem}.md"
    path.write_text(md, encoding="utf-8")
    write_json(
        out / f"{stem}.json",
        {
            "checks": checks,
            "cards": cards,
            "ranking_stability": rank_stab,
            "runs": [str(d) for d in run_dirs],
        },
    )
    return path


def _checks(rows: dict, cards: list[dict]) -> list[dict[str, Any]]:
    missing = [
        f"{m}/{c}/{k}"
        for (m, c), row in rows.items()
        if row["n_scorable"] > 0
        for k in ("A", "B", "C", "feasible")
        if not np.isfinite(row[k][1])
    ]
    with_prof = [c["method"] for c in cards if c["tolerance_profile"]]
    flagged = [c["method"] for c in cards if c["brittle"]]
    cleared = [c["method"] for c in cards if not c["brittle"]]
    return [
        {
            "check": "every_headline_metric_has_uncertainty",
            "passed": not missing,
            "detail": f"{len(rows)} method×condition rows; missing CIs: {missing[:5] or 'none'}",
        },
        {
            "check": "every_method_has_tolerance_profile",
            "passed": len(with_prof) == len(cards),
            "detail": f"{len(with_prof)}/{len(cards)} methods profiled",
        },
        {
            "check": "brittleness_identified",
            "passed": bool(flagged) and bool(cleared),
            "detail": f"flagged {flagged or 'none'}; cleared {len(cleared)}",
        },
    ]


def _fmt_ci(t: tuple[float, float, float] | None) -> str:
    if t is None or not np.isfinite(t[0]):
        return "–"
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if np.isfinite(t[1]) else f"{t[0]:.2f}"


def _f(v: Any, fmt: str = ".2f") -> str:
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def _markdown(cfg, infos, cards, conditions, checks, rank_stab, figs) -> str:
    worlds = sorted({i["world_id"] for i in infos})
    L = [
        f"# Robustness, uncertainty and action sensitivity — `{cfg.name}`",
        "",
        f"{utc_timestamp()} · worlds {worlds} · 95% CIs from a hierarchical bootstrap "
        f"(worlds → units, {cfg.n_boot} draws). Trajectory methods: first-slice reading; "
        "action-native methods: declared action. ‡ = control.",
        "",
        "## Week-7 acceptance checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        *[
            f"| {c['check']} | {'**PASS**' if c['passed'] else '**FAIL**'} | {c['detail']} |"
            for c in checks
        ],
        "",
        "## World success C under shifts",
        "",
        "| method | " + " | ".join(conditions) + " |",
        "|---|" + "---|" * len(conditions),
    ]
    for c in cards:
        L.append(
            f"| {c['method']} | "
            + " | ".join(
                _fmt_ci(c["by_condition"][k]["C"]) if c["by_condition"].get(k) else "–"
                for k in conditions
            )
            + " |"
        )
    L += [
        "",
        "## Summary",
        "",
        "| method | A | C | p_robust | largest C drop [95% CI] | action stability | replay-error sensitivity | calibration gap | seed variability | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cards:
        verdict = (
            "**model-valid, brittle under mechanism uncertainty**"
            if c["brittle"]
            else (
                "model-valid; no significant mechanism-shift drop"
                if c["model_valid"]
                else "not model-valid"
            )
        )
        L.append(
            f"| {c['method']} | {_fmt_ci(c['base']['A'])} | {_fmt_ci(c['base']['C'])} | {_fmt_ci(c['p_robust'])} | "
            f"{_fmt_ci(c['max_drop_ci'])} ({c['max_drop_condition'] or '–'}) | {_f(c['stability'].get('action_stability'))} | "
            f"{c['replay_error_sensitivity']} | {_f(c['calibration']['abs_gap'])} | "
            f"{_f(c['seed_variability'], '.3f')} | {verdict} |"
        )
    L += [
        "",
        "## Tolerance sensitivity",
        "",
        "Ranking stability of methods by D (Kendall τ vs τ = 1e-3·scale): "
        + (", ".join(f"τ={t:g}: {_f(v)}" for t, v in sorted(rank_stab.items())) or "–"),
        "",
        "| method | "
        + " | ".join(f"D @ {t:g}" for t in cfg.taus)
        + " | "
        + " | ".join(f"CF-faith @ {t:g}" for t in cfg.taus)
        + " |",
        "|---|" + "---|" * (2 * len(cfg.taus)),
    ]
    for c in cards:
        tp = c["tolerance_profile"]
        L.append(
            f"| {c['method']} | "
            + " | ".join(_f(tp.get(t, {}).get("D"), ".1f") for t in cfg.taus)
            + " | "
            + " | ".join(_f(tp.get(t, {}).get("cf_faith")) for t in cfg.taus)
            + " |"
        )
    L += ["", "## Method robustness cards", ""]
    for c in cards:
        bc, s = c["by_condition"], c["stability"]

        def cond(prefix: str, bc: dict = bc) -> str:
            vals = [(k, v) for k, v in bc.items() if k.startswith(prefix) and v]
            return "; ".join(f"{k}: {_fmt_ci(v['C'])}" for k, v in vals) or "–"

        L += [
            f"### {c['method']}",
            "",
            "```text",
            f"Base world success:            {_fmt_ci(c['base']['C'])}",
            f"Innovation-shift success:      {cond('innovation')}",
            f"Mechanism-shift success:       {cond('mechanism')}",
            f"Observation-shift success:     {cond('observation')}",
            f"Decision-shift success:        {cond('decision')}",
            f"p_robust (mechanism ensemble): {_fmt_ci(c['p_robust'])}",
            f"Action stability:              {_f(s.get('action_stability'))} (onset {_f(s.get('onset'))}, variables {_f(s.get('variables'))}, outcome {_f(s.get('outcome'))}, target {_f(s.get('target'))})",
            f"Replay-error sensitivity:      {c['replay_error_sensitivity']} (Δ median E_replay {_f(c['replay_error_increase'], '.3g')})",
            f"Reported-probability calibration: gap {_f(c['calibration']['abs_gap'])}, ECE {_f(c['calibration']['ece'])} (n={int(c['calibration']['n'])})",
            f"Seed variability (proposal diversity): {_f(c['seed_variability'], '.3f')}",
            "```",
            "",
        ]
    L += ["## Figures", ""] + [f"![{p.stem}]({p.name})" for p in figs if p]
    return "\n".join(L) + "\n"


def _heatmap(cards, conditions, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.array(
        [
            [
                c["by_condition"][k]["C"][0] if c["by_condition"].get(k) else np.nan
                for k in conditions
            ]
            for c in cards
        ]
    )
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(conditions), 0.4 * len(cards) + 1.6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(conditions)), conditions, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(cards)), [c["method"] for c in cards], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if M[i, j] < 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, label="C: world success")
    ax.set_title("World success under shifts", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _tolerance_plot(pooled, path: Path) -> Path | None:
    if not pooled:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for m, prof in sorted(pooled.items()):
        taus = sorted(prof)
        a1.plot(taus, [prof[t]["D"] for t in taus], marker="o", label=m)
        a2.plot(taus, [prof[t]["cf_faith"] for t in taus], marker="o", label=m)
    for a, lab in ((a1, "D (imposed timesteps)"), (a2, "hard Pearl CF-faith")):
        a.set_xscale("log")
        a.set_xlabel("tolerance τ (× channel std)")
        a.set_ylabel(lab)
    a2.set_ylim(-0.05, 1.05)
    a1.legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle("Tolerance sensitivity", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    from twist.config import load_yaml

    return RobustnessConfig.model_validate(load_yaml(path))
