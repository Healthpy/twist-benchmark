"""Benchmark validity (plan §Week 8, tasks 1–2): do the metrics measure what they claim?

**Construct validity.** Controlled pairs that differ in exactly one property are generated from a
successful oracle explanation (``PearlSCMRecourse``: a native point action whose proposal is its
own Pearl replay and that reaches the world target). The metric for the varied property must
separate the pair (paired 95% CI excluding 0 in the expected direction), and the held property
must not differ (paired CI containing 0, or identical by construction).

| Pair | held equal | varied | construction |
|---|---|---|---|
| A | model validity | mechanism consistency | same action; proposal jittered after the action, classifier decision kept |
| B | proximity | world success | one-entry edit of identical size at the label site: label channel vs distractor channel |
| C | replay error | feasibility | same native action; amplitude inside vs outside the bound (both replay exactly) |
| D | world success | action cost | the action vs a 1.5× version that also reaches the target |
| E | plausibility | causal-support correctness | the same edit shape aligned with the action vs shifted 5 steps before it |

**Discriminant validity.** Properties that must not collapse into one another:

1. low replay error ⇏ low action cost — among explanations with replay error ≤ 1e-9, action size
   spans from one timestep to ≥ 0.9·T (Pearl control vs full-schedule oracle);
2. high plausibility ⇏ causal correctness — most mechanism-wrong edits are plausible
   (ΔOOD ≥ −0.05) and not mechanism-consistent (replay error ≥ 1e-3, the action-grounded threshold);
3. high target validity ⇏ safe-world success — a method with validity ≥ 0.95 has safe-world
   success ≤ 0.5 (trajectory baseline, first-slice reading);
4. low total disagreement ⇏ good recourse when A is low — under-powered oracle actions (0.1× the
   recourse action; the proposal is its own replay) have |A − C| ≤ 0.1 while A ≤ 0.2.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from twist.actions.feasibility import AmplitudeBounds
from twist.actions.observed_variable import PointIntervention
from twist.actions.schemas import ActionPlan
from twist.config import ClassifierConfig, EvaluationConfig, load_yaml
from twist.evaluation.result_schema import Explanation
from twist.evaluation.statistical_analysis import paired_difference
from twist.utils.logging import get_logger
from twist.utils.serialization import utc_timestamp, write_json

log = get_logger(__name__)


class ValidityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "validity"
    output_dir: str = "artifacts/validity"
    suite: str
    worlds: list[str]
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    n_factuals: int = 60
    seed: int = 501
    recourse: dict[str, Any] = Field(
        default_factory=lambda: {"steps": 200, "t0_fracs": [0.5, 0.75, 0.9]}
    )
    n_boot: int = 1000

    @model_validator(mode="before")
    @classmethod
    def _resolve_refs(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("classifier"), str):
            data = {**data, "classifier": load_yaml(data["classifier"])}
        return data


def _explain(name: str, xs, ids, tg, ctx, **params) -> list[Explanation]:
    from twist.explainers.registry import build_explainer

    return build_explainer(name, seed=501, **params).explain(xs, ids, tg, ctx)


def _variant(
    e: Explanation,
    tag: str,
    proposal: np.ndarray | None = None,
    action: Any = None,
    native: bool = True,
) -> Explanation:
    v = copy.copy(e)
    v.explanation_id = f"{e.explanation_id}-{tag}"
    v.explainer = tag
    if proposal is not None:
        v.proposed_trajectory = proposal
    if action is not None:
        v.action = ActionPlan(action, native=True, semantics="native_point_pearl")
    if not native:
        v.action, v.declares_actions = None, False
    return v


Base = tuple[Explanation, PointIntervention, np.ndarray, np.ndarray]  # (e, action, x, proposal)


def _strengthen(e: Explanation, x: np.ndarray, world: Any, clf: Any) -> Base | None:
    """The oracle's point action, amplified by the smallest factor in (1 … 4) for which both the
    classifier and the world reach the target (the optimizer stops at the decision boundary)."""
    from twist.replay import replay_pearl

    if e.action is None or not isinstance(e.action.action, PointIntervention):
        return None
    a = e.action.action
    x0 = x[a.t, a.targets]
    for f in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        act = PointIntervention(a.t, a.targets, (x0 + f * (np.asarray(a.values) - x0)).tolist())
        prop = replay_pearl(x, act, world)
        if int(world.label(prop)) == e.target and int(clf.predict(prop[None])[0]) == e.target:
            return _variant(e, "base", proposal=prop, action=act), act, x, prop
    return None


def run_world(cfg: ValidityConfig, world_id: str) -> list[dict[str, Any]]:
    """All validity records of one world (tagged with ``pair``/``variant``/``check``)."""
    from twist.classifiers.train import train_classifier
    from twist.evaluation.multiworld_runner import suite_world_config
    from twist.evaluation.paired_runner import generate_world_data, select_factuals
    from twist.evaluation.protocol import evaluate_explanations
    from twist.explainers.mechanism_access.regimes import build_context
    from twist.metrics.conventional import OODScorer, channel_iqr
    from twist.replay import replay_pearl
    from twist.worlds.io import subset_exo

    wcfg, _ = suite_world_config(cfg.suite, world_id)
    data = generate_world_data(wcfg)
    world = data["world"]
    if not world.assumptions.get("exact_abduction", True):
        raise NotImplementedError("validity pairs need an exact world")
    clf = train_classifier(
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
        **cfg.classifier.model_dump(),
    )
    idx, tg = select_factuals(clf, data["x_test"], 1, cfg.n_factuals)
    xs, exo = data["x_test"][idx], subset_exo(data["exo_test"], idx)
    ids = [f"{world_id}/test/{i}" for i in idx]
    ctx5 = build_context("R5", clf, world, train_x=data["x_train"], train_y=data["y_train"])
    ev = EvaluationConfig(
        readings=["first_change_atomic", "full_changed_schedule"], compute_dtw=False
    )
    ood = OODScorer(data["x_train"])
    iqr = channel_iqr(data["x_train"])
    rng = np.random.default_rng(cfg.seed)
    T, k = world.T, world.k
    sd = data["x_train"].reshape(-1, k).std(0)

    def evaluate(expls, constraints=None):
        if not expls:
            return []
        pos = {e.factual_id: i for i, e in enumerate(base)}
        sel = [pos[e.factual_id] for e in expls]
        return evaluate_explanations(
            world, clf, xs[sel], expls, ev, constraints, iqr, ood, exogenous=subset_exo(exo, sel)
        )

    base = _explain("PearlSCMRecourse", xs, ids, tg, ctx5, **cfg.recourse)
    base_recs = {r["explanation_id"]: r for r in evaluate(base)}
    good = [g for e, x in zip(base, xs, strict=True) if (g := _strengthen(e, x, world, clf))]
    log.info("%s: %d/%d base actions reach the target", world_id, len(good), len(base))
    out: list[dict[str, Any]] = []

    def tag(recs, pair, variant):
        for r in recs:
            if r["reading"] in ("native", "full_changed_schedule"):
                r.update(
                    pair=pair,
                    variant=variant,
                    world_id=world_id,
                    architecture="-",
                    unit=r["factual_id"].rsplit("/", 1)[1],
                    explainer_seed=0,
                )
                out.append(r)

    # Pair A: same validity, different mechanism consistency
    a1, a2 = [], []
    for e, a, _x, prop0 in good:
        for _ in range(5):
            prop = prop0.copy()
            prop[a.t + 1 :] += 0.05 * sd * rng.normal(size=prop[a.t + 1 :].shape)
            if clf.predict(prop[None])[0] == e.target:
                a1.append(_variant(e, "A:consistent"))
                a2.append(_variant(e, "A:jittered", proposal=prop))
                break
    tag(evaluate(a1), "A", "consistent")
    tag(evaluate(a2), "A", "jittered")
    # Pair B: same proximity, different world success (imposed one-entry edits at the label site)
    lab = world.label
    d = world.roles.get("distractor")
    if getattr(lab, "type", "") == "threshold" and d is not None:  # one-entry label site
        site, ch = lab.site(T), lab.channel
        b1, b2 = [], []
        for e, x in zip(base, xs, strict=True):
            delta = (lab.threshold - x[site, ch]) + 0.5 * sd[ch]
            if delta <= 0:
                continue
            p1, p2 = x.copy(), x.copy()
            p1[site, ch] += delta
            p2[site, d] += delta
            b1.append(_variant(e, "B:label-channel", proposal=p1, native=False))
            b2.append(_variant(e, "B:distractor", proposal=p2, native=False))
        tag(evaluate(b1), "B", "label-channel")
        tag(evaluate(b2), "B", "distractor")

    # Pair C: same replay error (exact), different feasibility
    bound = max(
        (float(np.max(np.abs(np.asarray(a.values) - x[a.t, a.targets]))) for _, a, x, _ in good),
        default=1.0,
    )
    cons = [AmplitudeBounds(1.1 * bound)]
    c1, c2 = [], []
    for e, a, x, _ in good:
        x0 = x[a.t, a.targets]
        sign = np.sign(np.asarray(a.values) - x0 + 1e-12)
        big = PointIntervention(a.t, a.targets, (x0 + 1.5 * bound * sign).tolist())
        c1.append(_variant(e, "C:within"))
        c2.append(_variant(e, "C:beyond", proposal=replay_pearl(x, big, world), action=big))
    tag(evaluate(c1, cons), "C", "within")
    tag(evaluate(c2, cons), "C", "beyond")
    # Pair D: same world success, different cost
    d1, d2 = [], []
    for e, a, x, _ in good:
        x0 = x[a.t, a.targets]
        bigger = PointIntervention(
            a.t, a.targets, (x0 + 1.5 * (np.asarray(a.values) - x0)).tolist()
        )
        prop = replay_pearl(x, bigger, world)
        if int(world.label(prop)) == e.target:
            d1.append(_variant(e, "D:action"))
            d2.append(_variant(e, "D:scaled", proposal=prop, action=bigger))
    tag(evaluate(d1), "D", "action")
    tag(evaluate(d2), "D", "scaled")
    # Pair E: same plausibility, different causal support (edit shifted before the action)
    e1, e2 = [], []
    for e, a, x, prop in good:
        if a.t < 6:
            continue
        shifted = np.roll(prop - x, -5, axis=0)
        shifted[-5:] = 0.0
        e1.append(_variant(e, "E:aligned"))
        e2.append(_variant(e, "E:retroactive", proposal=x + shifted))
    tag(evaluate(e1), "E", "aligned")
    tag(evaluate(e2), "E", "retroactive")
    # discriminant checks
    for name, params, check in [
        ("FullScheduleOracle", {}, "D1"),
        ("PlausibleButMechanismWrongEdit", {**cfg.recourse, "steps": 150}, "D2"),
        ("WachterReference", {"steps": 300}, "D3"),
    ]:
        for r in evaluate(_explain(name, xs, ids, tg, ctx5, **params)):
            r.update(
                check=check,
                world_id=world_id,
                architecture="-",
                unit=r["factual_id"].rsplit("/", 1)[1],
                explainer_seed=0,
            )
            out.append(r)
    # D4: under-powered oracle actions (×0.1): the proposal is its own replay, so A = B
    weak = []
    for e, x in zip(base, xs, strict=True):
        if e.action is not None and isinstance(e.action.action, PointIntervention):
            a = e.action.action
            x0 = x[a.t, a.targets]
            act = PointIntervention(
                a.t, a.targets, (x0 + 0.1 * (np.asarray(a.values) - x0)).tolist()
            )
            weak.append(
                _variant(e, "UnderpoweredAction", proposal=replay_pearl(x, act, world), action=act)
            )
    for r in evaluate(weak):
        r.update(
            check="D4",
            world_id=world_id,
            architecture="-",
            unit=r["factual_id"].rsplit("/", 1)[1],
            explainer_seed=0,
        )
        out.append(r)
    for r in base_recs.values():
        if r["reading"] == "native":
            r.update(
                check="D1",
                world_id=world_id,
                architecture="-",
                unit=r["factual_id"].rsplit("/", 1)[1],
                explainer_seed=0,
            )
            out.append(r)
    for r in out:
        r.pop("_proposal", None)
        r.pop("_replay", None)
    return out


# ------------------------------------------------------------------------------------- report
def _val(key: str):
    def f(r: dict) -> float:
        v = r.get(key)
        if isinstance(v, bool):
            return float(v)
        return float("nan") if v is None else float(v)

    return f


def _cost(r: dict) -> float:
    return float((r.get("costs") or {}).get("magnitude", np.nan))


def _success(r: dict) -> float:
    return float(r["scorable"] and r["world_replay"] == r["target"])


PAIRS = {
    "A": (
        "consistent",
        "jittered",
        ("model validity", _val("valid")),
        ("replay error", _val("replay_error_pearl"), -1),
    ),
    "B": (
        "label-channel",
        "distractor",
        ("L1 proximity", _val("l1")),
        ("world success C", _success, +1),
    ),
    "C": (
        "within",
        "beyond",
        ("replay error", _val("replay_error_pearl")),
        ("feasible", _val("feasible"), +1),
    ),
    "D": ("action", "scaled", ("world success C", _success), ("action magnitude", _cost, -1)),
    "E": (
        "aligned",
        "retroactive",
        ("OOD score", _val("ood_proposal")),
        ("causal-path consistency", _val("causal_path_consistency"), +1),
    ),
}
HELD_TOL = {"A": 0.0, "B": 1e-9, "C": 1e-9, "D": 0.0, "E": 0.05}


def validity_checks(records: list[dict], n_boot: int = 1000) -> dict[str, Any]:
    pairs = []
    for key, (va, vb, (held_name, held_fn), (diff_name, diff_fn, sign)) in PAIRS.items():
        ra = [r for r in records if r.get("pair") == key and r.get("variant") == va]
        rb = [r for r in records if r.get("pair") == key and r.get("variant") == vb]
        held = paired_difference(ra, rb, held_fn, n_boot=n_boot, seed=1)
        diff = paired_difference(ra, rb, diff_fn, n_boot=n_boot, seed=2)
        lo, hi = diff["ci"]
        separated = (lo > 0) if sign > 0 else (hi < 0)
        hl, hh = held["ci"]
        held_ok = (abs(held["mean_diff"]) <= HELD_TOL[key]) or (
            hl <= 0 <= hh and abs(held["mean_diff"]) <= 0.05
        )
        pairs.append(
            {
                "pair": key,
                "held": held_name,
                "varied": diff_name,
                "n_pairs": diff["n_units"],
                "held_diff": held["mean_diff"],
                "held_ci": held["ci"],
                "varied_diff": diff["mean_diff"],
                "varied_ci": diff["ci"],
                "effect_size_dz": diff["effect_size_dz"],
                "passed": bool(diff["n_units"] > 0 and separated and held_ok),
            }
        )

    def by(check: str, explainer: str | None = None) -> list[dict]:
        return [
            r
            for r in records
            if r.get("check") == check and (explainer is None or r["explainer"] == explainer)
        ]

    disc = []
    exact = [
        r
        for r in by("D1")
        if r["reading"] == "native" and r["scorable"] and r["replay_error_pearl"] <= 1e-9
    ]
    sizes = [r["action_timesteps"] for r in exact]
    T = max((r["t_label"] for r in records if r.get("t_label") is not None), default=1) + 1
    disc.append(
        {
            "check": "low replay error ⇏ low action cost",
            "evidence": f"{len(exact)} exact-replay explanations; action size {min(sizes, default=0)}–{max(sizes, default=0)} of T={T}",
            "passed": bool(sizes) and min(sizes) <= 1 and max(sizes) >= 0.9 * T,
        }
    )
    pw = [r for r in by("D2") if r["reading"] == "native" and r["scorable"]]
    frac = (
        np.mean(
            [
                (r["ood_proposal"] - r["ood_factual"]) >= -0.05 and r["replay_error_pearl"] >= 1e-3
                for r in pw
            ]
        )
        if pw
        else float("nan")
    )
    disc.append(
        {
            "check": "high plausibility ⇏ causal correctness",
            "evidence": f"{frac:.2f} of {len(pw)} mechanism-wrong edits are plausible yet not mechanism-consistent",
            "passed": bool(pw) and frac >= 0.5,
        }
    )
    w = [r for r in by("D3") if r["reading"] == "first_change_atomic"]
    validity = np.mean([r["valid"] for r in w]) if w else float("nan")
    safe = (
        np.mean(
            [bool(r["feasible"]) and r["scorable"] and r["world_replay"] == r["target"] for r in w]
        )
        if w
        else float("nan")
    )
    disc.append(
        {
            "check": "high target validity ⇏ safe-world success",
            "evidence": f"trajectory baseline: validity {validity:.2f}, safe-world success {safe:.2f}",
            "passed": bool(w) and validity >= 0.95 and safe <= 0.5,
        }
    )
    n = [r for r in by("D4") if r["reading"] == "native"]
    A = np.mean([r["h_proposal"] == r["target"] for r in n if r["scorable"]]) if n else float("nan")
    C = (
        np.mean([r["world_replay"] == r["target"] for r in n if r["scorable"]])
        if n
        else float("nan")
    )
    disc.append(
        {
            "check": "low total disagreement ⇏ good recourse when A is low",
            "evidence": f"under-powered oracle actions: A = {A:.2f}, C = {C:.2f}, abs(A − C) = {abs(A - C):.2f}",
            "passed": bool(n) and A <= 0.2 and abs(A - C) <= 0.1,
        }
    )
    return {
        "pairs": pairs,
        "discriminant": disc,
        "passed": all(p["passed"] for p in pairs) and all(d["passed"] for d in disc),
    }


def build_validity_report(cfg: ValidityConfig, records: list[dict]) -> Path:
    res = validity_checks(records, cfg.n_boot)
    out = Path(cfg.output_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}__{utc_timestamp().replace(':', '').replace('-', '')}"
    f = lambda v, fmt=".3g": "–" if v is None or not np.isfinite(v) else format(v, fmt)  # noqa: E731
    L = [
        f"# Benchmark validity report — `{cfg.name}`",
        "",
        f"{utc_timestamp()} · worlds {cfg.worlds} · overall **{'PASS' if res['passed'] else 'FAIL'}**. "
        "Paired differences (first − second variant) with hierarchical 95% CIs (worlds → units).",
        "",
        "## Construct validity (controlled pairs)",
        "",
        "| pair | held equal | Δ held [95% CI] | varied | Δ varied [95% CI] | d_z | n | result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in res["pairs"]:
        L.append(
            f"| {p['pair']} | {p['held']} | {f(p['held_diff'])} [{f(p['held_ci'][0])}, {f(p['held_ci'][1])}] | "
            f"{p['varied']} | {f(p['varied_diff'])} [{f(p['varied_ci'][0])}, {f(p['varied_ci'][1])}] | "
            f"{f(p['effect_size_dz'], '.2f')} | {p['n_pairs']} | {'**PASS**' if p['passed'] else '**FAIL**'} |"
        )
    L += [
        "",
        "## Discriminant validity",
        "",
        "| property pair | evidence | result |",
        "|---|---|---|",
    ]
    L += [
        f"| {d['check']} | {d['evidence']} | {'**PASS**' if d['passed'] else '**FAIL**'} |"
        for d in res["discriminant"]
    ]
    L += ["", "Pair constructions and criteria: `twist.evaluation.validity` module docstring."]
    path = out / f"{stem}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    write_json(out / f"{stem}.json", res)
    write_json(out / f"{stem}__records.json", records)
    return path


def run_validity(cfg: ValidityConfig) -> Path:
    from twist.utils.seeds import configure_threads

    configure_threads()
    records: list[dict] = []
    for w in cfg.worlds:
        try:
            records += run_world(cfg, w)
        except NotImplementedError as exc:
            log.warning("skipping %s: %s", w, exc)
    return build_validity_report(cfg, records)


def load_validity_config(path: str | Path) -> ValidityConfig:
    return ValidityConfig.model_validate(load_yaml(path))
