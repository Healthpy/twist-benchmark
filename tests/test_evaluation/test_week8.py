"""Week 8: statistical protocol (paired comparisons, blocked bootstrap, Pareto) and validity."""

import numpy as np
import pytest

from twist.evaluation.statistical_analysis import (
    blocked_difference,
    estimate,
    mean_of,
    paired_difference,
    pareto_front,
    proposal_success_itt,
    unit_values,
    world_success_itt,
)


def _recs(method, shift, n_worlds=3, n_units=40, seeds=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for w in range(n_worlds):
        for u in range(n_units):
            base = rng.normal()  # shared unit effect: pairing removes it
            for s in seeds:
                out.append(
                    {
                        "world_id": f"w{w}",
                        "architecture": "gru",
                        "unit": str(u),
                        "explainer_seed": s,
                        "explainer": method,
                        "v": base + shift + 0.1 * rng.normal(),
                    }
                )
    return out


def test_itt_success_counts_missing_actions_as_failures():
    ok = {"scorable": True, "world_replay": 1, "target": 1, "has_proposal": True, "h_proposal": 1}
    assert world_success_itt(ok) == 1.0 and proposal_success_itt(ok) == 1.0
    assert world_success_itt({**ok, "scorable": False}) == 0.0
    assert proposal_success_itt({**ok, "has_proposal": False}) == 0.0


def test_unit_values_average_over_seeds():
    recs = _recs("m", 0.0, n_worlds=1, n_units=2)
    uv = unit_values(recs, lambda r: r["v"])
    assert len(uv) == 2
    k = ("w0", "gru", "0")
    assert uv[k] == pytest.approx(np.mean([r["v"] for r in recs if r["unit"] == "0"]))


def test_paired_difference_detects_a_shift_hidden_by_unit_variance():
    a = _recs("a", 0.3)
    b = [{**r, "explainer": "b", "v": r["v"] - 0.3} for r in a]  # same units, shifted
    res = paired_difference(a, b, lambda r: r["v"], n_boot=200, seed=0)
    assert res["n_units"] == 120 and res["n_worlds"] == 3
    assert res["mean_diff"] == pytest.approx(0.3)
    assert res["ci"][0] > 0.25 and res["prob_superiority"] == 1.0
    null = paired_difference(a, a, lambda r: r["v"], n_boot=100)
    assert null["mean_diff"] == 0.0 and null["prob_superiority"] == 0.5
    assert paired_difference(a, [], lambda r: r["v"])["n_units"] == 0


def test_estimate_has_a_finite_ci_over_all_levels():
    recs = _recs("m", 0.5)
    p, lo, hi = estimate(recs, mean_of(lambda r: r["v"]), n_boot=200)
    assert lo < p < hi and np.isfinite(lo)


def test_vectorized_draws_match_the_reference_bootstrap():
    from collections import Counter

    from twist.evaluation.statistical_analysis import bootstrap_draws, column_estimates
    from twist.metrics.aggregation import hierarchical_bootstrap

    rng = np.random.default_rng(0)
    recs = [
        {
            "world_id": f"w{w}",
            "architecture": a,
            "unit": str(u),
            "explainer_seed": s,
            "v": float(rng.random() < 0.2 + 0.15 * w + 0.1 * (a == "t")),
        }
        for w in range(5)
        for a in ("g", "t")
        for u in range(40)
        for s in (1, 2, 3)
    ]
    draws = bootstrap_draws(recs, n_boot=400, seed=1)
    keys = Counter(
        (recs[i]["world_id"], recs[i]["architecture"], recs[i]["unit"]) for i in draws[0]
    )
    assert all(c % 3 == 0 for c in keys.values())  # each drawn unit: 3 resampled seed records
    est = column_estimates({"v": (np.array([r["v"] for r in recs]), "mean")}, draws)["v"]
    ref = hierarchical_bootstrap(
        recs,
        mean_of(lambda r: r["v"]),
        ("world_id", "architecture", "unit", "explainer_seed"),
        400,
        seed=1,
    )
    assert est[0] == pytest.approx(ref[0])
    assert est[1] == pytest.approx(ref[1], abs=0.08) and est[2] == pytest.approx(ref[2], abs=0.08)
    assert len(bootstrap_draws([], n_boot=3)) == 3


def test_blocked_difference_for_unpaired_populations():
    rng = np.random.default_rng(3)
    a = [
        {"world_id": f"w{w}", "unit": f"a{u}", "v": float(rng.random() < 0.8)}
        for w in range(3)
        for u in range(60)
    ]
    b = [
        {"world_id": f"w{w}", "unit": f"b{u}", "v": float(rng.random() < 0.3)}
        for w in range(3)
        for u in range(60)
    ]
    stat = mean_of(lambda r: r["v"])
    p, lo, hi = blocked_difference(a, b, stat, n_boot=200)
    assert 0.3 < lo < p < hi
    assert np.isnan(blocked_difference(a, [], stat)[0])


def test_pareto_front():
    pts = {
        "cheap_bad": (1.0, 0.2),
        "costly_good": (10.0, 0.9),
        "dominated": (10.0, 0.5),
        "mid": (5.0, 0.6),
        "nan": (float("nan"), 1.0),
    }
    assert pareto_front(pts) == ["cheap_bad", "costly_good", "mid"]


# ------------------------------------------------------------------------------- validity
def _vrec(pair, variant, unit, **kw):
    r = {
        "pair": pair,
        "variant": variant,
        "world_id": "w",
        "architecture": "-",
        "unit": str(unit),
        "explainer_seed": 0,
        "explainer": f"{pair}:{variant}",
        "reading": "native",
        "scorable": True,
        "valid": True,
        "world_replay": 1,
        "target": 1,
        "l1": 1.0,
        "replay_error_pearl": 0.0,
        "feasible": True,
        "costs": {"magnitude": 1.0},
        "ood_proposal": 0.1,
        "causal_path_consistency": 1.0,
    }
    r.update(kw)
    return r


def _synthetic_validity_records(n=30):
    rng = np.random.default_rng(0)
    recs = []
    for u in range(n):
        recs += [
            _vrec("A", "consistent", u),
            _vrec("A", "jittered", u, replay_error_pearl=0.01 + 0.001 * rng.random()),
        ]
        recs += [_vrec("B", "label-channel", u), _vrec("B", "distractor", u, world_replay=0)]
        recs += [_vrec("C", "within", u), _vrec("C", "beyond", u, feasible=False)]
        recs += [_vrec("D", "action", u), _vrec("D", "scaled", u, costs={"magnitude": 1.5})]
        recs += [
            _vrec("E", "aligned", u, ood_proposal=0.1 + 0.001 * rng.normal()),
            _vrec(
                "E",
                "retroactive",
                u,
                ood_proposal=0.1 + 0.001 * rng.normal(),
                causal_path_consistency=0.4,
            ),
        ]
        base = {
            "world_id": "w",
            "architecture": "-",
            "unit": str(u),
            "explainer_seed": 0,
            "scorable": True,
            "target": 1,
            "t_label": 19,
        }
        recs += [
            {
                **base,
                "check": "D1",
                "explainer": "Pearl",
                "reading": "native",
                "replay_error_pearl": 0.0,
                "action_timesteps": 1,
            },
            {
                **base,
                "check": "D1",
                "explainer": "Full",
                "reading": "native",
                "replay_error_pearl": 0.0,
                "action_timesteps": 20,
            },
            {
                **base,
                "check": "D2",
                "explainer": "PW",
                "reading": "native",
                "ood_proposal": 0.1,
                "ood_factual": 0.1,
                "replay_error_pearl": 0.01,
            },
            {
                **base,
                "check": "D3",
                "explainer": "W",
                "reading": "first_change_atomic",
                "valid": True,
                "feasible": True,
                "world_replay": 0,
            },
            {
                **base,
                "check": "D4",
                "explainer": "U",
                "reading": "native",
                "h_proposal": 0,
                "world_replay": 0,
            },
        ]
    return recs


def test_validity_checks_pass_on_well_formed_pairs():
    from twist.evaluation.validity import validity_checks

    res = validity_checks(_synthetic_validity_records(), n_boot=100)
    assert all(p["passed"] for p in res["pairs"]), res["pairs"]
    assert all(d["passed"] for d in res["discriminant"]), res["discriminant"]
    assert res["passed"]


def test_validity_checks_fail_when_the_held_property_differs_or_no_separation():
    from twist.evaluation.validity import validity_checks

    recs = _synthetic_validity_records()
    for r in recs:
        if r.get("pair") == "B" and r["variant"] == "distractor":
            r["l1"] = 3.0  # proximity no longer held equal
        if r.get("pair") == "C" and r["variant"] == "beyond":
            r["feasible"] = True  # feasibility no longer separates
    res = validity_checks(recs, n_boot=100)
    by = {p["pair"]: p["passed"] for p in res["pairs"]}
    assert not by["B"] and not by["C"] and by["A"] and not res["passed"]


@pytest.mark.smoke
def test_mini_validity_report(tmp_path, monkeypatch):
    from twist.evaluation.validity import ValidityConfig, run_validity
    from twist.worlds.suite import SuiteConfig, build_suite

    monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    suite = build_suite(
        SuiteConfig(
            name="s8",
            output_dir=str(tmp_path / "suites"),
            families=["linear"],
            worlds_per_split={"dev": 1},
            k=4,
            T=20,
            n_samples=300,
        )
    )
    cfg = ValidityConfig(
        name="val8",
        output_dir=str(tmp_path / "val"),
        suite=str(suite),
        worlds=["linear_dev_000"],
        classifier={"architecture": "logreg"},
        n_factuals=8,
        recourse={"steps": 40, "t0_fracs": [0.5, 0.9]},
        n_boot=50,
    )
    path = run_validity(cfg)
    text = path.read_text()
    for s in ("Construct validity", "Discriminant validity", "| C |", "| D |"):
        assert s in text
    import json

    res = json.loads(path.with_suffix(".json").read_text())
    by = {p["pair"]: p for p in res["pairs"]}
    # replay error identical, feasibility differs; same success, scaled cost: hold by construction
    assert by["C"]["passed"] and by["D"]["passed"]


@pytest.mark.smoke
def test_mini_study_report(tmp_path, monkeypatch):
    import json

    from twist.evaluation.study import StudyConfig, run_study
    from twist.worlds.suite import SuiteConfig, build_suite

    monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    suite = build_suite(
        SuiteConfig(
            name="s8b",
            output_dir=str(tmp_path / "suites"),
            families=["linear", "nonlinear"],
            worlds_per_split={"dev": 1},
            k=4,
            T=20,
            n_samples=300,
        )
    )
    rec = {"steps": 10, "t0_fracs": [0.5]}
    cfg = StudyConfig.model_validate(
        {
            "name": "study8",
            "output_dir": str(tmp_path / "study"),
            "suite": str(suite),
            "worlds": ["linear_dev_000", "nonlinear_dev_000"],
            "classifiers": [{"architecture": "logreg"}],
            "explainers": [
                {"name": "NUN", "regime": "R1"},
                {"name": "WachterReference", "regime": "R0", "params": {"steps": 20}},
                {"name": "PearlSCMRecourse", "regime": "R5", "params": rec},
                {"name": "FullScheduleOracle", "regime": "R5"},
                {"name": "NoActionAbstain", "regime": "R0"},
                {"name": "AdversarialTrajectoryEdit", "regime": "R0", "params": {"steps": 10}},
                {"name": "PlausibleButMechanismWrongEdit", "regime": "R5", "params": rec},
                {
                    "name": "OracleActionRecourse",
                    "regime": "R6",
                    "params": {**rec, "label": "OracleActionRecourse"},
                },
            ],
            "explainer_seeds": [1, 2],
            "n_factuals": 6,
            "n_boot": 50,
            "reference": "NUN",
            "evaluation": {
                "readings": ["first_change_atomic", "full_changed_schedule"],
                "compute_dtw": False,
                "constraints": [{"name": "amplitude_bounds", "max_abs": 3.0}],
            },
        }
    )
    path = run_study(cfg)
    text = path.read_text()
    for s in (
        "Primary metric table",
        "Paired method comparisons",
        "Pareto frontier",
        "Draft replication",
        "Failure-mode taxonomy",
        "conclusions_are_paired | **PASS**",
    ):
        assert s in text, s
    res = json.loads(path.with_suffix(".json").read_text())
    assert {"NUN", "WachterReference"} <= set(res["table"])
    assert res["paired"][0]["n_units"] > 0  # same flip candidates for every method
    pos = {c["control"]: c for c in res["controls"] if c["kind"] == "positive"}
    assert pos["FullScheduleOracle"]["passed"]
    neg = {c["control"]: c for c in res["controls"] if c["kind"] == "negative"}
    assert neg["NoActionAbstain"]["passed"] and neg["FullScheduleOracle"]["passed"]
