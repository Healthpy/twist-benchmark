"""Week 7: shifts, perturbation, uncertainty, stability, tolerance sensitivity, report."""

import json

import numpy as np
import pytest

from twist.actions import PointIntervention
from twist.config import EvaluationConfig, ExperimentConfig
from twist.evaluation.protocol import evaluate_explanations
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import build_explainer
from twist.metrics.aggregation import hierarchical_bootstrap
from twist.metrics.robustness import ensemble_outcomes, mechanism_ensemble, stability_pairs
from twist.metrics.sensitivity import ranking_stability, tolerance_profile
from twist.replay import replay_pearl
from twist.worlds import make_linear_var, make_statespace
from twist.worlds.shifts import SHIFT_LEVELS, apply_shift, perturb_units, shift_id


@pytest.fixture(scope="module")
def lin():
    w = make_linear_var(
        "w7",
        1,
        k=4,
        T=24,
        density=0.4,
        spectral_radius=0.9,
        rescale="exact",
        ensure_route=True,
        distractor=True,
    )
    x, eta = w.simulate(300, np.random.default_rng(0))
    w.label.fit(x)
    return w, x, {"state": x, "eta": eta}


@pytest.mark.parametrize("level,kind", [(lv, k) for lv, ks in SHIFT_LEVELS.items() for k in ks])
def test_every_shift_yields_a_valid_world(lin, level, kind):
    w, x, _ = lin
    amount = 0.7 if kind == "site" else 0.3
    s = apply_shift(w, {"level": level, "kind": kind, "amount": amount}, x)
    xs, exo = s.simulate_units(50, np.random.default_rng(1))
    assert np.all(np.isfinite(xs)) and s.world_id.endswith(
        shift_id({"level": level, "kind": kind, "amount": amount})
    )
    assert s.assumptions.get("exact_abduction", True) == (level != "observation")
    y = s.outcome(exo["state"])
    assert set(np.unique(y)) <= {0, 1}


def test_decision_shifts_change_the_rule(lin):
    w, x, _ = lin
    up = apply_shift(w, {"level": "decision", "kind": "threshold", "amount": 0.5}, x)
    assert up.label.threshold > w.label.threshold and up.outcome(x).mean() < w.outcome(x).mean()
    site = apply_shift(w, {"level": "decision", "kind": "site", "amount": 0.5}, x)
    assert site.label.t_label == 12 and abs(site.outcome(x).mean() - 0.5) < 0.05
    with pytest.raises(ValueError):
        apply_shift(w, {"level": "decision", "kind": "teleport"}, x)


def test_perturb_units_is_coherent(lin):
    w, x, exo = lin
    sub = {k: v[:5] for k, v in exo.items()}
    x0, _ = perturb_units(w, sub, 0.0, np.random.default_rng(0))
    np.testing.assert_allclose(x0, x[:5], atol=1e-10)
    x1, _ = perturb_units(w, sub, 0.05, np.random.default_rng(0))
    assert 0 < np.abs(x1 - x[:5]).mean() < 0.05
    np.testing.assert_allclose(replay_pearl(x1, None, w), x1, atol=1e-10)  # a valid world unit
    ss = make_statespace("s7", 0, base_params=dict(k=4, T=20), observation=dict(noise_scale=0.1))
    _, exo2 = ss.simulate_units(4, np.random.default_rng(0))
    xp, ep = perturb_units(ss, exo2, 0.05, np.random.default_rng(0))
    np.testing.assert_allclose(
        ss.observation.observe(ep["state"], ep["obs_noise"], ep["obs_mask"]), xp
    )


def test_hierarchical_bootstrap():
    rng = np.random.default_rng(0)
    recs = [
        {"world_id": w, "unit": f"{w}/{u}", "v": float(rng.random() < 0.3 + 0.2 * w)}
        for w in range(3)
        for u in range(30)
        for _ in range(2)
    ]  # two records per unit
    stat = lambda rs: float(np.mean([r["v"] for r in rs]))  # noqa: E731
    p, lo, hi = hierarchical_bootstrap(recs, stat, n_boot=200, seed=1)
    assert lo < p < hi
    assert hierarchical_bootstrap(recs, stat, n_boot=200, seed=1) == (p, lo, hi)
    assert np.isnan(hierarchical_bootstrap([], lambda rs: float("nan"))[1])


def test_stability_pairs_hand_case():
    def rec(fid, t, var, out, valid=True, mag=1.0):
        return {
            "explainer": "m",
            "reading": "native",
            "explainer_seed": 1,
            "factual_id": fid,
            "valid": valid,
            "scorable": True,
            "action_t_first": t,
            "world_replay": out,
            "action": {"target_variables": var},
            "costs": {"magnitude": mag},
        }

    base = [rec("w/test/1", 5, [0, 1], 1), rec("w/test/2", 5, [0], 1)]
    pert = [rec("w__pert/test/1", 5, [0, 1], 1), rec("w__pert/test/2", 7, [1], 0, mag=3.0)]
    s = stability_pairs(base, pert)[("m", "native")]
    assert s["onset"] == 0.5 and s["variables"] == 0.5 and s["outcome"] == 0.5
    assert s["cost"] == pytest.approx((1 + 0.5) / 2) and s["n_pairs"] == 2


def test_ensemble_contains_the_world(lin):
    w, x, _ = lin
    ens = mechanism_ensemble(w, size=4, rel_scale=0.25)
    plans = [PointIntervention(12, [1], [x[i, 12, 1] + 1.0]) for i in range(5)]
    y = ensemble_outcomes(ens, x[:5], plans)
    np.testing.assert_array_equal(y[0], w.label(replay_pearl(x[:5], plans, w)))
    assert y.shape == (4, 5)


def test_tolerance_profile_and_ranking(lin):
    w, x, _ = lin
    pearl = replay_pearl(
        x[:6], [PointIntervention(12, [1], [x[i, 12, 1] + 1.0]) for i in range(6)], w
    )
    noisy = x[:6] + np.random.default_rng(0).normal(0, 0.3, x[:6].shape)
    tiny = x[:6].copy()
    tiny[:, 15:, 0] += 1e-4 * x.std()
    prof = tolerance_profile(w, x[:6], {"pearl": pearl, "noisy": noisy, "tiny": tiny})
    assert prof["pearl"][1e-3]["D"] == 1.0 and prof["pearl"][1e-3]["cf_faith"] == 1.0
    assert prof["noisy"][1e-3]["D"] == w.T
    assert prof["tiny"][1e-5]["D"] > 0 and prof["tiny"][1e-2]["D"] == 0  # tolerance-dependent
    rs = ranking_stability(prof)
    assert rs[1e-3] == pytest.approx(1.0)


def test_reported_and_interventional_target_probabilities(control_world, tiny_classifier):
    world, x, _ = control_world
    idx = np.flatnonzero(tiny_classifier.predict(x[200:]) == 0)[:3] + 200
    xs = x[idx]
    ctx = build_context("R5", tiny_classifier, world)
    ex = build_explainer(
        "PearlSCMRecourse", steps=15, t0_fracs=[0.5], uncertainty_samples=8
    ).explain(xs, ["a", "b", "c"], np.ones(3, int), ctx)
    for e in ex:
        assert 0.0 <= e.uncertainty["p_target_model"] <= 1.0
    recs = evaluate_explanations(
        world,
        tiny_classifier,
        xs,
        ex,
        EvaluationConfig(interventional_samples=8),
        exogenous={"state": xs},
    )
    for r in recs:
        assert 0.0 <= r["p_world_interventional"] <= 1.0 and r["p_target_reported"] is not None


def _mini_exp(tmp_path, **extra):
    return ExperimentConfig.model_validate(
        {
            "name": "mini7",
            "output_dir": str(tmp_path),
            "world": {
                "family": "linear",
                "world_id": "m7",
                "seed": 0,
                "n_samples": 300,
                "params": {
                    "k": 4,
                    "T": 20,
                    "density": 0.4,
                    "spectral_radius": 0.95,
                    "rescale": "exact",
                    "ensure_route": True,
                    "distractor": True,
                },
            },
            "classifier": {"architecture": "logreg"},
            "explainers": [
                {"name": "NUN", "regime": "R1"},
                {
                    "name": "PearlSCMRecourse",
                    "regime": "R5",
                    "params": {"steps": 10, "t0_fracs": [0.5]},
                },
            ],
            "n_factuals": 4,
            "evaluation": {"readings": ["first_change_atomic"], "compute_dtw": False},
            **extra,
        }
    )


@pytest.mark.smoke
def test_shifted_and_perturbed_runs_are_auditable_and_paired(tmp_path):
    from twist.evaluation.audit import audit_run
    from twist.evaluation.paired_runner import run_paired_experiment

    base = run_paired_experiment(_mini_exp(tmp_path))
    shifted = run_paired_experiment(
        _mini_exp(
            tmp_path, name="mini7s", shift={"level": "mechanism", "kind": "delay", "amount": 0.5}
        )
    )
    pert = run_paired_experiment(_mini_exp(tmp_path, name="mini7p", perturb={"scale": 0.05}))
    retrain = run_paired_experiment(
        _mini_exp(tmp_path, name="mini7r", shift={"level": "decision", "kind": "retrain"})
    )
    for d in (base, shifted, pert, retrain):
        assert audit_run(d)["passed"], d
    info = json.loads((retrain / "run_info.json").read_text())
    assert info["model_dir"].endswith("__retrained")

    def ids(d):
        return sorted(
            r["factual_id"].rsplit("/", 1)[1] for r in json.loads((d / "records.json").read_text())
        )

    assert ids(base) == ids(pert)  # the same units, perturbed


@pytest.mark.smoke
def test_mini_robustness_report(tmp_path, monkeypatch):
    from twist.evaluation.robustness import (
        RobustnessConfig,
        build_robustness_report,
        run_robustness,
    )
    from twist.worlds.suite import SuiteConfig, build_suite

    monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    suite = build_suite(
        SuiteConfig(
            name="s7",
            output_dir=str(tmp_path / "suites"),
            families=["linear"],
            worlds_per_split={"dev": 1},
            k=4,
            T=20,
            n_samples=300,
        )
    )
    cfg = RobustnessConfig.model_validate(
        {
            "name": "rob7",
            "output_dir": str(tmp_path / "rob"),
            "suite": str(suite),
            "worlds": ["linear_dev_000"],
            "classifiers": [{"architecture": "logreg"}],
            "explainers": [
                {"name": "NUN", "regime": "R1"},
                {"name": "WachterReference", "regime": "R0", "params": {"steps": 20}},
                {
                    "name": "PearlSCMRecourse",
                    "regime": "R5",
                    "params": {"steps": 10, "t0_fracs": [0.5], "uncertainty_samples": 4},
                },
            ],
            "explainer_seeds": [1],
            "n_factuals": 5,
            "shift_factuals": 4,
            "n_boot": 50,
            "ensemble_size": 3,
            "shifts": [
                {"level": "innovation", "kind": "uniform"},
                {"level": "mechanism", "kind": "coefficients", "amount": 0.5},
                {"level": "decision", "kind": "retrain"},
            ],
            "evaluation": {
                "readings": ["first_change_atomic"],
                "compute_dtw": False,
                "interventional_samples": 4,
            },
        }
    )
    runs = run_robustness(cfg)
    assert len(runs) == 5  # base, perturb, 3 shifts
    text = build_robustness_report(cfg, runs).read_text()
    for s in (
        "every_headline_metric_has_uncertainty",
        "every_method_has_tolerance_profile",
        "brittleness_identified",
        "Method robustness cards",
        "p_robust",
    ):
        assert s in text
    assert "every_headline_metric_has_uncertainty | **PASS**" in text
    assert "every_method_has_tolerance_profile | **PASS**" in text
