"""Week 6: regime contexts, action-native recourse along the ladder, attribution, report."""

import numpy as np
import pytest

from twist.actions.feasibility import AmplitudeBounds, MinLeadTime
from twist.config import EvaluationConfig, ExperimentConfig
from twist.evaluation.protocol import attribute_outcome, evaluate_explanations
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import build_explainer
from twist.mechanisms.estimators import fit_neural
from twist.mechanisms.misspecification import misspecify
from twist.metrics.mechanism import replay_error
from twist.replay import replay_pearl
from twist.worlds import make_statespace
from twist.worlds.integrity import ancestors

FAST = {"steps": 25, "t0_fracs": [0.5], "duration": 2}


@pytest.fixture(scope="module")
def setup(control_world, tiny_classifier):
    world, x, _ = control_world
    idx = np.flatnonzero(tiny_classifier.predict(x[200:]) == 0)[:3] + 200
    return world, tiny_classifier, x, x[idx], [str(i) for i in idx], np.ones(len(idx), int)


def test_regime_contexts_expose_the_right_information(setup):
    world, clf, _x, *_ = setup
    r2 = build_context("R2", clf, world)
    assert r2.graph is not None and r2.outcome_channels == [0] and r2.mechanism is None
    with pytest.raises(ValueError):
        build_context("R4", clf, world)
    est = misspecify(world, "delay", 0.5)
    r4 = build_context("R4", clf, world, mechanism=est, mechanism_label="m")
    assert r4.mechanism is est and r4.world is None and r4.graph.shape == est.graph.shape
    r6 = build_context("R6", clf, world, constraints=[AmplitudeBounds(1.0)])
    assert r6.mechanism is world and r6.actionable == [1, 2, 3] and r6.constraints


def test_graph_recourse_imposes_on_ancestors_only(setup):
    world, clf, _, xs, ids, tg = setup
    ctx = build_context("R2", clf, world)
    ex = build_explainer("GraphActionRecourse", **FAST).explain(xs, ids, tg, ctx)
    allowed = ancestors(ctx.graph, [0]) | {0}
    for e, xf in zip(ex, xs, strict=True):
        changed = np.abs(e.proposed_trajectory - xf) > 0
        assert set(np.flatnonzero(changed.any(0))) <= allowed
        sched = e.action.action.to_schedule(world.T, world.k)
        np.testing.assert_array_equal(changed, sched.active & changed)  # nothing propagates
        assert e.action.semantics.endswith("@observed")


def test_estimated_recourse_with_exact_mechanism_is_its_own_replay(setup):
    world, clf, _, xs, ids, tg = setup
    ctx = build_context(
        "R4",
        clf,
        world,
        mechanism=misspecify(world, "coefficients", 0.0),
        mechanism_label="misspecified:coefficients-0",
    )
    for e, xf in zip(
        build_explainer("EstimatedActionRecourse", **FAST).explain(xs, ids, tg, ctx),
        xs,
        strict=True,
    ):
        true = replay_pearl(xf, e.action, world)
        assert replay_error(e.proposed_trajectory[None], true[None])[0] < 1e-10


def test_oracle_recourse_respects_constraints_and_uses_controls(setup):
    world, clf, _, xs, ids, tg = setup
    cons = [AmplitudeBounds(0.5), MinLeadTime(t_label=19, min_lead=10)]
    ctx = build_context("R6", clf, world, constraints=cons)
    ex = build_explainer("OracleActionRecourse", steps=25, t0_fracs=[0.5, 0.75]).explain(
        xs, ids, tg, ctx
    )
    for e in ex:
        a = e.action.action
        assert a.kind == "control" and e.explainer == "OracleActionRecourse[control]"
        assert a.start <= 9 and max(abs(v) for v in a.values) <= 0.5 / 1.0 + 1e-9
        assert a.validate(world.action_space()) == []


def test_attribution_categories():
    base = dict(has_proposal=True, invalid_action=False, scorable=True, target=1)
    assert (
        attribute_outcome({**base, "world_replay": 1, "h_proposal": 0, "h_replay": 0}) == "success"
    )
    assert (
        attribute_outcome({**base, "world_replay": 0, "h_proposal": 0, "h_replay": 0})
        == "optimization"
    )
    assert (
        attribute_outcome({**base, "world_replay": 0, "h_proposal": 1, "h_replay": 0})
        == "replay_mismatch"
    )
    assert (
        attribute_outcome({**base, "world_replay": 0, "h_proposal": 1, "h_replay": 1})
        == "outcome_disagreement"
    )
    assert (
        attribute_outcome(
            {**base, "scorable": False, "world_replay": 0, "h_proposal": 1, "h_replay": 1}
        )
        == "no_action"
    )
    assert (
        attribute_outcome(
            {**base, "has_proposal": False, "world_replay": 0, "h_proposal": 0, "h_replay": 0}
        )
        == "no_proposal"
    )


def test_estimated_recourse_on_partially_observed_world(tiny_classifier):
    w = make_statespace(
        "ss6e",
        4,
        base_params=dict(k=5, T=20, density=0.4, ensure_route=True),
        observation=dict(hidden=[3], noise_scale=0.05),
    )
    xo, exo = w.simulate_units(200, np.random.default_rng(0))
    w.label.fit(exo["state"])
    est = fit_neural(xo[:150], 1, w, hidden=8, epochs=3)
    ctx = build_context("R4", tiny_classifier, w, mechanism=est, mechanism_label="estimated:neural")
    xs, sub = xo[150:153], {k: v[150:153] for k, v in exo.items()}
    ex = build_explainer("EstimatedActionRecourse", **FAST).explain(
        xs, ["a", "b", "c"], np.ones(3, int), ctx
    )
    recs = evaluate_explanations(w, tiny_classifier, xs, ex, EvaluationConfig(), exogenous=sub)
    for r in recs:
        assert not r["invalid_action"], r["action_violations"]
        # observed channel 3 is state channel 4 (state channel 3 is hidden)
        targets = set(r["action"]["target_variables"])
        assert 3 not in targets
        assert r["attribution"] in (
            "success",
            "optimization",
            "replay_mismatch",
            "outcome_disagreement",
        )


@pytest.mark.smoke
def test_mini_ladder_report(tmp_path):
    from twist.evaluation.ladder_report import build_ladder_report
    from twist.evaluation.paired_runner import run_paired_experiment

    fast = {"steps": 15, "t0_fracs": [0.5], "duration": 2}
    cfg = ExperimentConfig.model_validate(
        {
            "name": "mini_ladder",
            "output_dir": str(tmp_path),
            "world": {
                "family": "linear",
                "world_id": "ml",
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
                {"name": "GraphActionRecourse", "regime": "R2", "params": fast},
                {
                    "name": "EstimatedActionRecourse",
                    "regime": "R3",
                    "params": {**fast, "label": "EAR R3"},
                    "mechanism": {"estimator": "masked"},
                },
                {
                    "name": "EstimatedActionRecourse",
                    "regime": "R4",
                    "params": {**fast, "label": "EAR m0"},
                    "mechanism": {"misspecify": {"kind": "delay", "level": 0.0}},
                },
                {
                    "name": "EstimatedActionRecourse",
                    "regime": "R4",
                    "params": {**fast, "label": "EAR m1"},
                    "mechanism": {"misspecify": {"kind": "delay", "level": 1.0}},
                },
                {"name": "OracleActionRecourse", "regime": "R6", "params": fast},
            ],
            "n_factuals": 4,
            "evaluation": {"readings": ["first_change_atomic"], "compute_dtw": False},
        }
    )
    run = run_paired_experiment(cfg)
    text = build_ladder_report("mini", [run], tmp_path / "report").read_text()
    assert "failures_attributed | **PASS**" in text and "misspecification_degrades" in text
    assert "| delay | 1 |" in text
