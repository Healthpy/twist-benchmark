import numpy as np
import pytest

from twist.config import EvaluationConfig
from twist.evaluation.protocol import evaluate_explanations, summarize
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import build_explainer
from twist.metrics.mechanism import cf_faith
from twist.replay import replay_pearl


@pytest.fixture(scope="module")
def setup(linear_world, tiny_classifier):
    world, x, _ = linear_world
    pred = tiny_classifier.predict(x)
    idx = np.flatnonzero(pred == 0)[:4]
    return world, tiny_classifier, x[idx], [str(i) for i in idx], np.ones(len(idx), int)


def test_regime_gating_blocks_oracle_access(setup):
    world, clf, xs, ids, tg = setup
    ctx = build_context("R0", clf, world)
    assert ctx.world is None and ctx.graph is None
    with pytest.raises(PermissionError):
        build_explainer("PearlSCMRecourse", steps=5).explain(xs, ids, tg, ctx)


def test_pearl_control_proposal_is_its_own_replay(setup):
    world, clf, xs, ids, tg = setup
    ex = build_explainer("PearlSCMRecourse", steps=30, t0_fracs=[0.5]).explain(
        xs, ids, tg, build_context("R5", clf, world)
    )
    for e, x in zip(ex, xs, strict=True):
        assert e.native_action and e.declares_actions
        np.testing.assert_allclose(
            replay_pearl(x, e.action, world), e.proposed_trajectory, atol=1e-10
        )


def test_noiseless_control_semantics(setup):
    world, clf, xs, ids, tg = setup
    ex = build_explainer("NoiselessSCMRecourse", steps=30, t0_fracs=[0.5]).explain(
        xs, ids, tg, build_context("R5", clf, world)
    )
    props = np.stack([e.proposed_trajectory for e in ex])
    f = cf_faith(world, xs, props, "noiseless")
    assert np.all(f["hard"][f["scorable"]] == 1.0)


def test_abstain_control_shrinks_denominator(setup):
    world, clf, xs, ids, tg = setup
    ex = build_explainer("NoActionAbstain").explain(xs, ids, tg, build_context("R0", clf, world))
    rows = summarize(evaluate_explanations(world, clf, xs, ex, EvaluationConfig()))
    assert rows[0]["n_scorable"] == 0 and rows[0]["n_abstain"] == len(xs)


def test_trajectory_baseline_gets_registered_readings(setup):
    world, clf, xs, ids, tg = setup
    ex = build_explainer("WachterReference", steps=20).explain(
        xs, ids, tg, build_context("R0", clf, world)
    )
    recs = evaluate_explanations(world, clf, xs, ex, EvaluationConfig())
    assert {r["reading"] for r in recs} == {"first_change_atomic", "full_changed_schedule"}
    assert all(not r["native_action"] for r in recs)
    full = [r for r in recs if r["reading"] == "full_changed_schedule" and r["scorable"]]
    assert all(r["replay_error_pearl"] < 1e-3 for r in full)
