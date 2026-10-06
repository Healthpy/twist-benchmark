"""Week 3: new structural controls, certificate rendering and sanity-check logic."""

import numpy as np
import pytest

from twist.config import EvaluationConfig
from twist.evaluation.protocol import evaluate_explanations, summarize
from twist.evaluation.sanity import CHECKS, evaluate_checks
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import build_explainer
from twist.metrics.conventional import OODScorer
from twist.metrics.mechanism import replay_error
from twist.replay import replay_pearl
from twist.reporting.certificates import certificate_markdown, make_certificate


@pytest.fixture(scope="module")
def setup(linear_world, tiny_classifier):
    world, x, _ = linear_world
    idx = np.flatnonzero(tiny_classifier.predict(x) == 0)[:4]
    return world, tiny_classifier, x, x[idx], [str(i) for i in idx], np.ones(len(idx), int)


def _ctx(setup, regime):
    world, clf, x, *_ = setup
    return build_context(regime, clf, world, train_x=x[:200])


def test_full_schedule_oracle(setup):
    world, _, _, xs, ids, tg = setup
    with pytest.raises(PermissionError):
        build_explainer("FullScheduleOracle").explain(xs, ids, tg, _ctx(setup, "R0"))
    ex = build_explainer("FullScheduleOracle").explain(xs, ids, tg, _ctx(setup, "R5"))
    for e, x in zip(ex, xs, strict=True):
        sched = e.action.action.to_schedule(world.T, world.k)
        assert e.native_action and sched.n_timesteps == world.T
        np.testing.assert_allclose(
            replay_pearl(x, e.action, world), e.proposed_trajectory, atol=1e-12
        )
        assert world.label(e.proposed_trajectory) == 1  # oracle donor has world label = target


def test_adversarial_edit_is_bounded_and_trajectory_only(setup):
    _, _, _, xs, ids, tg = setup
    ex = build_explainer("AdversarialTrajectoryEdit", epsilons=[0.1, 0.5], steps=20).explain(
        xs, ids, tg, _ctx(setup, "R0")
    )
    std = xs.reshape(-1, xs.shape[-1]).std(0)
    for e, x in zip(ex, xs, strict=True):
        assert e.action is None and not e.declares_actions
        eps = e.selection_objective["epsilon_selected"] or 0.5
        assert np.all(np.abs(e.proposed_trajectory - x) <= eps * std + 1e-7)


def test_plausible_but_wrong_declares_action_it_does_not_generate(setup):
    world, _, _, xs, ids, tg = setup
    ex = build_explainer("PlausibleButMechanismWrongEdit", steps=20, t0_fracs=[0.5]).explain(
        xs, ids, tg, _ctx(setup, "R5")
    )
    for e, x in zip(ex, xs, strict=True):
        assert e.native_action and e.mechanism.startswith("misspecified:")
        t0 = e.action.action.t
        np.testing.assert_allclose(e.proposed_trajectory[:t0], x[:t0], atol=1e-10)  # prefix kept
        true_replay = replay_pearl(x, e.action, world)
        assert replay_error(e.proposed_trajectory[None], true_replay[None])[0] > 1e-6


def test_week3_record_fields_and_certificate(setup):
    world, clf, x, xs, ids, tg = setup
    ex = build_explainer("PearlSCMRecourse", steps=10, t0_fracs=[0.5]).explain(
        xs, ids, tg, _ctx(setup, "R5")
    )
    iqr = np.ones(world.k)
    recs = evaluate_explanations(
        world, clf, xs, ex, EvaluationConfig(), iqr=iqr, ood=OODScorer(x[:200])
    )
    r = recs[0]
    for key in (
        "dtw",
        "margin_change",
        "ood_proposal",
        "soft_faith_replay",
        "effect_corr",
        "support_overlap",
        "causal_path_consistency",
        "replay_hash",
        "_replay",
    ):
        assert key in r
    assert r["soft_faith_replay"] == pytest.approx(1.0) and r["causal_path_consistency"] == 1.0
    assert r["retroactive_fraction"] == 0.0
    manifest = {"run_id": "t", "config_hash": "c", "git_commit": "g"}
    md = certificate_markdown(make_certificate(r, world, manifest))
    for q in range(1, 8):
        assert f"## {q}." in md
    row = summarize(recs)[0]
    assert row["provenance_completeness"] == 1.0 and row["failure_rate"] >= 0.0


def test_sanity_checks_fail_closed_on_missing_rows():
    out = evaluate_checks([], [], 40, {"passed": True})
    assert len(out) == len(CHECKS)
    assert not any(c["passed"] for c in out if c["id"] != "audit")
    assert all("missing row" in c["note"] for c in out if c["id"] not in ("audit", "pearl_D1"))
