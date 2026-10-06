"""Week 2: native action forms in the structural controls and invalid-action rejection."""

import numpy as np
import pytest

from twist.config import EvaluationConfig
from twist.evaluation.protocol import evaluate_explanations, summarize
from twist.evaluation.result_schema import Explanation
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import build_explainer
from twist.replay import replay_noiseless, replay_pearl

KINDS = {"point": "point", "interval": "interval", "pulse": "pulse", "control": "control"}


@pytest.fixture(scope="module")
def setup(control_world, tiny_classifier):
    world, x, _ = control_world
    idx = np.flatnonzero(tiny_classifier.predict(x) == 0)[:3]
    return world, tiny_classifier, x[idx], [str(i) for i in idx], np.ones(len(idx), int)


@pytest.mark.parametrize("form", list(KINDS))
@pytest.mark.parametrize(
    "name, replay", [("PearlSCMRecourse", replay_pearl), ("NoiselessSCMRecourse", replay_noiseless)]
)
def test_native_forms_are_their_own_replay_and_valid(setup, form, name, replay):
    world, clf, xs, ids, tg = setup
    ex = build_explainer(name, steps=15, t0_fracs=[0.5], action_form=form, duration=3).explain(
        xs, ids, tg, build_context("R5", clf, world)
    )
    space = world.action_space()
    for e, x in zip(ex, xs, strict=True):
        assert e.native_action and e.action.action.kind == KINDS[form]
        assert e.action.action.validate(space) == []  # projection keeps actions valid
        assert 0 not in e.action.action.variables  # immutable channel never intervened
        np.testing.assert_allclose(replay(x, e.action, world), e.proposed_trajectory, atol=1e-10)
    if form != "point":
        assert ex[0].explainer == f"{name}[{form}]"


def test_invalid_actions_are_rejected_with_reports(setup):
    world, clf, xs, ids, _ = setup
    proposals = xs.copy()
    proposals[:, 8, 0] += 1.0  # edits the immutable label channel
    expls = [Explanation("hand", fid, 1, p) for fid, p in zip(ids, proposals, strict=True)]
    recs = evaluate_explanations(
        world, clf, xs, expls, EvaluationConfig(readings=["first_change_atomic"])
    )
    for r in recs:
        assert r["invalid_action"] and not r["scorable"]
        assert any("non-actionable" in v and "x0" in v for v in r["action_violations"])
        assert "invalid_action" in r["provenance"]["known_failure_modes"]
    row = summarize(recs)[0]
    assert row["n_invalid_action"] == len(xs) and row["n_abstain"] == 0 and row["n_scorable"] == 0


def test_all_registered_readings_evaluate(setup):
    world, clf, xs, ids, _ = setup
    proposals = xs.copy()
    proposals[:, 8:, 1:] += 0.5
    expls = [Explanation("hand", fid, 1, p) for fid, p in zip(ids, proposals, strict=True)]
    readings = [
        "first_change_atomic",
        "first_changed_segment",
        "full_changed_schedule",
        "sparse_segmentation",
        "change_point_pulse_fit",
    ]
    recs = evaluate_explanations(world, clf, xs, expls, EvaluationConfig(readings=readings))
    assert {r["reading"] for r in recs} == set(readings)
    for r in recs:
        if r["reading"] == "change_point_pulse_fit":
            # editing x1..x3 changes x0's parents; holding x0 at its factual value needs a shift
            # on x0, which is immutable here — the reading is rejected, not silently scored
            assert r["invalid_action"] and "x0" in r["action_violations"][0]
        else:
            assert r["scorable"] and r["action"]["fit_error"] is not None
