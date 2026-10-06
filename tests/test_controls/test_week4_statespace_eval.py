"""Week 4: evaluating explanations on a partially observed world (oracle replay)."""

import numpy as np

from twist.config import EvaluationConfig
from twist.evaluation.protocol import evaluate_explanations, summarize
from twist.evaluation.result_schema import Explanation
from twist.worlds import make_statespace


def test_statespace_evaluation_uses_oracle_replay(tiny_classifier):
    w = make_statespace(
        "ss_eval",
        3,
        base_params=dict(k=4, T=20, density=0.4, ensure_route=True),
        observation=dict(noise_scale=0.1),
    )
    x, exo = w.simulate_units(40, np.random.default_rng(0))
    w.label.fit(exo["state"])
    xs, ex = x[:5], {k: v[:5] for k, v in exo.items()}
    props = xs.copy()
    props[:, 12:, 1] += 0.5
    expl = [Explanation("hand", str(i), 1, p) for i, p in enumerate(props)]
    readings = ["first_change_atomic", "full_changed_schedule", "change_point_pulse_fit"]
    recs = evaluate_explanations(
        w, tiny_classifier, xs, expl, EvaluationConfig(readings=readings), exogenous=ex
    )
    by = {}
    for r in recs:
        by.setdefault(r["reading"], []).append(r)
    for r in by["first_change_atomic"] + by["full_changed_schedule"]:
        assert r["scorable"] and np.isnan(r["cf_faith_pearl_hard"]) and np.isnan(r["D_pearl"])
        assert np.isfinite(r["replay_error_pearl"])
        assert r["action"]["semantics"].endswith("_via_observation")
    for r in by["change_point_pulse_fit"]:
        assert r["invalid_action"] and "exact abduction" in r["action_violations"][0]
    rows = {row["reading"]: row for row in summarize(recs)}
    assert np.isnan(rows["first_change_atomic"]["cf_faith_pearl"])
    assert rows["change_point_pulse_fit"]["n_invalid_action"] == 5
