import numpy as np

from twist.metrics.aggregation import bootstrap_ci
from twist.metrics.conventional import first_change, proximity_lp, sparsity
from twist.metrics.mechanism import cf_faith, do_complexity
from twist.metrics.outcome import abc_decomposition
from twist.metrics.robustness import shift_validity_retention
from twist.metrics.traceability import REQUIRED_PROVENANCE_FIELDS, provenance_completeness


def test_abc_identity_and_denominators():
    rng = np.random.default_rng(0)
    n = 50
    a, b, c = (rng.integers(0, 2, n) for _ in range(3))
    s = rng.random(n) > 0.2
    out = abc_decomposition(a, b, c, 1, s)
    assert np.isclose(out["delta_total"], out["delta_traj"] + out["delta_outcome"])
    assert out["n_scorable"] == s.sum() and out["n_abstain"] == n - s.sum()
    assert np.isclose(out["proposal_model_success"], a[s].mean())


def test_abstentions_are_not_scored_as_success():
    out = abc_decomposition(np.ones(3), np.ones(3), np.ones(3), 1, np.zeros(3, bool))
    assert np.isnan(out["proposal_model_success"]) and out["frac_abstain"] == 1.0


def test_safe_world_and_action_grounded_success():
    out = abc_decomposition(
        np.ones(4),
        np.ones(4),
        np.array([1, 1, 0, 1]),
        1,
        np.ones(4, bool),
        feasible=np.array([1, 0, 1, 1], bool),
        replay_faithful=np.array([1, 1, 1, 0], bool),
    )
    assert out["safe_world_success"] == 0.5 and out["action_grounded_success"] == 0.5


def test_cf_faith_unscorable_cases_are_nan(linear_world):
    world, x, _ = linear_world
    xs = x[:3].copy()
    xh = xs.copy()
    xh[1, -1, 0] += 1.0  # terminal-only edit: no suffix
    f = cf_faith(world, xs, xh, "pearl")
    assert np.isnan(f["hard"][0]) and np.isnan(f["hard"][1]) and not f["scorable"].any()


def test_retroactive_edit_scores_zero(linear_world):
    world, x, _ = linear_world
    xs = x[:1]
    xh = xs.copy()
    xh[0, 2, 0] += 1.0
    f = cf_faith(world, xs, xh, "pearl", t_action=np.array([5]))
    assert f["hard"][0] == 0.0


def test_do_complexity_extremes(linear_world):
    world, x, _ = linear_world
    xs = x[:2]
    assert np.all(do_complexity(world, xs, xs) == 0)
    dense = xs + np.random.default_rng(0).normal(0, 1, xs.shape)
    assert np.all(do_complexity(world, xs, dense) == world.T)


def test_conventional_hand_case():
    x = np.zeros((1, 4, 2))
    xh = x.copy()
    xh[0, 2, 1] = 3.0
    t0, has = first_change(x, xh)
    assert t0[0] == 2 and has[0]
    assert proximity_lp(x, xh, 1)[0] == 3.0
    sp = sparsity(x, xh)
    assert sp["unchanged_entries"][0] == 7 / 8 and sp["unchanged_channels"][0] == 0.5
    assert first_change(x, x)[0][0] == 3  # draft convention: t0 = T-1 when nothing changed


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci(np.r_[np.random.default_rng(0).normal(1, 1, 200), np.nan])
    assert lo < m < hi


def test_shift_vr_suppressed_below_threshold():
    assert np.isnan(shift_validity_retention(0.2, 0.3))
    assert shift_validity_retention(0.5, 0.25) == 0.5


def test_provenance_completeness():
    rec = dict.fromkeys(REQUIRED_PROVENANCE_FIELDS)
    assert provenance_completeness(rec)["completeness"] == 1.0
    rec.pop("uncertainty")
    assert provenance_completeness(rec)["missing_fields"] == ["uncertainty"]
