"""Week 3: hand-constructed cases for every new metric."""

import numpy as np
import pytest

from twist.actions import PointIntervention
from twist.metrics import conventional as conv
from twist.metrics import mechanism as mech
from twist.replay import replay_pearl
from twist.worlds import make_linear_var, make_nonlinear_scm
from twist.worlds.shifts import perturb_mechanism


def test_logit_margin():
    z = np.array([[2.0, -1.0], [0.5, 1.5]])
    np.testing.assert_allclose(conv.logit_margin(z, 1), [-3.0, 1.0])
    np.testing.assert_allclose(conv.logit_margin(z, np.array([0, 0])), [3.0, -1.0])


def test_dtw_zero_for_identical_and_below_l1_for_time_shift():
    t = np.linspace(0, 4 * np.pi, 30)
    x = np.sin(t)[None, :, None]
    shifted = np.roll(x, 2, axis=1)
    assert conv.dtw_distance(x, x)[0] == 0.0
    l1 = np.abs(shifted - x).mean()
    assert conv.dtw_distance(x, shifted)[0] < 0.5 * l1
    assert conv.dtw_distance(x, shifted, window=0)[0] == pytest.approx(l1)


def test_interval_sparsity():
    x = np.zeros((1, 10, 2))
    xh = x.copy()
    xh[0, 1:3, 0] = 1
    xh[0, 6, 0] = 1
    xh[0, 2:4, 1] = 1
    s = conv.interval_sparsity(x, xh)
    assert s["channel_segments"][0] == 3 and s["time_segments"][0] == 2


def test_diversity():
    a = np.zeros((5, 2))
    assert np.isnan(conv.diversity(a[None]))
    assert conv.diversity(np.stack([a, a])) == 0.0
    assert conv.diversity(np.stack([a, a + 1, a + 3])) == pytest.approx((1 + 3 + 2) / 3)


def test_ood_scorer_ranks_outliers_lower():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, 10, 2))
    scorer = conv.OODScorer(train)
    assert scorer.score(train[:20]).mean() > scorer.score(train[:20] + 6.0).mean()


def test_normalized_replay_error_and_soft_faith():
    x = np.zeros((2, 4, 2))
    xr = x.copy()
    xr[1] += 2.0
    e = mech.normalized_replay_error(x, xr, np.array([1.0, 4.0]))
    np.testing.assert_allclose(e, [0.0, (2 + 0.5) / 2], rtol=1e-6)
    s = mech.soft_faith(e)
    assert s[0] == 1.0 and 0 < s[1] < 1


def test_effect_correlation_overlap_retroactivity_prefix():
    x = np.zeros((10, 2))
    xh = x.copy()
    xh[4:, 0] = np.arange(6)
    assert mech.effect_profile_correlation(x, xh, xh) == pytest.approx(1.0)
    assert mech.effect_profile_correlation(x, xh, 2 * x - xh) == pytest.approx(-1.0)
    assert np.isnan(mech.effect_profile_correlation(x, x, xh))
    xr = x.copy()
    xr[5:7, 0] = 1.0
    # proposal changes t=5..9 (t=4 value is 0), replay changes t=5,6 → Jaccard 2/5
    assert mech.support_overlap(x, xh, xr) == pytest.approx(2 / 5)
    assert mech.support_overlap(x, x, x) == 1.0
    assert mech.retroactivity(x, xh, 5) == 0.0
    assert mech.retroactivity(x, xh, 7) == pytest.approx(2 / 5)
    assert mech.prefix_preservation(x, xh, 5) == 1.0
    assert mech.prefix_preservation(x, xh, 7) == pytest.approx(1 - 2 / 14)
    assert np.isnan(mech.prefix_preservation(x, xh, 0))


@pytest.fixture
def chain():
    """x0 → x1 at lag 1, nothing else."""
    w = make_linear_var("chain", 0, k=3, T=12, density=0.0)
    w.coef = np.zeros((1, 3, 3))
    w.coef[0, 1, 0] = 0.8
    w.graph = np.abs(w.coef.transpose(1, 2, 0)) > 0
    x, _ = w.simulate(2, np.random.default_rng(0))
    return w, x


def test_causal_reachability_follows_time_unrolled_graph(chain):
    w, _ = chain
    active = np.zeros((12, 3), bool)
    active[3, 0] = True
    reach = mech.causal_reachability(w, active)
    assert sorted(map(tuple, np.argwhere(reach))) == [(3, 0), (4, 1)]


def test_causal_path_consistency(chain):
    w, x = chain
    a = PointIntervention(3, [0], [x[0, 3, 0] + 2.0])
    active = a.to_schedule(12, 3).active
    good = replay_pearl(x[0], a, w)
    assert mech.causal_path_consistency(w, x[0], good, active) == 1.0
    bad = good.copy()
    bad[8, 2] += 1.0  # x2 is not a descendant of x0
    assert mech.causal_path_consistency(w, x[0], bad, active) == pytest.approx(2 / 3)
    assert np.isnan(mech.causal_path_consistency(w, x[0], x[0], active))


def test_nonlinear_dependency_graph_includes_decay_self_loops():
    w = make_nonlinear_scm("nl", 0, k=4, T=10, density=0.0)
    assert not w.graph.any()
    assert w.dependency_graph()[np.arange(4), np.arange(4), 0].all()


@pytest.mark.parametrize("family", ["linear", "nonlinear", "spring"])
def test_perturb_mechanism_keeps_graph_changes_strengths(any_world, family):
    world, x, _ = any_world
    wrong = perturb_mechanism(world, rel_scale=0.5, seed=1)
    np.testing.assert_array_equal(wrong.graph, world.graph)
    assert wrong.label.to_dict() == world.label.to_dict()
    past = x[:10, : world.L][:, ::-1]
    assert np.abs(wrong.structural_mean(past) - world.structural_mean(past)).max() > 1e-3
    # abducting under the wrong mechanism still reproduces the factual without an action
    np.testing.assert_allclose(replay_pearl(x[:3], None, wrong), x[:3], atol=1e-10)
