"""Week 6: mechanism estimators, quality metrics and controlled misspecification."""

import numpy as np
import pytest
import torch

from twist.actions import PointIntervention
from twist.mechanisms.estimators import (
    estimate_mechanism,
    fit_lasso_var,
    fit_masked_linear,
    fit_neural,
    lagged_design,
)
from twist.mechanisms.misspecification import KINDS, misspecify
from twist.mechanisms.quality import graph_recovery, mechanism_quality
from twist.replay import abduct_noise, replay, replay_pearl
from twist.worlds import make_linear_var, make_nonlinear_scm, make_statespace


@pytest.fixture(scope="module")
def lin():
    w = make_linear_var(
        "lin6",
        2,
        k=4,
        L=2,
        T=30,
        density=0.4,
        spectral_radius=0.9,
        rescale="exact",
        ensure_route=True,
        distractor=True,
    )
    x, eta = w.simulate(400, np.random.default_rng(0))
    w.label.fit(x)
    return w, x, {"state": x, "eta": eta}


def test_lagged_design_matches_mechanism_layout(lin):
    w, x, _ = lin
    X, _ = lagged_design(x[:3], w.L)
    W = w.coef.transpose(1, 0, 2).reshape(w.k, w.L * w.k)  # inverse of _coef_from_flat
    np.testing.assert_allclose(
        X @ W.T, w.one_step_means(x[:3])[:, w.L :].reshape(-1, w.k), atol=1e-12
    )


def test_masked_ols_recovers_coefficients_and_replays_exactly(lin):
    w, x, exo = lin
    est = fit_masked_linear(x, w.L, w.graph, w)
    assert np.abs(est.coef - w.coef).max() < 0.05
    q = mechanism_quality(est, w, x[300:], {k: v[300:] for k, v in exo.items()})
    assert q["graph_f1"] == 1.0 and q["replay_error"] < 0.02


def test_lasso_recovers_graph(lin):
    w, x, _ = lin
    est = fit_lasso_var(x, w.L, w)
    assert graph_recovery(est.graph, w.graph)["f1"] >= 0.8


def test_neural_mechanism_fits_and_is_differentiable(lin):
    w, x, _ = lin
    est = fit_neural(x, w.L, w, hidden=16, epochs=8, seed=0)
    q_est = est.one_step_means(x[300:])[:, w.L :]
    err = np.abs(x[300:, w.L :] - q_est).mean()
    oracle = np.abs(x[300:, w.L :] - w.one_step_means(x[300:])[:, w.L :]).mean()
    assert err < 1.5 * oracle
    e = torch.as_tensor(abduct_noise(x[:2], est))
    mask = np.zeros(x[:2].shape, bool)
    mask[:, 10, 0] = True
    val = torch.zeros(x[:2].shape, dtype=torch.float64, requires_grad=True)
    replay(est, e, mask, val, torch.zeros(x[:2].shape, dtype=torch.float64)).sum().backward()
    assert val.grad[:, 10, 0].abs().sum() > 0


def test_estimate_mechanism_dispatch(lin):
    w, x, _ = lin
    assert estimate_mechanism({"estimator": "masked"}, x, w, w.graph).world_id.endswith(
        "masked-ols"
    )
    with pytest.raises(ValueError):
        estimate_mechanism({"estimator": "masked"}, x, w, None)
    with pytest.raises(ValueError):
        estimate_mechanism({"estimator": "magic"}, x, w, None)


@pytest.mark.parametrize("kind", [k for k in KINDS if k != "observation"])
def test_misspecification_is_exact_at_zero_and_wrong_above(lin, kind):
    w, x, exo = lin
    sub = {k: v[300:340] for k, v in exo.items()}
    q0 = mechanism_quality(misspecify(w, kind, 0.0), w, x[300:340], sub)
    q1 = mechanism_quality(misspecify(w, kind, 1.0, seed=3), w, x[300:340], sub)
    assert q0["replay_error"] < 1e-9
    assert q1["replay_error"] > 1e-3


def test_misspecification_errors_are_explicit():
    nl = make_nonlinear_scm("nl6", 0, k=4, T=20)
    with pytest.raises(NotImplementedError):
        misspecify(nl, "delete_edges", 0.5)
    with pytest.raises(ValueError):
        misspecify(nl, "teleport", 0.5)
    with pytest.raises(ValueError):
        misspecify(nl, "observation", 0.5)


def test_observation_misspecification_on_partial_world():
    w = make_statespace(
        "ss6",
        0,
        base_params=dict(k=4, T=20),
        observation=dict(transform="saturating", saturation=0.5),
    )
    wrong = misspecify(w, "observation", 1.0)
    assert wrong.k == w.k and wrong.coef.shape == w.state.coef.shape


def test_graph_recovery_hand_case():
    t = np.zeros((3, 3, 1), bool)
    t[0, 1] = t[1, 2] = True
    e = np.zeros((3, 3, 1), bool)
    e[0, 1] = e[2, 0] = True
    r = graph_recovery(e, t)
    assert (r["precision"], r["recall"], r["shd"]) == (0.5, 0.5, 2)


def test_point_replay_of_estimate_matches_truth_for_exact_copy(lin):
    w, x, _ = lin
    copy = misspecify(w, "coefficients", 0.0)
    a = PointIntervention(12, [1], [x[0, 12, 1] + 1.0])
    np.testing.assert_allclose(replay_pearl(x[0], a, copy), replay_pearl(x[0], a, w), atol=1e-12)
