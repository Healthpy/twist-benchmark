"""Week 4: noise factors, label families, generator factors, regime and state-space worlds."""

import numpy as np
import pytest

from twist.actions import ActionPlan, PointIntervention, ScheduleIntervention
from twist.replay import abduct_noise, replay_pearl
from twist.worlds import (
    ExposureLabel,
    InnovationProcess,
    LabelFunction,
    MultiOutputLabel,
    ObservationModel,
    OnsetLabel,
    TemporalSCM,
    ThresholdLabel,
    make_linear_var,
    make_regime_scm,
    make_spring_system,
    make_statespace,
)
from twist.worlds.linear_var import companion_spectral_radius

RNG = np.random.default_rng


def test_correlated_innovations():
    eta = InnovationProcess("gaussian", 1.0, correlation=0.5).sample(RNG(0), (20000, 3))
    c = np.corrcoef(eta.T)
    assert abs(c[0, 1] - 0.5) < 0.03 and abs(eta.std() - 1.0) < 0.03


def test_heteroscedastic_world_abducts_and_replays_exactly():
    w = make_linear_var(
        "h", 0, k=4, T=30, density=0.4, noise=InnovationProcess("laplace", 0.1, heteroscedastic=1.0)
    )
    x, eta = w.simulate(50, RNG(1))
    np.testing.assert_allclose(abduct_noise(x, w)[:, 1:], eta[:, 1:], atol=1e-12)
    np.testing.assert_allclose(replay_pearl(x, None, w), x, atol=1e-12)
    s = w.one_step_scales(x)
    assert s[:, 1:].min() >= 1.0 and s[:, 1:].max() <= 2.0


@pytest.mark.parametrize(
    "label, expected_site",
    [
        (ThresholdLabel(0, -1), 29),
        (ThresholdLabel(0, 12), 12),
        (ExposureLabel(0, -8, -1), 29),
        (OnsetLabel(0, 15, -1), 29),
        (MultiOutputLabel([ThresholdLabel(0, -1), ExposureLabel(1, -6, -2)], "and"), 29),
    ],
)
def test_label_families(label, expected_site):
    w = make_linear_var("l", 1, k=3, T=30, density=0.5)
    x, _ = w.simulate(2000, RNG(0))
    label.fit(x)
    y = label(x)
    assert 0.3 <= y.mean() <= 0.7
    assert label.site(30) == expected_site
    assert label.support(30, 3).any()
    assert LabelFunction.from_dict(label.to_dict()).to_dict() == label.to_dict()
    np.testing.assert_array_equal(LabelFunction.from_dict(label.to_dict())(x), y)


def test_onset_time_and_exposure_hand_case():
    x = np.zeros((1, 10, 1))
    x[0, 6, 0] = 5.0
    on = OnsetLabel(0, 2, -1, threshold=1.0)
    assert on(x)[0] == 1 and on.onset_time(x)[0] == 6
    ex = ExposureLabel(0, -5, -1, threshold=0.5)
    assert ex.raw(x)[0] == pytest.approx(1.0)
    assert ExposureLabel(0, -5, -1, weights="ramp").raw(x)[0] == pytest.approx(5 * 2 / 15)


def test_linear_factors_route_distractor_and_oscillation():
    w = make_linear_var(
        "f",
        3,
        k=6,
        L=3,
        density=0.3,
        spectral_radius=0.9,
        rescale="exact",
        ensure_route=True,
        distractor=True,
        oscillatory=True,
    )
    r, d = w.roles["route_source"], w.roles["distractor"]
    assert w.graph[0, r, 0] and abs(w.coef[0, 0, r]) > 0
    assert not w.graph[[i for i in range(6) if i != d], d, :].any()  # distractor causes nothing
    assert w.graph[d, r, 0]  # …but is driven by a cause of the label
    assert companion_spectral_radius(w.coef) == pytest.approx(0.9, abs=1e-6)
    eig = np.linalg.eigvals(w.coef[0])
    assert np.abs(eig.imag).max() > 1e-3


def test_draft_defaults_unchanged():
    """Default arguments must reproduce the draft generator exactly (toy world regression)."""
    w = make_linear_var("t", 0, k=5, T=40, density=0.3, spectral_radius=0.95, rescale="exact")
    np.testing.assert_allclose(w.coef[0][0], [0.0, 1.7881, 1.7292, 0.6673, 0.0], atol=1e-4)


def test_spring_damping_and_distractor():
    undamped = make_spring_system("s", 0, n_particles=4)
    damped = make_spring_system(
        "s", 0, n_particles=4, damping=0.5, anchor=0.2, distractor=True, ensure_route=True
    )
    assert companion_spectral_radius(undamped.coef) == pytest.approx(1.0)
    assert companion_spectral_radius(damped.coef) < 0.999  # tether + damping: contractive
    weak = make_spring_system("s", 0, n_particles=4, damping=0.5)
    assert companion_spectral_radius(weak.coef) == pytest.approx(1.0)  # damping alone is not
    assert np.all(damped.coupling[:, 3] == 0) and damped.coupling[3, 1] > 0
    assert damped.coupling[0, 1] > 0


@pytest.mark.parametrize("driver", ["state", "exogenous"])
def test_regime_world(driver):
    w = make_regime_scm("r", 2, k=6, T=40, density=0.3, driver=driver)
    x, eta = w.simulate(200, RNG(0))
    np.testing.assert_allclose(abduct_noise(x, w)[:, 1:], eta[:, 1:], atol=1e-12)
    frac = w.regimes(x)[:, 1:].mean()
    assert 0.1 < frac < 0.9
    d = w.roles["distractor"]
    assert w.regime_weights[d] == 0  # the distractor never switches regimes
    G = w.dependency_graph()
    drivers = np.flatnonzero(w.regime_weights)
    assert G[0, drivers[0], 0] or np.allclose(w.coef[0][0, 0], w.coef[1][0, 0])
    if driver == "exogenous":
        assert 5 not in w.action_space().actionable and w.roles["regime_driver"] == 5
    # an action can switch the regime path (state driver) and replay stays consistent
    a = PointIntervention(10, [1], [x[0, 10, 1] + 3.0])
    xr = replay_pearl(x[0], a, w)
    np.testing.assert_allclose(xr[:10], x[0, :10], atol=1e-12)


def test_observation_model_locf_and_inverse():
    obs = ObservationModel(transform="saturating", saturation=2.0)
    z = np.array([[[0.5], [1.0], [1.5], [0.2]]])
    seen = np.array([[[True], [False], [True], [False]]])
    y = obs.observe(z, np.zeros_like(z), seen)
    expected = 2 * np.tanh(np.array([0.5, 0.5, 1.5, 1.5]) / 2)
    np.testing.assert_allclose(y[0, :, 0], expected)
    np.testing.assert_allclose(obs.h_inv(obs.h(z)), z)
    assert ObservationModel().is_identity and not obs.is_identity


def test_statespace_world_oracle_replay_and_flags(tmp_path):
    w = make_statespace(
        "ss",
        1,
        base_family="linear",
        base_params=dict(k=5, T=30, density=0.3, ensure_route=True, distractor=True),
        observation=dict(noise_scale=0.1, missing="mcar", missing_rate=0.2),
        latent_confounder=True,
    )
    x, exo = w.simulate_units(100, RNG(0))
    w.label.fit(exo["state"])
    assert w.k == 5 and w.k_state == 6 and w.obs_index == [0, 1, 2, 3, 4]
    assert not w.assumptions["exact_abduction"] and w.assumptions["confounding"]
    assert w.roles["latent_confounder"] == 5 and 5 not in w.action_space().actionable
    xr, zr = w.replay_units({k: v[:4] for k, v in exo.items()}, None)
    np.testing.assert_array_equal(xr, x[:4])
    np.testing.assert_array_equal(zr, exo["state"][:4])
    # outcomes are labels of the true state, not of the observations
    np.testing.assert_array_equal(w.outcome(exo["state"]), w.label(exo["state"]))
    w.save(tmp_path)
    assert TemporalSCM.load(tmp_path).card() == w.card()


def test_to_state_plan_maps_channels_through_inverse():
    w = make_statespace(
        "ss",
        0,
        base_params=dict(k=4, T=20),
        observation=dict(hidden=[1], transform="saturating", saturation=1.0),
    )
    assert w.obs_index == [0, 2, 3]
    plan = ActionPlan(ScheduleIntervention([(5, [1, 2], [0.5, -0.5])]), native=False, semantics="r")
    st = w.to_state_plan(plan).action
    assert st.entries[0][1] == [2, 3]
    np.testing.assert_allclose(st.entries[0][2], np.arctanh([0.5, -0.5]))
