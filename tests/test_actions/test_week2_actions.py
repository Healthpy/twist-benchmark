"""Week 2: high-level control actions, new trajectory adapters, and action validation."""

import numpy as np
import pytest

from twist.actions import HighLevelControl, ParametricPulse, PointIntervention, ScheduleIntervention
from twist.actions.trajectory_adapters import (
    ADAPTERS,
    REQUIRES_WORLD,
    change_point_pulse_fit,
    first_change_atomic,
    full_changed_schedule,
    runs,
    sparse_segmentation,
)
from twist.metrics.mechanism import replay_error
from twist.replay import replay_pearl
from twist.worlds import TemporalSCM, make_linear_var


def test_control_compiles_through_loadings_with_delay(control_world):
    world, _, _ = control_world
    space = world.action_space()
    s = HighLevelControl("u_drive", 4, 6, [1.0, -0.5]).compile(space)
    assert not s.mask.any()
    np.testing.assert_allclose(s.shift[5], [0, 1.0, 0.5, 0])
    np.testing.assert_allclose(s.shift[6], [0, -0.5, -0.25, 0])
    assert s.active_timesteps.tolist() == [5, 6] and s.first_time == 5
    with pytest.raises(NotImplementedError):
        HighLevelControl("u_drive", 4, 6, [1.0, -0.5]).to_schedule(world.T, world.k)


def test_control_validation_reports(control_world):
    world, _, _ = control_world
    space = world.action_space()
    assert HighLevelControl("u_drive", 4, 6, [1.0, -0.5]).validate(space) == []
    assert "unknown control" in HighLevelControl("u_x", 4, 6, [1, 1]).validate(space)[0]
    msgs = " ".join(HighLevelControl("u_drive", 4, 6, [3.0, 0.0]).validate(space))
    assert "above upper bound 2.0" in msgs
    assert "after the horizon" in " ".join(
        HighLevelControl("u_drive", 19, 20, [1.0]).validate(space)
    )
    assert "cannot compile" in HighLevelControl("u_drive", 4, 6, [1.0]).validate(space)[0]


def test_action_space_validation_immutable_and_bounds(control_world):
    world, _, _ = control_world
    space = world.action_space()
    v = PointIntervention(3, [0, 1], [0.0, 9.0]).validate(space)
    assert any("non-actionable" in m and "x0" in m for m in v)
    assert any("upper bound" in m for m in v)


def test_world_action_space_survives_save_load(control_world, tmp_path):
    world, _, _ = control_world
    world.save(tmp_path)
    loaded = TemporalSCM.load(tmp_path)
    assert loaded.action_space().to_dict() == world.action_space().to_dict()


def test_control_replay_matches_equivalent_soft_shift(control_world):
    world, x, _ = control_world
    act = HighLevelControl("u_drive", 4, 7, [1.0, 0.5, -1.0])
    xr = replay_pearl(x[0], act, world)
    xs = replay_pearl(x[0], act.compile(world.action_space()), world)
    np.testing.assert_allclose(xr, xs)
    np.testing.assert_allclose(xr[:5], x[0, :5], atol=1e-12)  # delay 1: effect starts at t=5
    assert np.abs(xr[5:] - x[0, 5:]).max() > 0.1


def test_runs_helper():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert runs(np.zeros(3, bool)) == []


def test_sparse_segmentation_hand_case():
    x = np.zeros((10, 2))
    xh = x.copy()
    xh[2:5, 0] = 1.0
    xh[7, 0] = 2.0
    xh[3:6, 1] = -1.0
    plan = sparse_segmentation(x, xh)
    assert not plan.native and plan.adapter == "sparse_segmentation"
    assert plan.action.to_dict()["entries"] == [[2, [0], [1.0]], [3, [1], [-1.0]], [7, [0], [2.0]]]
    assert sparse_segmentation(x, x) is None


@pytest.fixture
def chain_world():
    """x0 → x1 at lag 1 only, so each atomic action's effect is a short, separate segment."""
    w = make_linear_var("chain", 0, k=2, T=20, density=0.0)
    w.coef = np.array([[[0.0, 0.0], [0.8, 0.0]]])
    w.graph = np.abs(w.coef.transpose(1, 2, 0)) > 0
    x, _ = w.simulate(5, np.random.default_rng(0))
    w.label.fit(x)
    return w, x


def test_sparse_segmentation_reproduces_multi_action_proposal(chain_world):
    world, x = chain_world
    two_actions = ScheduleIntervention([(3, [0], [x[0, 3, 0] + 2]), (10, [0], [x[0, 10, 0] - 2])])
    proposal = replay_pearl(x[0], two_actions, world)
    sparse = sparse_segmentation(x[0], proposal)
    first = first_change_atomic(x[0], proposal)
    full = full_changed_schedule(x[0], proposal)
    err = {
        name: replay_error(proposal[None], replay_pearl(x[0], plan, world)[None])[0]
        for name, plan in {"sparse": sparse, "first": first, "full": full}.items()
    }
    assert err["sparse"] < 1e-12 and err["full"] < 1e-3 and err["first"] > 0.05
    n = {
        k: p.action.to_schedule(world.T, world.k).n_timesteps
        for k, p in {"sparse": sparse, "first": first, "full": full}.items()
    }
    assert n["first"] <= n["sparse"] <= n["full"]


@pytest.mark.parametrize("shape", ["rect", "raised_cosine"])
def test_pulse_fit_recovers_planted_pulse(control_world, shape):
    world, x, _ = control_world
    planted = ParametricPulse(5, 4, [1, 3], [0.8, -0.5], shape)
    proposal = replay_pearl(x[0], planted, world)
    plan = change_point_pulse_fit(x[0], proposal, world=world)
    a = plan.action
    assert (a.start, a.duration, a.targets, a.shape) == (5, 4, [1, 3], shape)
    np.testing.assert_allclose(a.amplitude, [0.8, -0.5], atol=1e-9)
    assert replay_error(proposal[None], replay_pearl(x[0], plan, world)[None])[0] < 1e-10


def test_pulse_fit_needs_world_and_handles_no_edit(control_world):
    world, x, _ = control_world
    assert "change_point_pulse_fit" in REQUIRES_WORLD
    with pytest.raises(ValueError):
        change_point_pulse_fit(x[0], x[0] + 1.0)
    assert change_point_pulse_fit(x[0], x[0], world=world) is None


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_adapter_is_non_native_and_serializable(control_world, name):
    world, x, _ = control_world
    proposal = x[0] + np.random.default_rng(1).normal(0, 0.3, x[0].shape)
    plan = ADAPTERS[name](x[0], proposal, 1e-3, world=world)
    assert plan is not None and not plan.native and plan.adapter == name
    assert type(plan).from_dict(plan.to_dict()).to_dict() == plan.to_dict()
