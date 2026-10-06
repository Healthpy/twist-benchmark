import numpy as np
import pytest
import torch

from twist.actions import PointIntervention
from twist.actions.trajectory_adapters import full_changed_schedule
from twist.metrics.mechanism import cf_faith, do_complexity, replay_error
from twist.replay import (
    abduct_noise,
    replay,
    replay_interventional_distribution,
    replay_noiseless,
    replay_pearl,
)
from twist.replay.pearl_replay import ScheduleBatch
from twist.utils.validation import TwistValidationError

TOL = 1e-10


def _atomic(world, x, t=8):
    return PointIntervention(t, [0, 1], (x[t, :2] + 1.0).tolist())


def test_abduction_recovers_true_innovations(any_world):
    world, x, eps = any_world
    e = abduct_noise(x, world)
    np.testing.assert_allclose(e[:, world.L :], eps[:, world.L :], atol=TOL)


def test_empty_action_replays_factual_exactly(any_world):
    world, x, _ = any_world
    np.testing.assert_allclose(replay_pearl(x[:10], None, world), x[:10], atol=TOL)


def test_pearl_replay_keeps_prefix_and_imposes_action(any_world):
    world, x, _ = any_world
    a = _atomic(world, x[0])
    xr = replay_pearl(x[0], a, world)
    np.testing.assert_array_equal(xr[:8], x[0, :8])
    np.testing.assert_allclose(xr[8, :2], x[0, 8, :2] + 1.0)


def test_pearl_atomic_control_is_faithful_with_unit_complexity(any_world):
    """Acceptance (Week 2/3): a one-action structural control has Pearl replay error ≈ 0, D = 1."""
    world, x, _ = any_world
    xs = x[:6]
    actions = [_atomic(world, xi) for xi in xs]
    proposals = replay_pearl(xs, actions, world)
    assert np.all(replay_error(proposals, replay_pearl(xs, actions, world)) < TOL)
    np.testing.assert_array_equal(do_complexity(world, xs, proposals, "pearl"), 1)
    f = cf_faith(world, xs, proposals, "pearl")
    np.testing.assert_array_equal(f["hard"], 1.0)


def test_noiseless_control_passes_rollout_and_fails_pearl(any_world):
    """Acceptance (Week 2): noiseless control passes noiseless rollout and fails Pearl replay."""
    world, x, _ = any_world
    xs = x[:6]
    actions = [_atomic(world, xi) for xi in xs]
    proposals = replay_noiseless(xs, actions, world)
    np.testing.assert_array_equal(cf_faith(world, xs, proposals, "noiseless")["hard"], 1.0)
    np.testing.assert_array_equal(cf_faith(world, xs, proposals, "pearl")["hard"], 0.0)
    assert np.all(do_complexity(world, xs, proposals, "pearl") > 1)
    np.testing.assert_array_equal(do_complexity(world, xs, proposals, "noiseless"), 1)


def test_full_schedule_reconstructs_proposal_with_high_complexity(any_world):
    """Acceptance (Week 2): full schedule reproduces the proposal but has high schedule complexity."""
    world, x, _ = any_world
    rng = np.random.default_rng(0)
    proposal = x[0] + rng.normal(0, 0.5, x[0].shape)
    plan = full_changed_schedule(x[0], proposal)
    xr = replay_pearl(x[0], plan, world)
    np.testing.assert_allclose(xr, proposal, atol=1e-3)
    assert plan.action.to_schedule(world.T, world.k).n_timesteps == world.T


def test_torch_replay_matches_numpy(any_world):
    world, x, _ = any_world
    xs = x[:4]
    sb = ScheduleBatch.from_actions([_atomic(world, xi) for xi in xs], 4, world.T, world.k)
    e = abduct_noise(xs, world)
    np_out = replay(world, e, sb.mask, sb.value, sb.shift)
    t_out = replay(
        world, torch.as_tensor(e), sb.mask, torch.as_tensor(sb.value), torch.as_tensor(sb.shift)
    ).numpy()
    np.testing.assert_allclose(np_out, t_out, atol=1e-10)


def test_interventional_distribution_keeps_prefix(linear_world):
    world, x, _ = linear_world
    samples = replay_interventional_distribution(
        x[0], _atomic(world, x[0]), world, 5, np.random.default_rng(0)
    )
    assert samples.shape == (5, world.T, world.k)
    np.testing.assert_allclose(
        samples[:, :9],
        np.broadcast_to(replay_pearl(x[0], _atomic(world, x[0]), world)[:9], (5, 9, world.k)),
        atol=TOL,
    )
    assert samples[:, 9:].std(axis=0).mean() > 0


def test_replay_rejects_bad_inputs(linear_world):
    world, x, _ = linear_world
    with pytest.raises(TwistValidationError):
        replay_pearl(x[0, :5], None, world)
    with pytest.raises(TwistValidationError):
        bad = x[0].copy()
        bad[3, 1] = np.nan
        replay_pearl(bad, None, world)
