import numpy as np
import pytest
import torch

from twist.worlds import TemporalSCM, make_world
from twist.worlds.linear_var import companion_spectral_radius


def test_simulation_shapes_and_finite(any_world):
    world, x, eps = any_world
    assert x.shape == eps.shape == (300, world.T, world.k)
    assert np.all(np.isfinite(x))


def test_generation_is_deterministic():
    a = make_world("nonlinear", "w", 11, k=4, T=15)
    b = make_world("nonlinear", "w", 11, k=4, T=15)
    xa, _ = a.simulate(10, np.random.default_rng(0))
    xb, _ = b.simulate(10, np.random.default_rng(0))
    np.testing.assert_array_equal(xa, xb)


def test_different_seeds_give_different_mechanisms():
    a, b = make_world("linear", "a", 1, k=6), make_world("linear", "b", 2, k=6)
    assert not np.allclose(a.coef, b.coef)


@pytest.mark.parametrize("rescale", ["cap", "exact"])
def test_linear_world_is_stable(rescale):
    w = make_world("linear", "w", 0, k=8, L=2, density=0.5, spectral_radius=0.9, rescale=rescale)
    assert companion_spectral_radius(w.coef) <= 0.9 + 1e-9


def test_no_lag1_self_loops_in_sampled_graph():
    w = make_world("linear", "w", 0, k=8, density=1.0)
    assert not np.any(w.graph[np.arange(8), np.arange(8), 0])


def test_spring_exogenous_particles_have_no_incoming_coupling(spring_world):
    world, _, _ = spring_world
    assert np.all(world.coupling[-1] == 0)


def test_label_is_median_split(any_world):
    world, x, _ = any_world
    assert abs(world.label(x).mean() - 0.5) < 0.05


def test_torch_and_numpy_mechanisms_agree(any_world):
    world, x, _ = any_world
    past = x[:7, 3 : 3 + world.L][:, ::-1].copy()
    np_out = world.structural_mean(past)
    t_out = world.structural_mean(torch.as_tensor(past)).numpy()
    np.testing.assert_allclose(np_out, t_out, atol=1e-12)


def test_save_load_roundtrip(any_world, tmp_path):
    world, x, _ = any_world
    world.save(tmp_path)
    loaded = TemporalSCM.load(tmp_path)
    assert loaded.card() == world.card()
    past = x[:5, 4 : 4 + world.L][:, ::-1]
    np.testing.assert_allclose(loaded.structural_mean(past), world.structural_mean(past))
    np.testing.assert_array_equal(loaded.label(x), world.label(x))
