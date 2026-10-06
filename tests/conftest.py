"""Shared fixtures: small worlds of every family and a tiny trained classifier."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twist.worlds import make_linear_var, make_nonlinear_scm, make_spring_system

torch.set_num_threads(1)


def _fit(world, n=300, seed=1):
    x, eps = world.simulate(n, np.random.default_rng(seed))
    world.label.fit(x)
    return world, x, eps


@pytest.fixture(scope="session")
def linear_world():
    return _fit(
        make_linear_var(
            "lin_test", 3, k=4, T=20, density=0.4, spectral_radius=0.95, rescale="exact"
        )
    )


@pytest.fixture(scope="session")
def nonlinear_world():
    return _fit(make_nonlinear_scm("nl_test", 4, k=4, T=20, density=0.4))


@pytest.fixture(scope="session")
def spring_world():
    return _fit(make_spring_system("spring_test", 5, n_particles=3, T=20, n_exogenous=1))


@pytest.fixture(scope="session", params=["linear", "nonlinear", "spring"])
def any_world(request, linear_world, nonlinear_world, spring_world):
    return {"linear": linear_world, "nonlinear": nonlinear_world, "spring": spring_world}[
        request.param
    ]


@pytest.fixture(scope="session")
def tiny_classifier(linear_world):
    from twist.classifiers.train import train_classifier

    world, x, _ = linear_world
    y = world.label(x)
    return train_classifier(x[:240], y[:240], x[240:], y[240:], hidden=8, epochs=3, seed=7)


CONTROL_SPEC = {
    "actionable": [1, 2, 3],  # channel 0 (the label channel) is immutable
    "lower": -5.0,
    "upper": 5.0,
    "controls": {
        "u_drive": {
            "loadings": [0.0, 1.0, 0.5, 0.0],
            "delay": 1,
            "lower": -2.0,
            "upper": 2.0,
            "unit": "a.u.",
            "description": "drives x1 and x2 one step later",
        },
    },
}


@pytest.fixture(scope="session")
def control_world():
    """Same mechanism and shape as ``linear_world`` (so ``tiny_classifier`` applies), plus an
    action space with an immutable channel, value bounds and one high-level control."""
    world = make_linear_var(
        "lin_ctrl", 3, k=4, T=20, density=0.4, spectral_radius=0.95, rescale="exact"
    )
    world.configure_action_space(**CONTROL_SPEC)
    return _fit(world)
