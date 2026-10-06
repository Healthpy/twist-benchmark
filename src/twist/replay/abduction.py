"""Exact abduction ``η_t = (x_t − F*(Pa(x_t))) / s(Pa(x_t))`` (draft Eq. 2; ``s ≡ 1`` unless the
world's innovations are heteroscedastic)."""

from __future__ import annotations

import numpy as np

from twist.utils.validation import as_batch, check_trajectory
from twist.worlds.base_scm import TemporalSCM


def abduct_noise(factual: np.ndarray, world: TemporalSCM) -> np.ndarray:
    """Recover factual innovations. Roots (``t < L``) get ``ε_t = x_t``.

    Accepts ``(T, k)`` or ``(B, T, k)`` and returns the same shape.
    """
    check_trajectory(factual, world.T, world.k)
    x, single = as_batch(factual)
    means = np.nan_to_num(world.one_step_means(x), nan=0.0)
    eps = (x - means) / world.one_step_scales(x)
    return eps[0] if single else eps


def structural_residual(
    factual: np.ndarray, proposal: np.ndarray, world: TemporalSCM
) -> np.ndarray:
    """Shift the mechanism would need at each entry to produce ``proposal``.

    ``r_t = x̂_t − (F*(x̂_{t-L:t}) + ε_t)`` with ``ε`` abducted from the factual (roots: ``x̂_t − x_t``).
    Teacher forcing on ``x̂`` makes this exact and non-recursive: replaying the soft intervention
    ``shift = r`` reproduces ``x̂``, and ``|r| > tol`` marks the entries a ``do()`` must impose.
    """
    xf, single = as_batch(np.asarray(factual, float))
    xh, _ = as_batch(np.asarray(proposal, float))
    eps = abduct_noise(xf, world)
    r = xh - world.one_step_prediction(xh, eps)
    return r[0] if single else r
