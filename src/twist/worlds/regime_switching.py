"""Hybrid regime-switching SCM ``x_t = F_{r_t}(x_{t−L:t}) + s·η_t`` (plan §Week 4, task 2).

Two linear regimes share one graph but have independently drawn coefficients. The regime is a
deterministic function of the *observed* past, so abduction and unit-level replay stay exact:

* ``driver="state"``: ``r_t = 1{w · x_{t−1} > c}``. Actions can switch the regime.
* ``driver="exogenous"``: an extra observed channel ``s`` (last channel) follows its own AR(1)
  process with no parents, and ``r_t = 1{s_{t−1} > 0}``. ``s`` is exogenous to the system (like
  weather) and is not actionable. Hiding it with a partial observation model
  (``observation.hide``) gives a latent-regime challenge world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from twist.utils.backend import Array, as_like, einsum, where
from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import LabelFunction, ThresholdLabel
from twist.worlds.linear_var import (
    designate_route_and_distractor,
    rescale_to_radius,
    sample_lagged_graph,
)


@dataclass
class RegimeSwitchingSCM(TemporalSCM):
    family: ClassVar[str] = "regime"
    coef: np.ndarray = field(default_factory=lambda: np.zeros((2, 1, 1, 1)))  # (R, L, k, k)
    regime_weights: np.ndarray = field(default_factory=lambda: np.zeros(1))  # (k,)
    regime_threshold: float = 0.0

    def regime(self, past: Array) -> Array:
        """Boolean regime indicator ``r_t`` (True = regime 1) from the lag window ``(B, L, k)``."""
        return (
            einsum("bk,k->b", past[:, 0], as_like(self.regime_weights, past))
            > self.regime_threshold
        )

    def regimes(self, x: np.ndarray) -> np.ndarray:
        """Regime path ``(B, T)`` of trajectories; roots (``t < L``) are reported as regime 0."""
        B, T, _ = x.shape
        out = np.zeros((B, T), bool)
        for t in range(self.L, T):
            out[:, t] = self.regime(x[:, t - self.L : t][:, ::-1])
        return out

    def structural_mean(self, past: Array) -> Array:
        m0 = einsum("lij,blj->bi", as_like(self.coef[0], past), past)
        m1 = einsum("lij,blj->bi", as_like(self.coef[1], past), past)
        return where(self.regime(past)[:, None], m1, m0)

    def dependency_graph(self) -> np.ndarray:
        """Union of the regime graphs plus regime-driver → every channel whose dynamics switch."""
        g = (np.abs(self.coef) > 0).any(axis=0).transpose(1, 2, 0)
        switching = np.abs(self.coef[1] - self.coef[0]).sum(axis=(0, 2)) > 0  # rows that differ
        drivers = np.abs(self.regime_weights) > 0
        g[:, :, 0] |= switching[:, None] & drivers[None, :]
        return g | np.asarray(self.graph, bool)

    def params(self) -> dict[str, np.ndarray]:
        return {
            "coef": self.coef,
            "regime_weights": self.regime_weights,
            "regime_threshold": np.asarray(self.regime_threshold),
        }

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> RegimeSwitchingSCM:
        return cls(
            coef=arrays["coef"],
            regime_weights=arrays["regime_weights"],
            regime_threshold=float(arrays["regime_threshold"]),
            **cls._common_from_card(card, arrays),
        )


def make_regime_scm(
    world_id: str,
    seed: int,
    k: int = 10,
    L: int = 1,
    T: int = 100,
    density: float = 0.2,
    spectral_radius: float = 0.9,
    rescale: str = "cap",
    strength: float = 0.5,
    driver: str = "state",
    driver_persistence: float = 0.9,
    ensure_route: bool = True,
    distractor: bool = True,
    label_channel: int = 0,
    noise: InnovationProcess | None = None,
    label: LabelFunction | None = None,
    burn_in: int = 100,
) -> RegimeSwitchingSCM:
    if driver not in ("state", "exogenous"):
        raise ValueError(f"driver must be 'state' or 'exogenous', got {driver!r}")
    rng = np.random.default_rng(seed)
    k_sys = k - 1 if driver == "exogenous" else k
    g_sys = sample_lagged_graph(rng, k_sys, L, density)
    g_sys, r, d = designate_route_and_distractor(g_sys, label_channel, ensure_route, distractor)
    lag_scale = (1.0 / np.arange(1, L + 1))[:, None, None]
    coef = np.zeros((2, L, k, k))
    for regime in range(2):
        A = (
            rng.uniform(-strength, strength, (L, k_sys, k_sys))
            * lag_scale
            * g_sys.transpose(2, 0, 1)
        )
        if r is not None:
            A[0, label_channel, r] = rng.choice([-1, 1]) * strength * rng.uniform(0.5, 1.0)
        coef[regime, :, :k_sys, :k_sys] = rescale_to_radius(A, spectral_radius, rescale)
    g = np.zeros((k, k, L), bool)
    g[:k_sys, :k_sys] = g_sys
    if driver == "exogenous":
        s = k - 1
        coef[:, 0, s, s] = driver_persistence
        g[s, s, 0] = True
        w = np.zeros(k)
        w[s] = 1.0
        names = [f"x{i}" for i in range(k_sys)] + ["s_regime"]
        actionable: list[int] | None = list(range(k_sys))
    else:
        w = np.zeros(k)
        w[:k_sys] = rng.normal(size=k_sys)
        if d is not None:  # the distractor must not cause anything, including regime switches
            w[d] = 0.0
        w /= np.linalg.norm(w)
        names, actionable = [], None
    world = RegimeSwitchingSCM(
        world_id=world_id,
        seed=seed,
        k=k,
        L=L,
        T=T,
        graph=g,
        noise=noise or InnovationProcess("laplace", 0.1),
        label=label or ThresholdLabel(channel=label_channel, t_label=-1),
        burn_in=burn_in,
        variable_names=names,
        coef=coef,
        regime_weights=w,
        regime_threshold=0.0,
    )
    world.actionable = actionable
    world.roles = {
        "route_source": r,
        "distractor": d,
        "label_channel": label_channel,
        "regime_driver": "state" if driver == "state" else k - 1,
    }
    return world
