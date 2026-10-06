"""NlinearSCM-T: bounded per-node masked-MLP transition (draft Eq. 9).

    x^i_t = decay_i · x^i_{t-1} + gain · tanh(MLP_i(masked lags / sqrt(max(n_parents_i, 1)))) + ε^i_t

Registered hyperparameters: hidden width 16, gain 1.5, decay_i ~ U[0.1, 0.4], spectral-norm cap
6.0 per weight matrix, init gain 3.0. Hidden activation is tanh (this repository's choice).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from twist.utils.backend import Array, as_like, einsum, tanh
from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import LabelFunction, ThresholdLabel
from twist.worlds.linear_var import designate_route_and_distractor, sample_lagged_graph


@dataclass
class NonlinearMLPSCM(TemporalSCM):
    family: ClassVar[str] = "nonlinear"
    W1: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k, H, L*k), pre-masked
    b1: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k, H)
    w2: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k, H)
    b2: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,)
    decay: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,)
    input_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (k,)
    gain: float = 1.5

    def structural_mean(self, past: Array) -> Array:
        B = past.shape[0]
        z = past.reshape(B, self.L * self.k)
        h = tanh(
            einsum("ihd,bd->bih", as_like(self.W1 * self.input_scale[:, None, None], past), z)
            + as_like(self.b1, past)[None]
        )
        out = einsum("bih,ih->bi", h, as_like(self.w2, past)) + as_like(self.b2, past)[None]
        return as_like(self.decay, past)[None] * past[:, 0] + self.gain * tanh(out)

    def dependency_graph(self) -> np.ndarray:
        """Recorded graph plus the lag-1 self-dependence of the ``decay_i · x^i_{t-1}`` term."""
        g = np.asarray(self.graph, bool).copy()
        g[np.arange(self.k), np.arange(self.k), 0] |= self.decay != 0
        return g

    def params(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "decay": self.decay,
            "input_scale": self.input_scale,
            "gain": np.asarray(self.gain),
        }

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> NonlinearMLPSCM:
        p: dict[str, Any] = {n: arrays[n] for n in ("W1", "b1", "w2", "b2", "decay", "input_scale")}
        return cls(gain=float(arrays["gain"]), **p, **cls._common_from_card(card, arrays))


def _cap_spectral_norm(W: np.ndarray, cap: float) -> np.ndarray:
    s = np.linalg.norm(W, 2)
    return W * (cap / s) if s > cap else W


def make_nonlinear_scm(
    world_id: str,
    seed: int,
    k: int = 10,
    L: int = 1,
    T: int = 100,
    density: float = 0.2,
    hidden: int = 16,
    gain: float = 1.5,
    decay_range: tuple[float, float] = (0.1, 0.4),
    spectral_cap: float = 6.0,
    init_gain: float = 3.0,
    ensure_route: bool = False,
    distractor: bool = False,
    label_channel: int = 0,
    noise: InnovationProcess | None = None,
    label: LabelFunction | None = None,
    burn_in: int = 100,
) -> NonlinearMLPSCM:
    """Factor knobs: ``gain`` (effect strength), ``decay_range`` (stability: contractive
    ``(0.1, 0.4)``, weakly stable ``(0.6, 0.9)``, oscillatory ``(−0.9, −0.6)``), ``L``, ``density``;
    ``ensure_route``/``distractor`` as in ``designate_route_and_distractor``."""
    rng = np.random.default_rng(seed)
    g = sample_lagged_graph(rng, k, L, density)
    g, r, d = designate_route_and_distractor(g, label_channel, ensure_route, distractor)
    # mask[i, l*k + j] = G[i, j, l]  — matches past.reshape(B, L*k) with past[:, l] = x_{t-l-1}
    mask = g.transpose(0, 2, 1).reshape(k, L * k).astype(float)
    W1 = np.empty((k, hidden, L * k))
    w2 = np.empty((k, hidden))
    for i in range(k):
        W1[i] = _cap_spectral_norm(
            rng.normal(0, init_gain / np.sqrt(L * k), (hidden, L * k)) * mask[i][None], spectral_cap
        )
        w2[i] = _cap_spectral_norm(
            rng.normal(0, init_gain / np.sqrt(hidden), (1, hidden)), spectral_cap
        )[0]
    n_par = mask.sum(1)
    world = NonlinearMLPSCM(
        world_id=world_id,
        seed=seed,
        k=k,
        L=L,
        T=T,
        graph=g,
        noise=noise or InnovationProcess("laplace", 0.1),
        label=label or ThresholdLabel(channel=0, t_label=-1),
        burn_in=burn_in,
        W1=W1,
        b1=rng.normal(0, 0.1, (k, hidden)),
        w2=w2,
        b2=rng.normal(0, 0.1, k),
        decay=rng.uniform(*decay_range, k),
        input_scale=1.0 / np.sqrt(np.maximum(n_par, 1.0)),
        gain=gain,
    )
    world.roles = {"route_source": r, "distractor": d, "label_channel": label_channel}
    return world
