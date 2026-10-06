"""LinearSCM-T: stable sparse VAR(L) world (draft Eq. 8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from twist.utils.backend import Array, as_like, einsum
from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import LabelFunction, ThresholdLabel


def sample_lagged_graph(
    rng: np.random.Generator, k: int, L: int, density: float, allow_lag1_self: bool = False
) -> np.ndarray:
    """Bernoulli(density) lagged graph ``(k, k, L)``; lag-1 self-loops forbidden by default."""
    g = rng.random((k, k, L)) < density
    if not allow_lag1_self:
        g[np.arange(k), np.arange(k), 0] = False
    return g


def companion_spectral_radius(A: np.ndarray) -> float:
    """Spectral radius of the VAR(L) companion matrix for coefficients ``A`` of shape (L, k, k)."""
    L, k, _ = A.shape
    comp = np.zeros((L * k, L * k))
    comp[:k] = np.concatenate(list(A), axis=1)
    if L > 1:
        comp[k:, :-k] = np.eye((L - 1) * k)
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


@dataclass
class LinearVAR(TemporalSCM):
    family: ClassVar[str] = "linear"
    coef: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 1)))  # (L, k, k)

    def structural_mean(self, past: Array) -> Array:
        # x_t = Σ_l A_l x_{t-l}  with past[:, l-1] = x_{t-l}
        return einsum("lij,blj->bi", as_like(self.coef, past), past)

    def params(self) -> dict[str, np.ndarray]:
        return {"coef": self.coef}

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> LinearVAR:
        return cls(coef=arrays["coef"], **cls._common_from_card(card, arrays))


def rescale_to_radius(A: np.ndarray, target: float, mode: str = "cap") -> np.ndarray:
    """Scale all lag matrices by one factor so the companion radius is ≤ (``cap``) or = (``exact``)
    ``target``. Uses bisection, since for ``L > 1`` the radius is not linear in the factor."""
    if mode not in ("cap", "exact"):
        raise ValueError(f"rescale must be 'cap' or 'exact', got {mode!r}")
    rho = companion_spectral_radius(A)
    if rho == 0 or (mode == "cap" and rho <= target):
        return A
    if A.shape[0] == 1:  # closed form for L = 1 (bit-identical to the draft generator)
        return A * (target / rho)
    lo, hi = 0.0, target / rho if rho > target else 1.0
    while companion_spectral_radius(A * hi) < target and mode == "exact":
        lo, hi = hi, hi * 2
        if hi > 1e6:  # nilpotent-like structure: radius cannot reach the target
            return A
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if companion_spectral_radius(A * mid) <= target:
            lo = mid
        else:
            hi = mid
    return A * lo


def designate_route_and_distractor(
    g: np.ndarray,
    label_channel: int = 0,
    route: bool = True,
    distractor: bool = True,
) -> tuple[np.ndarray, int | None, int | None]:
    """Guarantee a causal route into the label channel and a distractor channel.

    * route: channel ``r`` (the first channel ≠ label) gets a lag-1 edge into the label channel;
    * distractor: the last channel ``d`` has no outgoing edges to other channels but receives a
      lag-1 edge from ``r``. It is correlated with a cause of the label without causing it.

    Returns the modified graph and the designated ``(route_source, distractor)`` channels.
    """
    g = g.copy()
    k = g.shape[0]
    r = next((c for c in range(k) if c != label_channel), None) if route else None
    d = None
    if r is not None:
        g[label_channel, r, 0] = True
    if distractor and k >= 3:
        d = k - 1 if k - 1 not in (label_channel, r) else None
        if d is not None:
            others = [i for i in range(k) if i != d]
            g[others, d, :] = False
            if r is not None:
                g[d, r, 0] = True
    return g, r, d


def make_linear_var(
    world_id: str,
    seed: int,
    k: int = 10,
    L: int = 1,
    T: int = 100,
    density: float = 0.2,
    spectral_radius: float = 0.9,
    rescale: str = "cap",
    strength: float = 0.5,
    oscillatory: bool = False,
    ensure_route: bool = False,
    distractor: bool = False,
    label_channel: int = 0,
    noise: InnovationProcess | None = None,
    label: LabelFunction | None = None,
    burn_in: int = 100,
) -> LinearVAR:
    """Entries ~ U(−strength, strength) (lag ℓ scaled by 1/ℓ), masked by G, then rescaled so the
    companion radius is ≤ ``spectral_radius`` (``cap``, the draft's rule) or equal to it
    (``exact``, which keeps the intervention-to-label effect from vanishing in sparse graphs).

    ``oscillatory`` adds a rotation between the label channel and the route channel, so the
    dominant modes are complex. ``ensure_route``/``distractor`` guarantee a causal route into the
    label channel and a non-causal correlated channel (``designate_route_and_distractor``).
    Defaults reproduce the draft generator.
    """
    rng = np.random.default_rng(seed)
    g = sample_lagged_graph(rng, k, L, density)
    g, r, d = designate_route_and_distractor(g, label_channel, ensure_route, distractor)
    lag_scale = (1.0 / np.arange(1, L + 1))[:, None, None]
    A = rng.uniform(-strength, strength, (L, k, k)) * lag_scale * g.transpose(2, 0, 1)
    if r is not None:  # the route edge is never negligible
        A[0, label_channel, r] = rng.choice([-1, 1]) * strength * rng.uniform(0.5, 1.0)
    if oscillatory and r is not None:
        a = strength * rng.uniform(0.7, 1.0)
        A[0, label_channel, r], A[0, r, label_channel] = a, -a
        g[r, label_channel, 0] = True
    A = rescale_to_radius(A, spectral_radius, rescale)
    world = LinearVAR(
        world_id=world_id,
        seed=seed,
        k=k,
        L=L,
        T=T,
        graph=g,
        noise=noise or InnovationProcess("laplace", 0.1),
        label=label or ThresholdLabel(channel=0, t_label=-1),
        burn_in=burn_in,
        coef=A,
    )
    world.roles = {"route_source": r, "distractor": d, "label_channel": label_channel}
    return world
