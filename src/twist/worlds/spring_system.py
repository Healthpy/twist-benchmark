"""SpringSCM-T: undamped coupled-spring particle system (draft Eq. 10).

``n_particles`` particles exposed as ``k = 2 n`` channels (positions then velocities). One
symplectic-Euler step with spring constant ``c``, step ``dt`` and unit mass::

    a_t     = -Lap(K) p_t
    v_{t+1} = v_t + dt · a_t
    p_{t+1} = p_t + dt · v_{t+1}

is linear in the state, so the world is a VAR(1) with a fixed, structured transition matrix.
``K`` is ``c`` times a particle-level Bernoulli(density) mask without self-coupling; the last
``n_exogenous`` particles receive no incoming coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from twist.worlds.base_scm import InnovationProcess
from twist.worlds.labels import LabelFunction, ThresholdLabel
from twist.worlds.linear_var import LinearVAR


@dataclass
class SpringSystem(LinearVAR):
    family: ClassVar[str] = "spring"
    coupling: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def params(self) -> dict[str, np.ndarray]:
        return {"coef": self.coef, "coupling": self.coupling}

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> SpringSystem:
        return cls(
            coef=arrays["coef"], coupling=arrays["coupling"], **cls._common_from_card(card, arrays)
        )


def spring_transition(
    K: np.ndarray, dt: float, damping: float = 0.0, anchor: float = 0.0
) -> np.ndarray:
    """Symplectic-Euler step ``v' = (1 − dt·γ)v − dt·(Lap(K) + κI)p``, ``p' = p + dt·v'``.

    Without a tether (``κ = 0``) the Laplacian's zero mode keeps the spectral radius at 1 even when
    damped, so positions random-walk under noise. ``κ > 0`` with ``γ > 0`` makes it contractive.
    """
    n = K.shape[0]
    lap = np.diag(K.sum(1)) - K + anchor * np.eye(n)
    eye = np.eye(n)
    keep = 1.0 - dt * damping
    return np.block([[eye - dt**2 * lap, dt * keep * eye], [-dt * lap, keep * eye]])


def make_spring_system(
    world_id: str,
    seed: int,
    n_particles: int = 5,
    T: int = 100,
    density: float = 0.3,
    spring_constant: float = 0.3,
    dt: float = 0.1,
    n_exogenous: int = 2,
    damping: float = 0.0,
    anchor: float = 0.0,
    heterogeneous: bool = False,
    ensure_route: bool = False,
    distractor: bool = False,
    noise: InnovationProcess | None = None,
    label: LabelFunction | None = None,
    burn_in: int = 100,
) -> SpringSystem:
    """Factor knobs: ``spring_constant`` (strength), ``damping`` γ and tether ``anchor`` κ
    (stability: γ = κ = 0 is the draft's undamped spring; γ, κ > 0 is contractive), ``density``. ``ensure_route`` couples particle 1 into particle 0
    (the label's particle); ``distractor`` makes the last particle receive from particle 1 while
    exerting no force on any other particle. Defaults reproduce the draft generator."""
    rng = np.random.default_rng(seed)
    n = n_particles
    m = rng.random((n, n)) < density
    np.fill_diagonal(m, False)
    d = n - 1 if distractor and n >= 3 else None
    exo_end = n - 1 if d is not None else n
    if n_exogenous:
        m[max(exo_end - n_exogenous, 1) : exo_end, :] = False
    r = 1 if ensure_route and n >= 2 else None
    if r is not None:
        m[0, r] = True
    if d is not None:
        m[:, d] = False
        m[d, :] = False
        m[d, 1] = True
    # the draft uses one stiffness for every spring; heterogeneous springs draw a per-edge factor
    # in [0.5, 1.5] so worlds sharing a coupling pattern still have distinct mechanisms
    K = spring_constant * m * (rng.uniform(0.5, 1.5, m.shape) if heterogeneous else 1.0)
    A = spring_transition(K, dt, damping, anchor)
    world = SpringSystem(
        world_id=world_id,
        seed=seed,
        k=2 * n,
        L=1,
        T=T,
        graph=(np.abs(A) > 0)[:, :, None],
        noise=noise or InnovationProcess("laplace", 0.1),
        label=label or ThresholdLabel(channel=0, t_label=-1),
        burn_in=burn_in,
        variable_names=[f"p{i}" for i in range(n)] + [f"v{i}" for i in range(n)],
        coef=A[None],
        coupling=K,
    )
    world.roles = {
        "route_source": None if r is None else r,
        "distractor": None if d is None else d,
        "label_channel": 0,
    }
    return world
