"""Shift and misspecification operators (Weeks 6–7).

Planned shift levels: innovation (Laplace → Gaussian/uniform/heteroscedastic), mechanism
(coefficients, edges, delays, nonlinearity), observation (sensor noise, missingness, transforms),
decision (label threshold/site, classifier retraining).
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np

from twist.worlds.base_scm import TemporalSCM


def innovation_shift(world: Any, kind: str, scale: float | None = None) -> Any:
    """Same graph, mechanism, label and seed with a different innovation law (draft Table 2).

    ``scale=None`` matches the variance of the current law (Laplace b ↔ Gaussian b√2 ↔ uniform
    half-width b√6), so the shift changes the shape of the innovations, not their size. Works for
    partially observed worlds by shifting their state world.
    """
    shifted = copy.deepcopy(world)
    target = getattr(shifted, "state", shifted)
    old = target.noise
    if scale is None:
        std = {"laplace": np.sqrt(2.0), "gaussian": 1.0, "uniform": 1 / np.sqrt(3.0)}[
            old.kind
        ] * old.scale
        scale = std * {"laplace": 1 / np.sqrt(2.0), "gaussian": 1.0, "uniform": np.sqrt(3.0)}[kind]
    target.noise = dataclasses.replace(old, kind=kind, scale=float(scale))
    shifted.world_id = f"{world.world_id}__innov-{kind}"
    return shifted


# parameters that encode fixed structure rather than learnable strength
_FIXED_PARAMS = frozenset({"input_scale", "coupling", "decay", "gain"})


def perturb_mechanism(
    world: TemporalSCM, rel_scale: float = 0.5, seed: int = 0, max_radius: float = 0.99
) -> TemporalSCM:
    """A misspecified copy of ``world``: same family, graph, noise and label; perturbed strengths.

    Every non-zero entry of each mechanism parameter array gets Gaussian noise with standard
    deviation ``rel_scale × std(non-zero entries)``. Zero entries stay zero, so the graph is
    preserved and only the functional form is wrong. Linear-family worlds are rescaled to a
    companion spectral radius ≤ ``max_radius`` so the wrong mechanism stays stable.
    """
    import numpy as np

    from twist.worlds.linear_var import LinearVAR, companion_spectral_radius

    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {"graph": np.asarray(world.graph)}
    for name, a in world.params().items():
        a = np.asarray(a, float)
        if name in _FIXED_PARAMS or a.ndim == 0:
            arrays[name] = a
            continue
        nz = a != 0
        spread = float(a[nz].std()) if nz.sum() > 1 else float(np.abs(a).max(initial=1.0))
        arrays[name] = a + rel_scale * spread * rng.normal(size=a.shape) * nz
    card = world.card()
    card["world_id"] = f"{world.world_id}__perturbed-{rel_scale:g}-s{seed}"
    wrong = type(world).from_saved(card, arrays)
    if isinstance(wrong, LinearVAR):
        rho = companion_spectral_radius(wrong.coef)
        if rho > max_radius:
            wrong.coef = wrong.coef * (max_radius / rho)
    wrong.label = copy.deepcopy(world.label)
    return wrong


# --------------------------------------------------------------------------- Week-7 shift library
SHIFT_LEVELS = {
    "innovation": ("gaussian", "uniform", "laplace", "heteroscedastic", "correlated"),
    "mechanism": ("coefficients", "delete_edges", "spurious_edges", "nonlinearity", "delay"),
    "observation": ("noise", "mcar", "saturating"),
    "decision": ("threshold", "site", "retrain"),
}


def shift_id(spec: dict[str, Any]) -> str:
    return f"{spec['level']}-{spec['kind']}-{float(spec.get('amount', 0)):g}"


def apply_shift(world: Any, spec: dict[str, Any], reference_state: np.ndarray) -> Any:
    """The deployment world after a shift (plan §Week 7, task 1).

    ``spec = {level, kind, amount}``:

    * innovation — ``gaussian`` / ``uniform`` / ``laplace`` (variance-matched law), ``heteroscedastic``
      (h += amount), ``correlated`` (cross-channel correlation = amount);
    * mechanism — any misspecification kind at level ``amount`` (the *world* changes; explainers
      keep the base mechanism);
    * observation — ``noise`` (sensor noise std = amount × state std), ``mcar`` (missing rate =
      amount), ``saturating`` (sensor ``c·tanh(z/c)``, c = amount × state std). The world becomes
      partially observed;
    * decision — ``threshold`` (label threshold += amount × label-score std), ``site`` (label read at
      ``amount × T`` and refit for balance), ``retrain`` (same world; the evaluation classifier is
      retrained by the runner).

    ``reference_state`` (base training states) calibrates relative amounts and refits.
    """
    from twist.mechanisms.misspecification import misspecify
    from twist.worlds.state_space import ObservationModel, ObservedSCM

    level, kind = spec["level"], spec["kind"]
    amount = float(spec.get("amount", 0.0))
    if kind not in SHIFT_LEVELS.get(level, ()):
        raise ValueError(f"unknown shift {level}/{kind}; available: {SHIFT_LEVELS}")
    state_sd = float(reference_state.std())
    if level == "innovation":
        if kind in ("gaussian", "uniform", "laplace"):
            shifted = innovation_shift(world, kind)
        else:
            shifted = copy.deepcopy(world)
            target = getattr(shifted, "state", shifted)
            n = target.noise
            target.noise = dataclasses.replace(
                n,
                correlation=amount if kind == "correlated" else n.correlation,
                heteroscedastic=n.heteroscedastic + amount
                if kind == "heteroscedastic"
                else n.heteroscedastic,
            )
    elif level == "mechanism":
        state = getattr(world, "state", world)
        wrong = misspecify(state, kind, amount, seed=int(spec.get("seed", 0)))
        wrong.label = copy.deepcopy(state.label)
        shifted = (
            ObservedSCM(world.world_id, wrong, copy.deepcopy(world.observation))
            if hasattr(world, "state")
            else wrong
        )
    elif level == "observation":
        state = copy.deepcopy(getattr(world, "state", world))
        obs = (
            copy.deepcopy(world.observation)
            if hasattr(world, "observation")
            else ObservationModel()
        )
        if kind == "noise":
            obs.noise_scale = amount * state_sd
        elif kind == "mcar":
            obs.missing, obs.missing_rate = "mcar", amount
        else:
            obs.transform, obs.saturation = "saturating", amount * state_sd
        shifted = ObservedSCM(world.world_id, state, obs)
    else:  # decision
        shifted = copy.deepcopy(world)
        label = shifted.label
        if kind == "threshold":
            if not hasattr(label, "threshold"):
                raise NotImplementedError(
                    f"threshold shift needs a threshold label, got {label.type}"
                )
            label.threshold += amount * float(np.std(label.raw(reference_state)))
        elif kind == "site":
            if not hasattr(label, "t_label"):
                raise NotImplementedError(f"site shift needs a threshold label, got {label.type}")
            label.t_label = int(amount * world.T)
            label.fit(reference_state)
    shifted.world_id = f"{world.world_id}__shift-{shift_id(spec)}"
    return shifted


def perturb_units(
    world: Any, exo: dict[str, np.ndarray], scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Nearby but coherent units: every innovation is perturbed by ``scale`` × its channel std and
    the unit is re-generated by the world (for partially observed worlds, re-observed with the same
    sensor exogenous). Used for explanation-stability tests (plan §Week 7, task 2)."""
    from twist.replay.abduction import abduct_noise
    from twist.replay.pearl_replay import replay

    state_world = getattr(world, "state", world)
    z = exo["state"]
    eta = abduct_noise(z, state_world)
    sd = eta.reshape(-1, eta.shape[-1]).std(0) + 1e-12
    eta_p = eta + scale * sd * rng.normal(size=eta.shape)
    zeros = np.zeros_like(z)
    z_p = replay(state_world, eta_p, np.zeros(z.shape, bool), zeros, zeros)
    new = {**exo, "state": z_p, "eta": eta_p}
    if hasattr(world, "observation"):
        x_p = world.observation.observe(z_p, exo["obs_noise"], exo["obs_mask"])
    else:
        x_p = z_p
    return x_p, new
