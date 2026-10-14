"""World-family registry and config-driven construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import make_label
from twist.worlds.linear_var import LinearVAR, make_linear_var
from twist.worlds.nonlinear_scm import NonlinearMLPSCM, make_nonlinear_scm
from twist.worlds.regime_switching import RegimeSwitchingSCM, make_regime_scm
from twist.worlds.semi_synthetic import EnergyGridSCM, make_energy_grid
from twist.worlds.spring_system import SpringSystem, make_spring_system
from twist.worlds.state_space import ObservedSCM, make_statespace

WORLD_FAMILIES: dict[str, Any] = {
    "linear": LinearVAR,
    "nonlinear": NonlinearMLPSCM,
    "spring": SpringSystem,
    "regime": RegimeSwitchingSCM,
    "statespace": ObservedSCM,
    "energy": EnergyGridSCM,
}


def _register_late() -> None:
    """Families defined outside ``twist.worlds`` (imported lazily to avoid cycles)."""
    from twist.mechanisms.misspecification import TransformedSCM

    WORLD_FAMILIES.setdefault("misspecified", TransformedSCM)


_FACTORIES: dict[str, Callable[..., Any]] = {
    "linear": make_linear_var,
    "nonlinear": make_nonlinear_scm,
    "spring": make_spring_system,
    "regime": make_regime_scm,
    "statespace": make_statespace,
    "energy": make_energy_grid,
}


def make_world(
    family: str,
    world_id: str,
    seed: int,
    noise: dict[str, Any] | None = None,
    label: dict[str, Any] | None = None,
    action_space: dict[str, Any] | None = None,
    **params: Any,
) -> TemporalSCM:
    if family not in _FACTORIES:
        raise KeyError(f"unknown world family {family!r}; available: {sorted(_FACTORIES)}")
    world = _FACTORIES[family](
        world_id=world_id,
        seed=seed,
        noise=InnovationProcess(**noise) if noise else None,
        label=make_label(label) if label else None,
        **params,
    )
    if action_space:
        world.configure_action_space(**action_space)
    return world
