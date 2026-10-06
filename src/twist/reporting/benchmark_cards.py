"""World cards (Markdown) for every family, including the replay-exactness disclosure."""

from __future__ import annotations

from typing import Any

import numpy as np

REPLAY_DISCLOSURE = {
    True: "**Exact.** Additive-noise abduction from the observations recovers the factual "
    "innovations, so unit-level replay is exact from `x` alone. All mechanism metrics "
    "(CF-faith, D, structural residuals) are defined.",
    False: "**Oracle replay only.** The observations do not identify the factual exogenous "
    "variables (hidden channels, sensor noise, nonlinear sensor or missingness). The benchmark "
    "replays from the stored exogenous variables of each unit. Abduction-based metrics "
    "(CF-faith, D, residual readings) are reported as NaN, and explainers that abduct from `x` "
    "are skipped; estimated-mechanism recourse plans on the observations.",
}


def _mechanism_summary(world: Any) -> list[str]:
    from twist.worlds.linear_var import LinearVAR, companion_spectral_radius
    from twist.worlds.nonlinear_scm import NonlinearMLPSCM
    from twist.worlds.regime_switching import RegimeSwitchingSCM
    from twist.worlds.spring_system import SpringSystem

    if isinstance(world, SpringSystem):
        rho = companion_spectral_radius(world.coef)
        return [
            f"- coupled springs: {int((world.coupling > 0).sum())} couplings, transition "
            f"spectral radius {rho:.3f}"
        ]
    if isinstance(world, LinearVAR):
        return [
            f"- VAR({world.L}): companion spectral radius {companion_spectral_radius(world.coef):.3f}"
        ]
    if isinstance(world, NonlinearMLPSCM):
        return [
            f"- per-node MLP, gain {world.gain}, decay ∈ [{world.decay.min():.2f}, "
            f"{world.decay.max():.2f}]"
        ]
    if isinstance(world, RegimeSwitchingSCM):
        radii = [companion_spectral_radius(world.coef[r]) for r in range(2)]
        return [
            f"- two linear regimes, spectral radii {radii[0]:.3f} / {radii[1]:.3f}, driver "
            f"`{world.roles.get('regime_driver')}`"
        ]
    return []


def world_card_markdown(world: Any, integrity: dict[str, Any] | None = None) -> str:
    c = world.card()
    state = getattr(world, "state", world)
    sc = state.card()
    exact = bool(
        world.assumptions.get("exact_abduction", world.assumptions.get("exact_replay", True))
    )
    space = world.action_space()
    lines = [
        f"# World card: `{c['world_id']}`",
        "",
        "## Identity",
        f"- family **{c['world_family']}**"
        + (f" (state family `{sc['world_family']}`)" if c["world_family"] == "statespace" else "")
        + f", seed {c['world_seed']}",
        f"- observed channels k={c['k']}, state channels {sc['k']}, lags L={sc['L']}, length "
        f"T={sc['T']}, burn-in {sc['burn_in']}",
        f"- mechanism hash `{c['mechanism_hash']}`",
        "",
        "## Mechanism",
        f"- graph: {sc['n_edges']} lagged edges (density {sc['graph_density']:.3f})",
        *_mechanism_summary(state),
        f"- innovations: {sc['noise']}",
        "",
        "## Label",
        f"- {c['label']}",
        f"- label channels {world.label.channels}, site t={world.label.site(sc['T'])}",
        "",
        "## Designated structure",
        f"- roles: {world.roles}",
        "",
        "## Action space",
        f"- actionable state channels: {space.actionable}",
        f"- value bounds: {space.lower} / {space.upper}",
        f"- controls: {sorted(space.controls) or 'none'}",
    ]
    if "observation" in c:
        lines += [
            "",
            "## Observation",
            f"- {c['observation']}",
            f"- observed state channels: {list(world.obs_index)}",
        ]
    lines += [
        "",
        "## Replay",
        f"- {REPLAY_DISCLOSURE[exact]}",
        f"- assumptions: {world.assumptions}",
    ]
    if integrity is not None:
        lines += [
            "",
            "## Integrity checks",
            f"**{'PASS' if integrity['passed'] else 'FAIL'}**",
            "",
            "| check | result | detail |",
            "|---|---|---|",
        ]
        lines += [
            f"| {k['check']} | {'pass' if k['passed'] else '**FAIL**'} | {k['detail']} |"
            for k in integrity["checks"]
        ]
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) and np.isfinite(v) else str(v)
