"""Controlled mechanism misspecification (plan §Week 6, task 5).

``misspecify(world, kind, level)`` returns a wrong mechanism the explainer believes, with a
severity ``level`` ≥ 0 (``0`` reproduces the world exactly, except for ``observation``):

| kind | what is wrong |
|---|---|
| ``coefficients`` | every non-zero parameter perturbed by ``level`` × its spread (graph kept) |
| ``delete_edges`` | a fraction ``level`` of the cross-channel edges removed |
| ``spurious_edges`` | ``level`` × #edges spurious cross-channel edges added (linear families) |
| ``nonlinearity`` | outputs pass through a saturation ``c·tanh(F/c)`` with ``c = s/level`` (s = typical |F|) |
| ``delay`` | mechanism mixes in one extra step of delay: ``(1−level)·F(x_{t−1..}) + level·F(x_{t−2..})`` |
| ``noise`` | heteroscedasticity misjudged: ``h → h + level`` (changes abduction) |
| ``observation`` | partially observed world: the true state dynamics applied to the observations as if they were the state (wrong observation map) |
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from twist.utils import backend as bk
from twist.utils.backend import Array
from twist.worlds.base_scm import TemporalSCM
from twist.worlds.linear_var import LinearVAR, companion_spectral_radius
from twist.worlds.shifts import perturb_mechanism

KINDS = (
    "coefficients",
    "delete_edges",
    "spurious_edges",
    "nonlinearity",
    "delay",
    "noise",
    "observation",
)


@dataclass
class TransformedSCM(TemporalSCM):
    """A base mechanism seen through a misspecifying transform (``nonlinearity`` or ``delay``)."""

    family: ClassVar[str] = "misspecified"
    base: Any = field(default=None, repr=False)
    kind: str = "nonlinearity"
    level: float = 0.0
    scale: float = 1.0

    def structural_mean(self, past: Array) -> Array:
        b = self.base
        if self.kind == "nonlinearity":
            f = b.structural_mean(past)
            return f if self.level == 0 else self.scale * bk.tanh(f / self.scale)
        if self.kind == "delay":  # L = base.L + 1
            now = b.structural_mean(past[:, : b.L])
            late = b.structural_mean(past[:, 1 : b.L + 1])
            return (1 - self.level) * now + self.level * late
        raise ValueError(self.kind)

    def params(self) -> dict[str, np.ndarray]:
        return {
            **self.base.params(),
            "level": np.asarray(self.level),
            "scale": np.asarray(self.scale),
            "base_graph": np.asarray(self.base.graph),
        }

    def card(self) -> dict[str, Any]:
        return {**super().card(), "kind": self.kind, "base_card": self.base.card()}

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> TransformedSCM:
        from twist.worlds.registry import WORLD_FAMILIES

        bc = card["base_card"]
        base_arrays = {k: v for k, v in arrays.items() if k not in ("level", "scale", "base_graph")}
        base_arrays["graph"] = arrays["base_graph"]
        base = WORLD_FAMILIES[bc["world_family"]].from_saved(bc, base_arrays)
        return cls(
            base=base,
            kind=card["kind"],
            level=float(arrays["level"]),
            scale=float(arrays["scale"]),
            **cls._common_from_card(card, arrays),
        )


def _clone_with(world: TemporalSCM, **changes: Any) -> TemporalSCM:
    new = copy.deepcopy(world)
    for k, v in changes.items():
        setattr(new, k, v)
    return new


def _edit_edges(world: TemporalSCM, level: float, seed: int, add: bool) -> TemporalSCM:
    if not isinstance(world, LinearVAR):
        raise NotImplementedError(f"{'spurious' if add else 'deleted'} edges: linear families only")
    rng = np.random.default_rng(seed)
    coef, g = world.coef.copy(), world.graph.copy()
    _L, k, _ = coef.shape
    off = ~np.eye(k, dtype=bool)
    if add:
        free = np.argwhere(off[:, :, None] & ~g)
        n = min(len(free), round(level * max(int((g & off[:, :, None]).sum()), 1)))
        spread = float(np.abs(coef[coef != 0]).mean()) if (coef != 0).any() else 0.1
        for i, j, lag in free[rng.permutation(len(free))[:n]]:
            coef[lag, i, j] = rng.choice([-1, 1]) * spread
            g[i, j, lag] = True
        rho = companion_spectral_radius(coef)
        if rho > 0.99:
            coef *= 0.99 / rho
    else:
        edges = np.argwhere(g & off[:, :, None])
        n = round(level * len(edges))
        for i, j, lag in edges[rng.permutation(len(edges))[:n]]:
            coef[lag, i, j] = 0.0
            g[i, j, lag] = False
    return _clone_with(world, coef=coef, graph=g)


def misspecify(world: Any, kind: str, level: float, seed: int = 0) -> TemporalSCM:
    if kind not in KINDS:
        raise ValueError(f"unknown misspecification {kind!r}; choose from {KINDS}")
    if kind == "observation":
        if not hasattr(world, "state"):
            raise ValueError("observation misspecification needs a partially observed world")
        if world.k != world.k_state:
            raise NotImplementedError(
                "hidden channels: the state mechanism has the wrong dimension"
            )
        wrong = copy.deepcopy(world.state)
    else:
        state = getattr(world, "state", world)
        if kind == "coefficients":
            wrong = (
                perturb_mechanism(state, rel_scale=level, seed=seed)
                if level > 0
                else copy.deepcopy(state)
            )
        elif kind in ("delete_edges", "spurious_edges"):
            wrong = _edit_edges(state, level, seed, add=kind == "spurious_edges")
        elif kind == "noise":
            n = state.noise
            wrong = _clone_with(
                state,
                noise=dataclasses.replace(n, heteroscedastic=n.heteroscedastic + level),
            )
        else:
            s = 1.0
            if kind == "nonlinearity":
                rng = np.random.default_rng(seed)
                probe = rng.normal(size=(256, state.L, state.k))
                s = float(np.abs(np.asarray(state.structural_mean(probe))).mean()) / max(
                    level, 1e-9
                )
            L = state.L + (1 if kind == "delay" else 0)
            g = (
                state.graph
                if kind != "delay"
                else np.concatenate([state.graph, state.graph], axis=2)[:, :, :L]
            )
            wrong = TransformedSCM(
                world_id=state.world_id,
                seed=state.seed,
                k=state.k,
                L=L,
                T=state.T,
                graph=g,
                noise=state.noise,
                label=state.label,
                burn_in=state.burn_in,
                variable_names=list(state.variable_names),
                assumptions=dict(state.assumptions),
                base=state,
                kind=kind,
                level=level,
                scale=s,
            )
    wrong.world_id = f"{getattr(world, 'world_id', 'w')}__misspec-{kind}-{level:g}"
    return wrong
