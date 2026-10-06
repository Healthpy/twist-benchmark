"""Temporal additive-noise SCM base class.

A world ``M* = (G*, F*, P*, ℓ*)`` with lagged parents only::

    X^i_t = F_i(x_{t-1}, ..., x_{t-L}) + ε^i_t,    t >= L

The first ``L`` retained timesteps are treated as exogenous roots (``ε_t := x_t``) so that
abduction is exact for every timestep of a stored trajectory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from twist.actions.schemas import ActionSpace, ControlChannel
from twist.utils import backend as bk
from twist.utils.backend import Array
from twist.utils.serialization import hash_arrays, read_json, write_json
from twist.worlds.labels import LabelFunction


@dataclass
class InnovationProcess:
    """Exogenous innovations ``η_t`` (i.i.d. over time).

    ``scale`` is the natural scale parameter of ``kind``. ``correlation`` ∈ [0, 1) mixes in a
    shared component across channels at each timestep (equicorrelated η). ``heteroscedastic`` ≥ 0
    makes the innovation enter as ``s(past)·η_t`` with ``s_i = 1 + h·tanh|x^i_{t−1}|``. The
    exogenous variable is still ``η``, so abduction and unit-level replay stay exact.
    """

    kind: str = "laplace"
    scale: float = 0.1
    correlation: float = 0.0
    heteroscedastic: float = 0.0

    def _draw(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        if self.kind == "laplace":
            return rng.laplace(0.0, self.scale, shape)
        if self.kind == "gaussian":
            return rng.normal(0.0, self.scale, shape)
        if self.kind == "uniform":
            return rng.uniform(-self.scale, self.scale, shape)
        raise ValueError(f"unknown innovation kind {self.kind!r}")

    def sample(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        eta = self._draw(rng, shape)
        if self.correlation > 0:
            common = self._draw(rng, (*shape[:-1], 1))
            eta = np.sqrt(1 - self.correlation) * eta + np.sqrt(self.correlation) * common
        return eta

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scale": self.scale,
            "correlation": self.correlation,
            "heteroscedastic": self.heteroscedastic,
        }


DEFAULT_ASSUMPTIONS = {
    "additive_noise": True,
    "observed_state": True,
    "confounding": False,
    "positive_lags": True,
    "exact_replay": True,
}


@dataclass
class TemporalSCM(ABC):
    """Abstract lagged additive-noise SCM with an exposed mechanism and label rule.

    ``graph[i, j, l-1] = True`` means ``X^j_{t-l} → X^i_t``.
    """

    family: ClassVar[str] = "base"

    world_id: str
    seed: int
    k: int
    L: int
    T: int
    graph: np.ndarray
    noise: InnovationProcess
    label: LabelFunction
    burn_in: int = 100
    variable_names: list[str] = field(default_factory=list)
    assumptions: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_ASSUMPTIONS))
    actionable: list[int] | None = None  # None → every observed channel
    action_lower: list[float] | None = None
    action_upper: list[float] | None = None
    controls: dict[str, ControlChannel] = field(default_factory=dict)
    # designated structure: {"label_channel", "route_source", "distractor", ...} (None = not set)
    roles: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variable_names:
            self.variable_names = [f"x{i}" for i in range(self.k)]

    # ----------------------------------------------------------------- mechanism
    @abstractmethod
    def structural_mean(self, past: Array) -> Array:
        """``F(past)`` for a batch of lag windows.

        ``past`` has shape ``(B, L, k)`` with ``past[:, l-1] = x_{t-l}``; returns ``(B, k)``.
        Must work for NumPy arrays and torch tensors.
        """

    @abstractmethod
    def params(self) -> dict[str, np.ndarray]:
        """Mechanism parameters to persist."""

    def dependency_graph(self) -> np.ndarray:
        """Full lagged dependency structure of ``F`` (recorded graph plus any built-in dynamics)."""
        return np.asarray(self.graph, bool)

    def innovation_scale(self, past: Array) -> Array | None:
        """``s(past)`` multiplying the innovation, shape ``(B, k)``; ``None`` means 1."""
        h = self.noise.heteroscedastic
        if h == 0:
            return None
        return 1.0 + h * bk.tanh(abs(past[:, 0]))

    def step(self, past: Array, eta: Array) -> Array:
        """One structural step ``F(past) + s(past)·η``."""
        s = self.innovation_scale(past)
        return self.structural_mean(past) + (eta if s is None else s * eta)

    def one_step_prediction(self, x: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Teacher-forced ``F(x_{t−L:t}) + s(x_{t−L:t})·η_t`` for every ``t`` (roots: ``η_t``)."""
        B, T, k = x.shape
        out = np.array(eta, dtype=float, copy=True)
        if T <= self.L:
            return out
        windows = np.stack([x[:, t - self.L : t][:, ::-1] for t in range(self.L, T)], axis=1)
        flat = windows.reshape(B * (T - self.L), self.L, k)
        e = eta[:, self.L :].reshape(B * (T - self.L), k)
        out[:, self.L :] = np.asarray(self.step(flat, e)).reshape(B, T - self.L, k)
        return out

    def one_step_scales(self, x: np.ndarray) -> np.ndarray:
        """Teacher-forced ``s(x_{t−L:t})`` (ones for roots and homoscedastic worlds)."""
        B, T, k = x.shape
        out = np.ones_like(x)
        if self.noise.heteroscedastic == 0 or T <= self.L:
            return out
        windows = np.stack([x[:, t - self.L : t][:, ::-1] for t in range(self.L, T)], axis=1)
        s = self.innovation_scale(windows.reshape(B * (T - self.L), self.L, k))
        out[:, self.L :] = np.asarray(s).reshape(B, T - self.L, k)
        return out

    def lag_window(self, x: Array, t: int) -> Array:
        """``(B, L, k)`` window of lags for timestep ``t`` from a batch ``(B, T, k)``."""
        return x[:, t - self.L : t].flip(1) if hasattr(x, "flip") else x[:, t - self.L : t][:, ::-1]

    def one_step_means(self, x: np.ndarray) -> np.ndarray:
        """Teacher-forced ``F(Pa(x_t))`` for ``t >= L`` (NaN for roots); shape ``(B, T, k)``."""
        B, T, k = x.shape
        out = np.full_like(x, np.nan)
        if T <= self.L:
            return out
        windows = np.stack([x[:, t - self.L : t][:, ::-1] for t in range(self.L, T)], axis=1)
        flat = windows.reshape(B * (T - self.L), self.L, k)
        out[:, self.L :] = np.asarray(self.structural_mean(flat)).reshape(B, T - self.L, k)
        return out

    # ----------------------------------------------------------------- sampling
    def simulate(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``n`` trajectories. Returns ``(x, η)`` each ``(n, T, k)``, after burn-in."""
        total = self.burn_in + self.T
        eps = self.noise.sample(rng, (n, total, self.k))
        x = np.zeros_like(eps)
        x[:, : self.L] = eps[:, : self.L]
        for t in range(self.L, total):
            x[:, t] = self.step(x[:, t - self.L : t][:, ::-1], eps[:, t])
        x = x[:, self.burn_in :]
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"world {self.world_id} diverged during simulation")
        return x, eps[:, self.burn_in :]

    # ----------------------------------------------------------------- unit-level world API
    # Shared with ObservedSCM so the pipeline treats observed-state and partially observed
    # worlds uniformly: units carry their exogenous variables, replay is done by the world.
    def simulate_units(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x, eta = self.simulate(n, rng)
        return x, {"state": x, "eta": eta}

    def replay_units(
        self, exo: dict[str, np.ndarray], actions: Any, semantics: str = "pearl"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oracle replay from the units' exogenous variables; returns ``(observed, state)``.

        For observed-state worlds both are the same trajectory, and innovations are abducted from
        the stored state exactly as ``replay_pearl`` does.
        """
        from twist.replay import replay_noiseless, replay_pearl

        fn = replay_pearl if semantics == "pearl" else replay_noiseless
        x_cf = fn(exo["state"], actions, self)
        return x_cf, x_cf

    def outcome(self, state: np.ndarray) -> np.ndarray:
        """World label ``ℓ*`` evaluated on the (true) state."""
        return self.label(state)

    def to_state_plan(self, plan: Any) -> Any:
        """Readings are made on observed channels; for observed-state worlds those are the state."""
        return plan

    @property
    def k_state(self) -> int:
        return self.k

    # ----------------------------------------------------------------- action interface
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            self.T,
            self.k,
            self.variable_names,
            list(range(self.k)) if self.actionable is None else list(self.actionable),
            lower=self.action_lower,
            upper=self.action_upper,
            controls=dict(self.controls),
        )

    def configure_action_space(
        self,
        actionable: list[int] | None = None,
        lower: float | list[float] | None = None,
        upper: float | list[float] | None = None,
        controls: dict[str, dict[str, Any]] | None = None,
    ) -> TemporalSCM:
        """Declare actionable channels, value bounds and high-level control inputs."""

        def per_channel(v: float | list[float] | None) -> list[float] | None:
            return None if v is None else np.broadcast_to(np.asarray(v, float), (self.k,)).tolist()

        self.actionable = actionable
        self.action_lower, self.action_upper = per_channel(lower), per_channel(upper)
        self.controls = {
            name: ControlChannel(name=name, **spec) for name, spec in (controls or {}).items()
        }
        for ch in self.controls.values():
            if len(ch.loadings) != self.k:
                raise ValueError(
                    f"control {ch.name!r} needs {self.k} loadings, got {len(ch.loadings)}"
                )
        return self

    # ----------------------------------------------------------------- persistence
    def card(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "world_family": self.family,
            "world_seed": self.seed,
            "k": self.k,
            "L": self.L,
            "T": self.T,
            "burn_in": self.burn_in,
            "noise": self.noise.to_dict(),
            "label": self.label.to_dict(),
            "variable_names": self.variable_names,
            "assumptions": self.assumptions,
            "roles": self.roles,
            "action_space": {
                "actionable": self.actionable,
                "lower": self.action_lower,
                "upper": self.action_upper,
                "controls": {n: c.to_dict() for n, c in self.controls.items()},
            },
            "graph_density": float(self.graph.mean()),
            "n_edges": int(self.graph.sum()),
            "mechanism_hash": hash_arrays(*self.params().values()),
        }

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"graph": self.graph, **self.params()}
        np.savez(d / "mechanism.npz", **arrays)
        write_json(d / "world_card.json", self.card())
        return d

    @classmethod
    def load(cls, directory: str | Path) -> TemporalSCM:
        from twist.worlds.registry import WORLD_FAMILIES, _register_late

        _register_late()
        d = Path(directory)
        card = read_json(d / "world_card.json")
        arrays = dict(np.load(d / "mechanism.npz"))
        return WORLD_FAMILIES[card["world_family"]].from_saved(card, arrays)

    @classmethod
    def _common_from_card(
        cls, card: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        return {
            "world_id": card["world_id"],
            "seed": card["world_seed"],
            "k": card["k"],
            "L": card["L"],
            "T": card["T"],
            "burn_in": card["burn_in"],
            "graph": arrays["graph"].astype(bool),
            "noise": InnovationProcess(**card["noise"]),
            "label": LabelFunction.from_dict(card["label"]),
            "variable_names": card["variable_names"],
            "assumptions": card["assumptions"],
            "roles": card.get("roles", {}),
            "actionable": card.get("action_space", {}).get("actionable"),
            "action_lower": card.get("action_space", {}).get("lower"),
            "action_upper": card.get("action_space", {}).get("upper"),
            "controls": {
                n: ControlChannel.from_dict(c)
                for n, c in card.get("action_space", {}).get("controls", {}).items()
            },
        }

    @classmethod
    @abstractmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> TemporalSCM: ...
