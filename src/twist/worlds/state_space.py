"""Partially observed state-space worlds ``z_t = F(z_{t−L:t}) + s·η_t``, ``x_t = H(z_t) + ν_t``.

An :class:`ObservedSCM` wraps any exact state world (linear, nonlinear, spring, regime) with an
:class:`ObservationModel`: partial visibility (hidden state channels, e.g. a latent confounder or
regime driver), sensor noise, a saturating nonlinear sensor, MCAR / block missingness and
irregular sampling (missing entries are carried forward, LOCF).

Replay semantics. Abduction from the observations ``x`` alone is no longer exact. The benchmark
generated every unit, though, and stores its exogenous variables ``(z, η, ν, mask)``. The oracle
replays an action on the *state* with the factual innovations retained, then re-observes it with
the factual sensor noise and missingness. That is still the unit-level counterfactual, defined by
the stored exogenous variables. What an explainer can infer from ``x`` is a separate mechanism-access question:
estimated-mechanism recourse (R3/R4) fits ``F̂`` on the observations and plans there. Actions act on state channels; world outcomes
(labels) are evaluated on the true state, so classifier/world disagreement (``Δoutcome``) can be
non-zero here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from twist.actions.domain_actions import HighLevelControl
from twist.actions.interval import IntervalIntervention, ParametricPulse
from twist.actions.observed_variable import PointIntervention, ScheduleIntervention
from twist.actions.schemas import ActionPlan, ActionSpace, TemporalAction
from twist.utils.serialization import read_json, write_json
from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import LabelFunction

TRANSFORMS = ("identity", "saturating")


@dataclass
class ObservationModel:
    """``x = H(z[observed]) + ν`` with missingness and irregular sampling (LOCF imputation).

    * ``hidden``: state channels that are not observed (everything else is, in order);
    * ``noise_scale``: std of Gaussian sensor noise ``ν`` (absolute units);
    * ``transform``: ``identity`` or ``saturating`` (``c·tanh(z/c)``, invertible, ``c = saturation``);
    * ``missing``: ``none`` | ``mcar`` (per entry, ``missing_rate``) | ``block`` (one block of
      ``block_length`` per channel with probability ``missing_rate``);
    * ``sampling_rate``: probability that a timestep is sampled at all (``1`` = regular).
    """

    hidden: list[int] = field(default_factory=list)
    noise_scale: float = 0.0
    transform: str = "identity"
    saturation: float = 1.0
    missing: str = "none"
    missing_rate: float = 0.0
    block_length: int = 5
    sampling_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        if self.missing not in ("none", "mcar", "block"):
            raise ValueError("missing must be 'none', 'mcar' or 'block'")

    def observed(self, k_state: int) -> list[int]:
        return [i for i in range(k_state) if i not in set(self.hidden)]

    @property
    def is_identity(self) -> bool:
        return (
            not self.hidden
            and self.noise_scale == 0
            and self.transform == "identity"
            and self.missing == "none"
            and self.sampling_rate >= 1.0
        )

    # ------------------------------------------------------------------ exogenous draws
    def sample_exogenous(
        self, rng: np.random.Generator, n: int, T: int, k_obs: int
    ) -> dict[str, np.ndarray]:
        nu = (
            rng.normal(0.0, self.noise_scale, (n, T, k_obs))
            if self.noise_scale > 0
            else np.zeros((n, T, k_obs))
        )
        seen = np.ones((n, T, k_obs), bool)
        if self.sampling_rate < 1.0:
            sampled = rng.random((n, T)) < self.sampling_rate
            sampled[:, 0] = True
            seen &= sampled[:, :, None]
        if self.missing == "mcar":
            seen &= rng.random((n, T, k_obs)) >= self.missing_rate
        elif self.missing == "block":
            for b in range(n):
                for c in range(k_obs):
                    if rng.random() < self.missing_rate and T > self.block_length:
                        s = rng.integers(1, T - self.block_length)
                        seen[b, s : s + self.block_length, c] = False
        seen[:, 0] = True  # the first timestep is always observed (defines LOCF start)
        return {"obs_noise": nu, "obs_mask": seen}

    # ------------------------------------------------------------------ observation map
    def h(self, z: np.ndarray) -> np.ndarray:
        if self.transform == "identity":
            return z
        return self.saturation * np.tanh(z / self.saturation)

    def h_inv(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "identity":
            return y
        c = self.saturation
        return c * np.arctanh(np.clip(y / c, -1 + 1e-9, 1 - 1e-9))

    def observe(self, z: np.ndarray, nu: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """Observed trajectories ``(B, T, k_obs)`` from states ``(B, T, k_state)``."""
        y = self.h(z[..., self.observed(z.shape[-1])]) + nu
        # LOCF: carry the last observed value forward over unobserved entries
        idx = np.where(seen, np.arange(y.shape[1])[None, :, None], 0)
        idx = np.maximum.accumulate(idx, axis=1)
        return np.take_along_axis(y, idx, axis=1)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ObservedSCM:
    """A state world seen through an observation model (duck-types the world API)."""

    family: ClassVar[str] = "statespace"
    world_id: str
    state: TemporalSCM
    observation: ObservationModel

    # ------------------------------------------------------------------ world API
    @property
    def seed(self) -> int:
        return self.state.seed

    @property
    def T(self) -> int:
        return self.state.T

    @property
    def L(self) -> int:
        return self.state.L

    @property
    def k_state(self) -> int:
        return self.state.k

    @property
    def obs_index(self) -> list[int]:
        return self.observation.observed(self.state.k)

    @property
    def k(self) -> int:
        return len(self.obs_index)

    @property
    def label(self) -> LabelFunction:
        return self.state.label

    @property
    def graph(self) -> np.ndarray:
        return self.state.graph

    @property
    def noise(self) -> InnovationProcess:
        return self.state.noise

    @property
    def variable_names(self) -> list[str]:
        return [f"obs_{self.state.variable_names[i]}" for i in self.obs_index]

    @property
    def roles(self) -> dict[str, Any]:
        return {**self.state.roles, "hidden": list(self.observation.hidden)}

    @property
    def assumptions(self) -> dict[str, bool]:
        exact = self.observation.is_identity
        return {
            **self.state.assumptions,
            "observed_state": exact,
            "confounding": bool(
                self.state.roles.get("latent_confounder") is not None
                and self.state.roles.get("latent_confounder") in self.observation.hidden
            ),
            "exact_abduction": exact,
            "exact_replay": exact,  # replay from observations alone
            "oracle_replay": True,  # replay from stored exogenous variables
        }

    def dependency_graph(self) -> np.ndarray:
        return self.state.dependency_graph()

    def action_space(self) -> ActionSpace:
        """Actions act on *state* channels (sensors are not causes)."""
        return self.state.action_space()

    def params(self) -> dict[str, np.ndarray]:
        return self.state.params()

    def configure_action_space(self, **kwargs: Any) -> ObservedSCM:
        self.state.configure_action_space(**kwargs)
        return self

    def outcome(self, state: np.ndarray) -> np.ndarray:
        return self.state.label(state)

    def simulate_units(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z, eta = self.state.simulate(n, rng)
        exo = self.observation.sample_exogenous(rng, n, self.T, self.k)
        x = self.observation.observe(z, exo["obs_noise"], exo["obs_mask"])
        return x, {"state": z, "eta": eta, **exo}

    def replay_units(
        self, exo: dict[str, np.ndarray], actions: Any, semantics: str = "pearl"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oracle replay: act on the stored state, re-observe with the factual sensor exogenous."""
        from twist.replay import replay_noiseless, replay_pearl

        fn = replay_pearl if semantics == "pearl" else replay_noiseless
        z_cf = fn(exo["state"], actions, self.state)
        single = z_cf.ndim == 2
        zb = z_cf[None] if single else z_cf
        nu = exo["obs_noise"][None] if single else exo["obs_noise"]
        seen = exo["obs_mask"][None] if single else exo["obs_mask"]
        x_cf = self.observation.observe(zb, nu, seen)
        return (x_cf[0], z_cf) if single else (x_cf, z_cf)

    def to_state_plan(self, plan: ActionPlan | None) -> ActionPlan | None:
        """Map a reading made on observed channels to the corresponding state channels.

        Values are passed through ``H⁻¹`` (sensor noise is ignored: the reading treats the
        proposed observation as the intended state value).
        """
        if plan is None:
            return None
        a = plan.action
        idx = self.obs_index
        new: TemporalAction
        if isinstance(a, PointIntervention):
            new = PointIntervention(
                a.t,
                [idx[j] for j in a.targets],
                self.observation.h_inv(np.asarray(a.values)).tolist(),
            )
        elif isinstance(a, ScheduleIntervention):
            new = ScheduleIntervention(
                [
                    (t, [idx[j] for j in s], self.observation.h_inv(np.asarray(v)).tolist())
                    for t, s, v in a.entries
                ]
            )
        elif isinstance(a, IntervalIntervention):
            vals = self.observation.h_inv(np.asarray(a.values, dtype=float))
            new = IntervalIntervention(a.start, a.end, [idx[j] for j in a.targets], vals.tolist())
        elif isinstance(a, ParametricPulse):
            # soft shifts keep their amplitude: exact for identity sensors, first-order otherwise
            new = ParametricPulse(
                a.start, a.duration, [idx[j] for j in a.targets], a.amplitude, a.shape
            )
        elif isinstance(a, HighLevelControl):
            return plan  # controls are world-level, not tied to observed channels
        else:
            raise NotImplementedError(
                f"cannot map {a.kind!r} actions through the observation model"
            )
        return ActionPlan(
            new,
            native=plan.native,
            semantics=plan.semantics + "_via_observation",
            constraints=plan.constraints,
            adapter=plan.adapter,
        )

    # ------------------------------------------------------------------ persistence
    def card(self) -> dict[str, Any]:
        sc = self.state.card()
        return {
            "world_id": self.world_id,
            "world_family": self.family,
            "world_seed": self.seed,
            "k": self.k,
            "k_state": self.k_state,
            "L": self.L,
            "T": self.T,
            "label": self.label.to_dict(),
            "variable_names": self.variable_names,
            "assumptions": self.assumptions,
            "roles": self.roles,
            "observation": self.observation.to_dict(),
            "state_card": sc,
            "mechanism_hash": sc["mechanism_hash"],
        }

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"graph": self.state.graph, **self.state.params()}
        np.savez(d / "mechanism.npz", **arrays)
        write_json(d / "world_card.json", self.card())
        return d

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> ObservedSCM:
        from twist.worlds.registry import WORLD_FAMILIES

        sc = card["state_card"]
        state = WORLD_FAMILIES[sc["world_family"]].from_saved(sc, arrays)
        return cls(card["world_id"], state, ObservationModel(**card["observation"]))

    @classmethod
    def load(cls, directory: str | Path) -> ObservedSCM:
        d = Path(directory)
        return cls.from_saved(read_json(d / "world_card.json"), dict(np.load(d / "mechanism.npz")))


def add_latent_confounder(
    world: TemporalSCM,
    targets: list[int],
    strength: float = 0.5,
    persistence: float = 0.8,
    seed: int = 0,
) -> TemporalSCM:
    """Append a confounder channel ``c`` (AR(1), no parents) driving ``targets`` at lag 1.

    Only for linear-family worlds (``coef`` of shape ``(L, k, k)``). Hide ``c`` with the observation
    model to obtain a latent-confounded stress test.
    """
    from twist.worlds.linear_var import LinearVAR, rescale_to_radius

    if not isinstance(world, LinearVAR):
        raise NotImplementedError("latent confounders are implemented for linear-family worlds")
    rng = np.random.default_rng(seed)
    L, k, _ = world.coef.shape
    coef = np.zeros((L, k + 1, k + 1))
    coef[:, :k, :k] = world.coef
    coef[0, k, k] = persistence
    for t in targets:
        coef[0, t, k] = rng.choice([-1, 1]) * strength * rng.uniform(0.5, 1.0)
    coef = rescale_to_radius(coef, max(0.95, persistence), "cap")
    g = np.zeros((k + 1, k + 1, L), bool)
    g[:k, :k] = world.graph
    g[:, k, 0] = np.abs(coef[0, :, k]) > 0
    new = type(world).from_saved(
        {**world.card(), "k": k + 1, "variable_names": [*world.variable_names, "u_confounder"]},
        {"graph": g, "coef": coef, **{n: v for n, v in world.params().items() if n != "coef"}},
    )
    new.label = world.label
    new.roles = {**world.roles, "latent_confounder": k, "confounded": list(targets)}
    new.actionable = list(range(k)) if world.actionable is None else list(world.actionable)
    return new


def make_statespace(
    world_id: str,
    seed: int,
    base_family: str = "linear",
    base_params: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    latent_confounder: bool = False,
    noise: InnovationProcess | None = None,
    label: LabelFunction | None = None,
) -> ObservedSCM:
    """Build a partially observed world around a base state world.

    With ``latent_confounder=True`` (linear base only) a hidden AR(1) channel drives the label
    channel and the distractor, so the distractor becomes spuriously predictive of the label.
    """
    from twist.worlds.registry import make_world

    base = make_world(base_family, f"{world_id}__state", seed, **(base_params or {}))
    if noise is not None:
        base.noise = noise
    if label is not None:
        base.label = label
    obs = dict(observation or {})
    if latent_confounder:
        targets = [base.roles.get("label_channel", 0)]
        if base.roles.get("distractor") is not None:
            targets.append(base.roles["distractor"])
        base = add_latent_confounder(base, targets, seed=seed)
        obs["hidden"] = sorted({*obs.get("hidden", []), base.k - 1})
    return ObservedSCM(world_id, base, ObservationModel(**obs))
