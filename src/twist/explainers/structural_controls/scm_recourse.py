"""PearlSCMRecourse and NoiselessSCMRecourse (draft Appendix E), generalized to every native
action form.

Both receive the oracle mechanism (R5), keep the factual prefix, optimise the parameters ``θ`` of
one native action starting at a candidate ``t0`` with Adam on
``λ_pred·CE(h(x̂), y*) + λ_prox·||θ||²``, and let the mechanism produce the rest of the trajectory —
with the factual innovations (Pearl) or with zero innovations after the action (noiseless).

``action_form`` (default ``"point"``, the draft's control):

* ``point``    — ``do(X^S_{t0} = x_{t0} + θ)``; ``D = 1`` under its own semantics.
* ``interval`` — ``do(X^S_{t0:t0+Δ} = x_{t0:t0+Δ} + θ)``.
* ``pulse``    — soft pulse with per-channel amplitude ``θ`` and fixed ``pulse_shape``.
* ``control``  — high-level control ``u_{t0:t0+Δ} = θ`` through the world's control interface.

``S`` is the world's actionable set; values and controls are projected onto their bounds after
every step, so the emitted actions always pass validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from twist.actions.domain_actions import HighLevelControl
from twist.actions.interval import IntervalIntervention, ParametricPulse
from twist.actions.observed_variable import PointIntervention
from twist.actions.schemas import ActionPlan, ActionSpace, ControlChannel, TemporalAction
from twist.evaluation.result_schema import Explanation, ExplanationStatus
from twist.explainers.base import Explainer, ExplainerContext
from twist.replay.abduction import abduct_noise
from twist.replay.deterministic_rollout import noiseless_eps
from twist.replay.pearl_replay import replay
from twist.worlds.base_scm import TemporalSCM
from twist.worlds.shifts import perturb_mechanism

ACTION_FORMS = ("point", "interval", "pulse", "control")


@dataclass
class ActionForm:
    """Differentiable parameterization of one native action form at a fixed start time."""

    kind: str
    t0: int
    space: ActionSpace
    duration: int = 1
    pulse_shape: str = "raised_cosine"
    control: ControlChannel | None = None
    max_abs: float | None = (
        None  # amplitude bound on the action (|value − factual|, |shift|, |u·load|)
    )

    def __post_init__(self) -> None:
        if self.kind not in ACTION_FORMS:
            raise ValueError(f"action_form must be one of {ACTION_FORMS}, got {self.kind!r}")
        if self.kind == "control" and self.control is None:
            raise ValueError("action_form='control' needs a control channel declared by the world")
        self.targets = np.asarray(self.space.actionable, int)
        self.end = (
            self.t0 + 1 if self.kind == "point" else min(self.t0 + self.duration, self.space.T)
        )
        self.n = self.end - self.t0

    @property
    def first_time(self) -> int:
        return self.t0 + (self.control.delay if self.control is not None else 0)

    def theta_shape(self, B: int) -> tuple[int, ...]:
        return {
            "point": (B, len(self.targets)),
            "interval": (B, self.n, len(self.targets)),
            "pulse": (B, len(self.targets)),
            "control": (B, self.n),
        }[self.kind]

    def _pulse_profile(self) -> np.ndarray:
        return ParametricPulse(self.t0, self.n, [0], 1.0, self.pulse_shape).profile()

    def build(self, theta: torch.Tensor, x: torch.Tensor):
        """Dense ``(mask, value, shift)`` for a batch; ``value``/``shift`` carry gradients."""
        B, T, k = x.shape
        mask = np.zeros((B, T, k), bool)
        value = torch.zeros_like(x)
        shift = torch.zeros_like(x)
        tg = torch.as_tensor(self.targets)
        if self.kind in ("point", "interval"):
            mask[:, self.t0 : self.end, self.targets] = True
            th = theta if self.kind == "interval" else theta[:, None, :]
            value[:, self.t0 : self.end, tg] = x[:, self.t0 : self.end][:, :, tg] + th
        elif self.kind == "pulse":
            phi = torch.as_tensor(self._pulse_profile(), dtype=x.dtype)
            shift[:, self.t0 : self.end, tg] = phi[None, :, None] * theta[:, None, :]
        else:
            assert self.control is not None
            load = torch.as_tensor(self.control.loadings, dtype=x.dtype)
            for j in range(self.n):
                t = self.t0 + j + self.control.delay
                if t < T:
                    shift[:, t] = shift[:, t] + theta[:, j, None] * load[None, :]
        return mask, value, shift

    @torch.no_grad()
    def project(self, theta: torch.Tensor, x: torch.Tensor) -> None:
        """Keep set values inside the action-space value bounds and controls inside theirs."""
        if self.kind in ("point", "interval"):
            base = x[:, self.t0 : self.end][:, :, self.targets]
            if self.kind == "point":
                base = base[:, 0]
            lo = self.space.lower
            hi = self.space.upper
            if lo is not None or hi is not None:
                lo_t = (
                    None
                    if lo is None
                    else torch.as_tensor(np.asarray(lo)[self.targets], dtype=x.dtype)
                )
                hi_t = (
                    None
                    if hi is None
                    else torch.as_tensor(np.asarray(hi)[self.targets], dtype=x.dtype)
                )
                theta.copy_(torch.clamp(base + theta, lo_t, hi_t) - base)
        elif self.kind == "control":
            assert self.control is not None
            theta.clamp_(float(self.control.lower), float(self.control.upper))
        if self.max_abs is not None:
            bound = self.max_abs
            if self.kind == "control" and self.control is not None:
                bound = self.max_abs / max(float(np.abs(self.control.loadings).max()), 1e-12)
            theta.clamp_(-bound, bound)

    def to_action(self, theta: np.ndarray, x: np.ndarray) -> TemporalAction:
        tg = self.targets.tolist()
        if self.kind == "point":
            return PointIntervention(self.t0, tg, (x[self.t0, self.targets] + theta).tolist())
        if self.kind == "interval":
            vals = x[self.t0 : self.end][:, self.targets] + theta
            return IntervalIntervention(self.t0, self.end, tg, vals.tolist())
        if self.kind == "pulse":
            return ParametricPulse(self.t0, self.n, tg, theta.tolist(), self.pulse_shape)
        assert self.control is not None
        return HighLevelControl(self.control.name, self.t0, self.end, theta.tolist())


class _SCMRecourse(Explainer):
    native_actions = True
    min_regime = "R5"
    is_control = True
    uses_gradients = True
    exact_world_only = True
    semantics = "pearl"
    default_prox = 0.1
    default_steps = 500
    default_form = "point"
    coordinates = (
        "state"  # "observed": actions are planned on observed channels (mapped by the evaluator)
    )

    def __init__(self, seed: int = 0, **params):
        super().__init__(seed=seed, **params)
        form = params.get("action_form", self.default_form)
        if params.get("label"):
            self.name = str(params["label"])
        elif form != "point":  # keep reports unambiguous when several forms run side by side
            self.name = f"{type(self).name}[{form}]"

    _mechanism_label = "oracle"

    def _mechanism(self, world: TemporalSCM) -> TemporalSCM:
        """Mechanism used to abduct and roll out proposals (the oracle for genuine controls)."""
        return world

    # ---------------------------------------------------------------- information hooks
    def _resolve_mechanism(self, ctx: ExplainerContext) -> tuple[Any, str]:
        """(mechanism used to abduct and roll out, label). ``None`` = no mechanism (imposed edit)."""
        if ctx.world is None:
            raise PermissionError(f"{self.name} requires oracle mechanism access (R5+)")
        if not ctx.world.assumptions.get("exact_abduction", True):
            raise NotImplementedError(
                f"{self.name} abducts from observations with the oracle mechanism; partially "
                "observed worlds need posterior abduction (estimated-mechanism recourse runs there)"
            )
        return self._mechanism(ctx.world), self._mechanism_label

    def _action_space(self, ctx: ExplainerContext, T: int, k: int) -> ActionSpace:
        assert ctx.world is not None
        return ctx.world.action_space()

    def _limits(self, ctx: ExplainerContext) -> dict[str, Any]:
        """Constraint knowledge the explainer may use when planning (R6 only)."""
        return {}

    def _feasible(
        self, ctx: ExplainerContext, x: np.ndarray, action: Any, x_hat: np.ndarray
    ) -> bool:
        return True

    def _form(
        self, t0: int, space: ActionSpace, limits: dict[str, Any] | None = None
    ) -> ActionForm:
        p = self.params
        limits = limits or {}
        kind = p.get("action_form", self.default_form)
        control = None
        if kind == "control":
            name = p.get("control") or next(iter(space.controls), None)
            if name is None or name not in space.controls:
                raise ValueError(
                    f"control {name!r} not declared; world controls: {sorted(space.controls)}"
                )
            control = space.controls[name]
        duration = int(p.get("duration", 3))
        if limits.get("max_duration") is not None:
            duration = min(duration, int(limits["max_duration"]))
        return ActionForm(
            kind=kind,
            t0=t0,
            space=space,
            duration=duration,
            pulse_shape=p.get("pulse_shape", "raised_cosine"),
            control=control,
            max_abs=limits.get("max_abs"),
        )

    def explain(self, factuals, factual_ids, targets, ctx: ExplainerContext) -> list[Explanation]:
        clf = ctx.classifier
        # the mechanism the explainer *believes*; the benchmark always replays in the true world
        mech, mech_label = self._resolve_mechanism(ctx)
        space = self._action_space(ctx, *factuals.shape[1:])
        limits = self._limits(ctx)
        p = self.params
        fracs = p.get("t0_fracs", [0.25, 0.5])
        steps = int(p.get("steps", self.default_steps))
        lam_pred = float(p.get("lambda_pred", 1.0))
        lam_prox = float(p.get("lambda_prox", self.default_prox))
        lr = float(p.get("lr", 0.05))
        torch.manual_seed(self.seed)

        B, T, _ = factuals.shape
        if limits.get("latest_start") is not None:
            fracs = [f for f in fracs if int(f * T) <= limits["latest_start"]] or [
                limits["latest_start"] / T
            ]
        eps = abduct_noise(factuals, mech) if mech is not None else np.zeros_like(factuals)
        best: list[tuple[float, ActionForm, np.ndarray, np.ndarray] | None] = [None] * B
        start = time.perf_counter()
        # all units try every candidate t0; units that never flip get one prediction-only retry
        for prox, retry in ((lam_prox, False), (0.0, True)):
            todo = [n for n in range(B) if not retry or not np.isfinite(best[n][0])]  # type: ignore[index]
            if not todo:
                break
            for frac in fracs:
                form = self._form(int(frac * T), space, limits)
                e = eps
                if self.semantics == "noiseless":
                    e = noiseless_eps(eps, np.full(B, form.first_time))
                thetas, x_hats, flipped = self._optimise(
                    mech,
                    clf,
                    form,
                    factuals[todo],
                    e[todo],
                    targets[todo],
                    steps,
                    lam_pred,
                    prox,
                    lr,
                )
                for j, n in enumerate(todo):
                    norm = float(np.linalg.norm(thetas[j])) if flipped[j] else np.inf
                    # prefer flipping, then feasible (under the explainer's own rollout), then small
                    feas = bool(flipped[j]) and self._feasible(
                        ctx, factuals[n], form.to_action(thetas[j], factuals[n]), x_hats[j]
                    )
                    key = (norm if feas else norm + 1e6) if np.isfinite(norm) else np.inf
                    if best[n] is None or key < best[n][0]:  # type: ignore[index]
                        best[n] = (key, form, thetas[j], x_hats[j])
        runtime = (time.perf_counter() - start) / max(B, 1)

        actions = [best[n][1].to_action(best[n][2], factuals[n]) for n in range(B)]  # type: ignore[index]
        p_model = self._predicted_target_prob(mech, clf, factuals, actions, targets, space)
        out = []
        for n in range(B):
            norm, form, _, x_hat = best[n]  # type: ignore[misc]
            action = actions[n]
            out.append(
                Explanation(
                    explainer=self.name,
                    factual_id=factual_ids[n],
                    target=int(targets[n]),
                    proposed_trajectory=x_hat,
                    action=ActionPlan(
                        action,
                        native=True,
                        semantics=f"native_{form.kind}_{self.semantics}"
                        + ("@observed" if self.coordinates == "observed" else ""),
                    ),
                    declares_actions=True,
                    mechanism=f"{mech_label}:{getattr(mech, 'world_id', 'none')}:{self.semantics}",
                    mechanism_access_regime=ctx.regime,
                    status=ExplanationStatus.OK
                    if np.isfinite(norm)
                    else ExplanationStatus.NO_TARGET_FLIP,
                    selection_objective={
                        "loss": "lambda_pred*CE + lambda_prox*||theta||^2",
                        "action_form": form.kind,
                        "lambda_pred": lam_pred,
                        "lambda_prox": lam_prox,
                        "steps": steps,
                        "t0_candidates": [int(f * T) for f in fracs],
                        "selection": "flipping first, then feasible, then smallest ||theta||",
                        "limits": limits,
                    },
                    candidate_count=2 * len(fracs),
                    uncertainty={}
                    if p_model is None
                    else {
                        "p_target_model": float(p_model[n]),
                        "definition": "P(h(X^A) = y*) under the explainer's mechanism, fresh noise",
                        "samples": int(self.params.get("uncertainty_samples", 16)),
                    },
                    explainer_seed=self.seed,
                    runtime_seconds=runtime,
                )
            )
        return out

    def _predicted_target_prob(self, mech, clf, factuals, actions, targets, space):
        """The explainer's own uncertainty report: how often its action reaches the target under
        its mechanism when future innovations are redrawn (none without a mechanism)."""
        S = int(self.params.get("uncertainty_samples", 16))
        if mech is None or S <= 0:
            return None
        from twist.replay import replay_interventional_distribution

        rng = np.random.default_rng(self.seed)
        sched = [a.compile(space) for a in actions]
        z = replay_interventional_distribution(factuals, sched, mech, S, rng)
        pred = clf.predict(z.reshape(-1, *z.shape[2:])).reshape(S, len(actions))
        return (pred == np.asarray(targets)[None]).mean(axis=0)

    @staticmethod
    def _optimise(world, clf, form: ActionForm, x, e, y, steps, lam_pred, lam_prox, lr):
        dtype = clf.dtype
        x_t = torch.as_tensor(x, dtype=dtype)
        e_t = torch.as_tensor(e, dtype=dtype)
        y_t = torch.as_tensor(y, dtype=torch.long)
        theta = torch.zeros(form.theta_shape(len(x)), dtype=dtype, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)

        def rollout(th):
            mask, value, shift = form.build(th, x_t)
            if world is None:  # no mechanism: the action is imposed, nothing propagates
                return torch.where(torch.as_tensor(mask), value, x_t + shift)
            return replay(world, e_t, mask, value, shift)

        def penalty(th):
            return (th**2).reshape(len(th), -1).sum(1).mean()

        for _ in range(steps):
            x_hat = rollout(theta)
            loss = lam_pred * F.cross_entropy(clf.forward(x_hat), y_t) + lam_prox * penalty(theta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            form.project(theta, x_t)
        with torch.no_grad():
            x_hat = rollout(theta)
            flipped = (clf.forward(x_hat).argmax(-1) == y_t).numpy()
        return theta.detach().numpy(), x_hat.numpy(), flipped


class PearlSCMRecourse(_SCMRecourse):
    name = "PearlSCMRecourse"
    semantics = "pearl"
    default_prox = 0.1
    default_steps = 500


class NoiselessSCMRecourse(_SCMRecourse):
    name = "NoiselessSCMRecourse"
    semantics = "noiseless"
    default_prox = 0.5
    default_steps = 300


class PlausibleButMechanismWrongEdit(_SCMRecourse):
    """Negative control: a native point action optimised and rolled out under a *misspecified*
    mechanism (same graph, perturbed strengths; ``perturb_mechanism``).

    Proposals keep the factual prefix and the abducted innovations, so they look like ordinary
    world trajectories, but the true world does not propagate the declared action that way. Metrics
    must separate plausible appearance from correct causal propagation.
    """

    name = "PlausibleButMechanismWrongEdit"
    semantics = "pearl"
    default_prox = 0.1
    default_steps = 300
    _mechanism_label = "misspecified"

    def _mechanism(self, world: TemporalSCM) -> TemporalSCM:
        return perturb_mechanism(
            world,
            rel_scale=float(self.params.get("rel_scale", 0.5)),
            seed=int(self.params.get("mechanism_seed", self.seed)),
        )
