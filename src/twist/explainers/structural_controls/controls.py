"""Structural controls besides the SCM-recourse family (Week 3).

* ``NoActionAbstain`` — abstention control (denominators).
* ``FullScheduleOracle`` — zero replay error with a near-complete schedule.
* ``AdversarialTrajectoryEdit`` — high model validity without mechanism consistency.

``PlausibleButMechanismWrongEdit`` lives in ``scm_recourse`` because it reuses that optimizer.
Controls test the scoring semantics; they are never ranked against explainers.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from twist.actions.observed_variable import ScheduleIntervention
from twist.actions.schemas import ActionPlan
from twist.evaluation.result_schema import Explanation, ExplanationStatus
from twist.explainers.base import Explainer
from twist.replay.pearl_replay import replay_pearl


class NoActionAbstain(Explainer):
    """Always abstains. Tests that abstentions shrink the scorable denominator instead of being
    scored as successful zero-effect interventions."""

    name = "NoActionAbstain"
    native_actions = True
    is_control = True

    def explain(self, factuals, factual_ids, targets, ctx):
        return [
            Explanation(
                self.name,
                fid,
                int(t),
                proposed_trajectory=None,
                action=None,
                status=ExplanationStatus.ABSTAIN,
                mechanism_access_regime=ctx.regime,
                declares_actions=True,
                explainer_seed=self.seed,
            )
            for fid, t in zip(factual_ids, targets, strict=True)
        ]


class FullScheduleOracle(Explainer):
    """Positive control for replay, negative control for compactness.

    Imposes, from ``t_start`` on, every actionable entry of the nearest training trajectory whose
    *world* label is the target (oracle labels, R5). Its proposal is the Pearl replay of that
    schedule, so replay error is exactly zero, while the schedule imposes nearly every timestep
    (high ``D`` and action size). It shows that low replay error alone does not certify a compact
    action.
    """

    name = "FullScheduleOracle"
    native_actions = True
    min_regime = "R5"
    is_control = True
    uses_reference_data = True
    exact_world_only = True

    def explain(self, factuals, factual_ids, targets, ctx):
        if ctx.world is None or ctx.train_x is None:
            raise PermissionError(f"{self.name} requires the oracle world and training data (R5+)")
        world, train = ctx.world, ctx.train_x
        if not world.assumptions.get("exact_abduction", True):
            raise NotImplementedError(f"{self.name} needs an observed-state world")
        space = world.action_space()
        T = world.T
        t_start = int(float(self.params.get("t_start_frac", 0.0)) * T)
        acts = np.asarray(space.actionable, int)
        train_labels = world.label(train)
        scale = train.reshape(-1, train.shape[-1]).std(0) + 1e-8
        out = []
        for x, fid, y in zip(factuals, factual_ids, targets, strict=True):
            t0 = time.perf_counter()
            pool = train[train_labels == y]
            if len(pool) == 0:
                out.append(
                    Explanation(
                        self.name,
                        fid,
                        int(y),
                        None,
                        status=ExplanationStatus.NO_VALID_ACTION,
                        declares_actions=True,
                        mechanism_access_regime=ctx.regime,
                    )
                )
                continue
            dist = (np.abs(pool[:, t_start:] - x[None, t_start:]) / scale).mean(axis=(1, 2))
            donor = pool[int(np.argmin(dist))]
            mask = np.zeros_like(x, bool)
            mask[t_start:, acts] = True
            action = ScheduleIntervention.from_dense(mask, donor)
            proposal = replay_pearl(x, action, world)
            out.append(
                Explanation(
                    explainer=self.name,
                    factual_id=fid,
                    target=int(y),
                    proposed_trajectory=proposal,
                    action=ActionPlan(action, native=True, semantics="native_full_schedule"),
                    declares_actions=True,
                    mechanism=f"oracle:{world.world_id}:pearl",
                    mechanism_access_regime=ctx.regime,
                    selection_objective={
                        "rule": "nearest training trajectory with world label = target",
                        "distance": "mean |x - donor| / channel std over t >= t_start",
                        "t_start": t_start,
                    },
                    candidate_count=len(pool),
                    explainer_seed=self.seed,
                    runtime_seconds=time.perf_counter() - t0,
                )
            )
        return out


class AdversarialTrajectoryEdit(Explainer):
    """Negative control: minimal-budget targeted L∞ PGD on the classifier input (no action).

    Each unit is attacked with an escalating budget ladder ``epsilons`` (multiples of the
    per-channel std, applied to every entry); the smallest budget that flips the classifier is
    kept. These edits flip the model without being generated by any compact action: high model
    validity, low mechanism consistency. The ladder reaches 4 std because robust classifiers can
    need large edits; the selected budget is recorded, so edit size is reported, not assumed small.
    """

    name = "AdversarialTrajectoryEdit"
    native_actions = False
    is_control = True
    uses_gradients = True

    def explain(self, factuals, factual_ids, targets, ctx):
        clf = ctx.classifier
        ladder = [float(e) for e in self.params.get("epsilons", [0.1, 0.25, 0.5, 1.0, 2.0, 4.0])]
        steps = int(self.params.get("steps", 100))
        torch.manual_seed(self.seed)
        t0 = time.perf_counter()
        std = factuals.reshape(-1, factuals.shape[-1]).std(0) + 1e-8
        B = len(factuals)
        x_hat = factuals.copy()
        chosen = np.full(B, np.nan)
        todo = np.arange(B)
        for eps_rel in ladder:
            if todo.size == 0:
                break
            adv, flipped = self._pgd(clf, factuals[todo], targets[todo], eps_rel * std, steps)
            x_hat[todo] = np.where(np.isnan(chosen[todo])[:, None, None], adv, x_hat[todo])
            chosen[todo[flipped]] = eps_rel
            todo = todo[~flipped]
        runtime = (time.perf_counter() - t0) / max(B, 1)
        return [
            Explanation(
                explainer=self.name,
                factual_id=fid,
                target=int(t),
                proposed_trajectory=x_hat[n],
                mechanism_access_regime=ctx.regime,
                status=ExplanationStatus.OK
                if np.isfinite(chosen[n])
                else ExplanationStatus.NO_TARGET_FLIP,
                selection_objective={
                    "loss": "targeted CE, L-inf PGD",
                    "epsilon_ladder": ladder,
                    "epsilon_selected": None if np.isnan(chosen[n]) else float(chosen[n]),
                    "steps": steps,
                    "selection": "smallest budget in the ladder that flips the classifier",
                },
                candidate_count=len(ladder),
                explainer_seed=self.seed,
                runtime_seconds=runtime,
            )
            for n, (fid, t) in enumerate(zip(factual_ids, targets, strict=True))
        ]

    @staticmethod
    def _pgd(clf, x_np, y_np, bound_np, steps):
        x = torch.as_tensor(x_np, dtype=clf.dtype)
        y = torch.as_tensor(y_np, dtype=torch.long)
        bound = torch.as_tensor(bound_np, dtype=clf.dtype)
        alpha = 2.5 * bound / steps
        delta = torch.zeros_like(x, requires_grad=True)
        for _ in range(steps):
            loss = F.cross_entropy(clf.forward(x + delta), y)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= alpha * grad.sign()
                delta.copy_(torch.maximum(torch.minimum(delta, bound), -bound))
        with torch.no_grad():
            adv = (x + delta).numpy()
            flipped = clf.forward(torch.as_tensor(adv)).argmax(-1).numpy() == y_np
        return adv, flipped
