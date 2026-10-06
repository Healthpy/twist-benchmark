"""Wachter-style gradient counterfactual on the full trajectory (in-house reference, `WachterReference`).

Minimises ``CE(h(x̂), y*) + λ·mean|x̂ − x|`` over all entries in classifier-normalized units. It
returns a trajectory only (no action); the evaluator applies registered diagnostic readings.
The draft's results use the third-party ``cfts`` implementation; that adapter lands in Week 5.
"""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from twist.evaluation.result_schema import Explanation, ExplanationStatus
from twist.explainers.base import Explainer, ExplainerContext


class WachterTrajectoryCF(Explainer):
    name = "WachterReference"
    uses_gradients = True
    native_actions = False
    min_regime = "R0"

    def explain(self, factuals, factual_ids, targets, ctx: ExplainerContext) -> list[Explanation]:
        clf = ctx.classifier
        steps = int(self.params.get("steps", 300))
        lam = float(self.params.get("lambda_prox", 0.05))
        lr = float(self.params.get("lr", 0.05))
        torch.manual_seed(self.seed)
        start = time.perf_counter()
        x = torch.as_tensor(factuals, dtype=clf.dtype)
        scale = torch.as_tensor(
            factuals.reshape(-1, factuals.shape[-1]).std(0) + 1e-6, dtype=clf.dtype
        )
        y = torch.as_tensor(targets, dtype=torch.long)
        delta = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            loss = F.cross_entropy(clf.forward(x + delta * scale), y) + lam * delta.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x_hat = (x + delta * scale).numpy()
            flipped = clf.forward(torch.as_tensor(x_hat)).argmax(-1).numpy() == targets
        runtime = (time.perf_counter() - start) / max(len(factuals), 1)
        return [
            Explanation(
                explainer=self.name,
                factual_id=fid,
                target=int(t),
                proposed_trajectory=x_hat[n],
                action=None,
                mechanism=None,
                mechanism_access_regime=ctx.regime,
                status=ExplanationStatus.OK if flipped[n] else ExplanationStatus.NO_TARGET_FLIP,
                selection_objective={
                    "loss": "CE + lambda*mean|delta|",
                    "lambda_prox": lam,
                    "steps": steps,
                    "lr": lr,
                },
                explainer_seed=self.seed,
                runtime_seconds=runtime,
            )
            for n, (fid, t) in enumerate(zip(factual_ids, targets, strict=True))
        ]
