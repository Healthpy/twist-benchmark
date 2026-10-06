"""Shared evaluation budgets (plan §Week 5, task 4).

Every explainer gets the same per-unit limits on model queries and wall-clock time. Third-party
adapters route all classifier calls through :class:`BudgetedModel`, which counts the samples
passed to the classifier and raises :class:`BudgetExceeded` as soon as a limit is crossed, so the
unit ends in a structured ``budget_exceeded`` or ``timeout`` record instead of hanging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from twist.classifiers.models import TorchClassifier


@dataclass
class Budget:
    max_queries: int | None = 200_000  # samples passed through the classifier per unit
    time_limit_s: float | None = 120.0  # wall clock per unit
    max_reference: int = 500  # reference trajectories handed to retrieval-based methods

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Budget:
        return cls(**(d or {}))


class BudgetExceeded(RuntimeError):
    def __init__(self, kind: str, used: float, limit: float):
        self.kind = kind  # "queries" | "time"
        super().__init__(f"{kind} budget exceeded: {used:.4g} > {limit:.4g}")


class BudgetedModel(nn.Module):
    """Channel-first float32 view of a TWIST classifier with query/time accounting.

    Third-party methods call ``model(x)`` with ``x`` of shape ``(B, C, L)`` in float32. This
    module converts to the classifier's ``(B, T, k)`` float64 input, returns float32 logits, and
    keeps gradients flowing for gradient-based methods.
    """

    def __init__(self, clf: TorchClassifier, budget: Budget):
        super().__init__()
        self.inner = clf.model  # registered so `next(model.parameters())` works upstream
        self.budget = budget
        self.queries = 0
        self.t0 = time.perf_counter()

    def start_unit(self) -> None:
        self.queries = 0
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        self.queries += int(x.shape[0])
        b = self.budget
        if b.max_queries is not None and self.queries > b.max_queries:
            raise BudgetExceeded("queries", self.queries, b.max_queries)
        if b.time_limit_s is not None and self.elapsed > b.time_limit_s:
            raise BudgetExceeded("time", self.elapsed, b.time_limit_s)
        z = self.inner(x.transpose(1, 2).to(torch.float64))
        return z.to(torch.float32)
