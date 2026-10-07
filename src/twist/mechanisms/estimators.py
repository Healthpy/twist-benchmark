"""Learned transition models from observational trajectories (mechanism-access regimes R3/R4).

* ``fit_masked_linear`` (R3, linear families): least squares restricted to the known graph.
* ``fit_lasso_var`` (R4, structure-aware): per-target Lasso on lagged features. The support is the
  estimated graph ``Ĝ``.
* ``fit_neural`` (R4 baseline transition predictor, or R3 with ``mask``): linear skip + MLP on the
  lag window. With a mask, each output only sees its known parents.

Every estimate gets Gaussian innovations with the residual scale and a placeholder label: the
explainer learns dynamics, not the benchmark's label rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from twist.utils.backend import Array, is_torch
from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import ThresholdLabel
from twist.worlds.linear_var import LinearVAR


def lagged_design(x: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``past.reshape(L·k)`` (``past[:, l] = x_{t−l−1}``) and targets ``x_t`` for ``t ≥ L``."""
    _B, T, k = x.shape
    windows = np.stack([x[:, t - L : t][:, ::-1] for t in range(L, T)], axis=1)
    return windows.reshape(-1, L * k), x[:, L:].reshape(-1, k)


def _estimate_common(
    template: TemporalSCM, name: str, residual: np.ndarray, L: int, k: int, graph: np.ndarray
) -> dict[str, Any]:
    return {
        "world_id": f"{template.world_id}__est-{name}",
        "seed": template.seed,
        "k": k,
        "L": L,
        "T": template.T,
        "graph": graph,
        "noise": InnovationProcess("gaussian", float(np.std(residual))),
        "label": ThresholdLabel(0, -1, threshold=0.0),  # placeholder: estimates carry no label rule
        "burn_in": 0,
        "variable_names": list(template.variable_names),
        "assumptions": {
            "additive_noise": True,
            "observed_state": True,
            "confounding": False,
            "positive_lags": True,
            "exact_replay": True,
            "estimated": True,
        },
    }


def _coef_from_flat(W: np.ndarray, L: int, k: int) -> np.ndarray:
    """``W`` is (k_out, L·k_in) over ``past.reshape``; return ``coef`` (L, k_out, k_in)."""
    return W.reshape(k, L, k).transpose(1, 0, 2)


def fit_masked_linear(
    x: np.ndarray,
    L: int,
    graph: np.ndarray,
    template: TemporalSCM,
    ridge: float = 1e-6,
    name: str = "masked-ols",
) -> LinearVAR:
    """OLS of each channel on its known parents only (``graph[i, j, l]``)."""
    k = x.shape[-1]
    X, Y = lagged_design(x, L)
    mask = np.asarray(graph, bool).transpose(0, 2, 1).reshape(k, L * k)  # (i, l*k + j)
    W = np.zeros((k, L * k))
    for i in range(k):
        cols = np.flatnonzero(mask[i])
        if cols.size:
            A = X[:, cols]
            W[i, cols] = np.linalg.solve(A.T @ A + ridge * np.eye(cols.size), A.T @ Y[:, i])
    coef = _coef_from_flat(W, L, k)
    resid = Y - X @ W.T
    est = LinearVAR(
        coef=coef, **_estimate_common(template, name, resid, L, k, np.asarray(graph, bool))
    )
    return est


def fit_lasso_var(
    x: np.ndarray,
    L: int,
    template: TemporalSCM,
    alpha: float | None = None,
    threshold: float = 1e-2,
    seed: int = 0,
) -> LinearVAR:
    """Per-target Lasso (``alpha`` by BIC on the LARS path when ``None``); ``Ĝ`` = support above
    ``threshold``. BIC-Lasso keeps tiny noise coefficients, so the default cut is 0.01 (true
    coefficients in the suites are an order of magnitude larger)."""
    from sklearn.linear_model import Lasso, LassoLarsIC

    k = x.shape[-1]
    X, Y = lagged_design(x, L)
    W = np.zeros((k, L * k))
    for i in range(k):
        model = LassoLarsIC(criterion="bic") if alpha is None else Lasso(alpha=alpha)
        W[i] = model.fit(X, Y[:, i]).coef_
    W[np.abs(W) < threshold] = 0.0
    coef = _coef_from_flat(W, L, k)
    graph = (np.abs(coef) > 0).transpose(1, 2, 0)
    resid = Y - X @ W.T
    return LinearVAR(coef=coef, **_estimate_common(template, "lasso-var", resid, L, k, graph))


class _TransitionNet(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in: int, k: int, hidden: int, mask: np.ndarray | None):
        super().__init__()
        self.linear = nn.Linear(d_in, k)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, k)
        m = torch.ones(k, d_in) if mask is None else torch.as_tensor(mask, dtype=torch.float64)
        self.register_buffer("mask", m)
        self.masked = mask is not None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.masked:
            return self.linear(z) + self.fc2(torch.tanh(self.fc1(z)))
        # each output i sees only its parents: run the network on masked copies of the input
        zi = z[:, None, :] * self.mask[None]  # (B, k, d)
        lin = (zi * self.linear.weight[None]).sum(-1) + self.linear.bias
        h = torch.tanh(torch.einsum("bkd,hd->bkh", zi, self.fc1.weight) + self.fc1.bias)
        out = torch.einsum("bkh,kh->bk", h, self.fc2.weight) + self.fc2.bias
        return lin + out


@dataclass
class NeuralTransitionSCM(TemporalSCM):
    """Learned transition ``F̂`` (torch module) behind the standard world interface."""

    family: ClassVar[str] = "estimated_neural"
    net: Any = field(default=None, repr=False)

    def structural_mean(self, past: Array) -> Array:
        B = past.shape[0]
        if is_torch(past):
            return self.net(past.reshape(B, self.L * self.k))
        with torch.no_grad():
            z = torch.as_tensor(
                np.ascontiguousarray(past).reshape(B, self.L * self.k), dtype=torch.float64
            )
            return self.net(z).numpy()

    def params(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> NeuralTransitionSCM:
        raise NotImplementedError("estimated mechanisms are refit, not persisted")


def fit_neural(
    x: np.ndarray,
    L: int,
    template: TemporalSCM,
    hidden: int = 64,
    epochs: int = 40,
    lr: float = 3e-3,
    batch_size: int = 512,
    mask_graph: np.ndarray | None = None,
    seed: int = 0,
    name: str | None = None,
) -> NeuralTransitionSCM:
    """Train ``F̂`` by MSE on one-step targets. ``mask_graph`` (k, k, L) restricts inputs (R3)."""
    torch.manual_seed(seed)
    k = x.shape[-1]
    X, Y = lagged_design(x, L)
    mask = None
    if mask_graph is not None:
        mask = np.asarray(mask_graph, bool).transpose(0, 2, 1).reshape(k, L * k).astype(float)
    net = _TransitionNet(L * k, k, hidden, mask).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i : i + batch_size]
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        resid = (net(Xt) - Yt).numpy()
    graph = np.ones((k, k, L), bool) if mask_graph is None else np.asarray(mask_graph, bool)
    name = name or ("neural-masked" if mask_graph is not None else "neural")
    return NeuralTransitionSCM(net=net, **_estimate_common(template, name, resid, L, k, graph))


ESTIMATORS = ("masked", "lasso", "neural", "pcmci", "varlingam", "corrupted")
DISCOVERY_ESTIMATORS = ("pcmci", "varlingam", "corrupted")


def mechanism_label(spec: dict[str, Any]) -> str:
    """Report label of an estimator spec; discovery settings are part of it, so that a sweep over
    e.g. PCMCI significance levels yields one row per level (no ``:`` — labels are split on it)."""
    kind = spec.get("estimator", "lasso")
    if kind == "pcmci":
        test = spec.get("test", "parcorr")
        lab = f"pcmci-a{float(spec.get('alpha', 0.01)):g}" + (
            "" if test == "parcorr" else f"-{test}"
        )
    elif kind == "varlingam":
        lab = "varlingam"
    elif kind == "corrupted":
        lab = f"corrupted-m{float(spec.get('missing', 0)):g}-s{float(spec.get('spurious', 0)):g}"
    elif kind == "masked" and spec.get("fit"):
        return f"masked-{spec['fit']}"
    else:
        return kind
    return lab + ("-neural" if spec.get("fit", "linear") == "neural" else "")


def estimate_mechanism(
    spec: dict[str, Any],
    x: np.ndarray,
    template: Any,
    known_graph: np.ndarray | None,
    seed: int = 0,
) -> TemporalSCM:
    """Fit the estimator named in ``spec["estimator"]`` on observational trajectories ``x``.

    ``masked`` needs ``known_graph`` (R3): linear least squares for linear-family templates, a
    graph-masked neural transition otherwise. ``lasso`` and ``neural`` use no structure (R4).
    Discovery estimators (``pcmci``, ``varlingam``; ``corrupted`` = controlled error on the true
    graph) learn ``Ĝ`` and fit restricted to it with ``fit: linear`` (masked least squares, the
    default) or ``fit: neural`` (masked neural transition); the family is not used (R4). The
    discovery record is attached as ``discovery_info``.
    """
    kind = spec.get("estimator", "lasso")
    L = int(spec.get("L", getattr(template, "L", 1)))
    if kind == "masked":
        if known_graph is None:
            raise ValueError("the masked estimator needs the known graph (R3)")
        state = getattr(template, "state", template)
        fit = spec.get("fit")  # None: by family (R3 knows the transition family)
        linear = isinstance(state, LinearVAR) and type(state).family in ("linear", "spring")
        if fit == "linear" or (fit is None and linear):
            return fit_masked_linear(x, L, known_graph, template)
        return fit_neural(
            x,
            L,
            template,
            mask_graph=known_graph,
            seed=seed,
            hidden=int(spec.get("hidden", 64)),
            epochs=int(spec.get("epochs", 40)),
        )
    if kind in DISCOVERY_ESTIMATORS:
        from twist.mechanisms.discovery import discover

        graph, info = discover(spec, x, L, known_graph, seed=seed)
        name = mechanism_label(spec)
        mech: Any
        if spec.get("fit", "linear") == "neural":
            mech = fit_neural(
                x,
                L,
                template,
                hidden=int(spec.get("hidden", 64)),
                epochs=int(spec.get("epochs", 40)),
                mask_graph=graph,
                seed=seed,
                name=name,
            )
        else:
            mech = fit_masked_linear(x, L, graph, template, name=name)
        mech.discovery_info = info
        return mech
    if kind == "lasso":
        return fit_lasso_var(x, L, template, alpha=spec.get("alpha"), seed=seed)
    if kind == "neural":
        return fit_neural(
            x,
            L,
            template,
            hidden=int(spec.get("hidden", 64)),
            epochs=int(spec.get("epochs", 40)),
            seed=seed,
        )
    raise ValueError(f"unknown estimator {kind!r}; choose from {ESTIMATORS}")
