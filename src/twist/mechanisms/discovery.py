"""Causal discovery of the lagged graph from observational trajectories (regime R4).

A discovery estimator learns ``Ĝ`` without any structural knowledge, then the mechanism is fitted
restricted to ``Ĝ`` with the same masked fitter that R3 uses on the true graph. The R3-vs-R4
comparison at a fixed fitter therefore isolates the cost of not knowing the graph.

Graph convention (as everywhere in TWIST): ``G[i, j, l−1] = True`` ⇔ ``X^j_{t−l} → X^i_t``.
TWIST mechanisms are strictly lagged; instantaneous edges a method reports are counted in the
discovery info but never enter the fitted mechanism.

* ``discover_pcmci`` — PCMCI (Runge et al., 2019) via Tigramite (optional extra ``discovery``):
  PC-stable parent preselection, then momentary conditional independence (MCI) tests, lags
  ``1 … L``. Tests: ``parcorr`` (linear partial correlation), ``robust_parcorr`` (rank-based),
  ``cmiknn`` (nonparametric, slow: subsample units). Significance ``alpha`` controls the
  sparsity/recall trade-off and is the sweep parameter.
* ``discover_varlingam`` — VAR-LiNGAM (Hyvärinen et al., 2010), implemented in-house (the
  ``lingam`` package would downgrade numpy/scipy): OLS VAR, DirectLiNGAM (Shimizu et al., 2011;
  pairwise likelihood-ratio measure of Hyvärinen & Smith, 2013) on the residuals for the
  instantaneous matrix ``B0``, lagged effects ``B_l = (I − B0) M_l``, adaptive-Lasso pruning
  (BIC) of each variable on its instantaneous predecessors and all lags. Needs non-Gaussian
  innovations for ``B0``; the lagged graph is identified either way.
* ``corrupt_graph`` — controlled graph error (benchmark-side, like misspecification): delete a
  fraction of the true cross edges and add a fraction of spurious ones, for dose–response curves of
  recourse quality against graph error.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

TESTS = ("parcorr", "robust_parcorr", "cmiknn")


def _subsample(x: np.ndarray, max_units: int | None, seed: int) -> np.ndarray:
    if max_units is None or len(x) <= max_units:
        return x
    idx = np.sort(np.random.default_rng(seed).choice(len(x), max_units, replace=False))
    return x[idx]


# ------------------------------------------------------------------------------------- PCMCI
def discover_pcmci(
    x: np.ndarray,
    L: int,
    alpha: float = 0.01,
    test: str = "parcorr",
    pc_alpha: float | None = None,
    max_units: int | None = 200,
    contemporaneous: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """PCMCI on ``x`` (B, T, k), units pooled as independent realisations of one process.

    Lags ``1 … L`` by default (TWIST mechanisms are lagged). ``contemporaneous=True`` also tests
    lag 0, so that spurious instantaneous links (e.g. from latent confounders) are counted."""
    try:
        from tigramite.data_processing import DataFrame
        from tigramite.pcmci import PCMCI
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError("PCMCI needs tigramite: pip install -e '.[discovery]'") from exc
    if test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr

        cit: Any = ParCorr(significance="analytic")
    elif test == "robust_parcorr":
        from tigramite.independence_tests.robust_parcorr import RobustParCorr

        cit = RobustParCorr(significance="analytic")
    elif test == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn

        cit = CMIknn(significance="shuffle_test", seed=seed)
    else:
        raise ValueError(f"unknown test {test!r}; choose from {TESTS}")
    t0 = time.perf_counter()
    xs = _subsample(x, max_units, seed)
    df = DataFrame({i: u for i, u in enumerate(xs)}, analysis_mode="multiple")
    res = PCMCI(dataframe=df, cond_ind_test=cit, verbosity=0).run_pcmci(
        tau_min=0 if contemporaneous else 1, tau_max=L, pc_alpha=pc_alpha, alpha_level=alpha
    )
    g = res["graph"]  # g[i, j, tau] == "-->": X^i_{t−tau} → X^j_t
    k = x.shape[-1]
    G = np.zeros((k, k, L), bool)
    for i, j, tau in zip(*np.nonzero(g == "-->"), strict=True):
        if tau >= 1:
            G[j, i, tau - 1] = True
    # contemporaneous links (each appears in both directions of the lag-0 slice); None if untested
    inst = int(np.sum(g[:, :, 0] != "")) // 2 if contemporaneous else None
    return G, {
        "method": "pcmci",
        "test": test,
        "alpha": alpha,
        "units": len(xs),
        "n_instantaneous": inst,
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------------- VAR-LiNGAM
def _entropy(u: np.ndarray) -> float:
    """Maximum-entropy approximation of differential entropy (Hyvärinen, 1998)."""
    k1, k2, gamma = 79.047, 7.4129, 0.37457
    return float(
        (1 + np.log(2 * np.pi)) / 2
        - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2
        - k2 * (np.mean(u * np.exp(-(u**2) / 2))) ** 2
    )


def _std(u: np.ndarray) -> np.ndarray:
    return (u - u.mean()) / (u.std() + 1e-12)


def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    return xi - np.cov(xi, xj, bias=True)[0, 1] / (np.var(xj) + 1e-12) * xj


def direct_lingam_order(U: np.ndarray) -> list[int]:
    """Causal order of the columns of ``U`` (n, k) by DirectLiNGAM with the pairwise
    likelihood-ratio measure: repeatedly pick the most exogenous variable and regress it out."""
    X = U - U.mean(0)
    remaining = list(range(X.shape[1]))
    order: list[int] = []
    while remaining:
        if len(remaining) == 1:
            order.append(remaining[0])
            break
        scores = []
        for i in remaining:
            xi = _std(X[:, i])
            m = 0.0
            for j in remaining:
                if i == j:
                    continue
                xj = _std(X[:, j])
                ri_j, rj_i = _residual(xi, xj), _residual(xj, xi)
                d = (_entropy(xj) + _entropy(_std(ri_j))) - (_entropy(xi) + _entropy(_std(rj_i)))
                m += min(0.0, d) ** 2
            scores.append(-m)
        root = remaining[int(np.argmax(scores))]
        order.append(root)
        remaining.remove(root)
        for j in remaining:
            X[:, j] = _residual(X[:, j], X[:, root])
    return order


def _adaptive_lasso(Z: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    from sklearn.linear_model import LassoLarsIC

    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    ys = y - y.mean()
    w = np.linalg.lstsq(Zs, ys, rcond=None)[0]
    scale = np.abs(w) ** gamma
    coef = LassoLarsIC(criterion="bic").fit(Zs * scale, ys).coef_ * scale
    return coef  # standardized coefficients


def discover_varlingam(
    x: np.ndarray,
    L: int,
    threshold: float = 1e-2,
    max_units: int | None = 400,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    from twist.mechanisms.estimators import lagged_design

    t0 = time.perf_counter()
    xs = _subsample(x, max_units, seed)
    past, target = lagged_design(xs, L)  # past[:, l*k + j] = x^j_{t−l−1}
    k = x.shape[-1]
    design = np.c_[past, np.ones(len(past))]
    U = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]  # VAR(L) residuals
    order = direct_lingam_order(U)
    G = np.zeros((k, k, L), bool)
    n_inst = 0
    for pos, i in enumerate(order):
        preds = order[:pos]
        Z = np.c_[target[:, preds], past] if preds else past
        coef = _adaptive_lasso(Z, target[:, i])
        n_inst += int(np.sum(np.abs(coef[: len(preds)]) > threshold))
        lag = np.abs(coef[len(preds) :]).reshape(L, k) > threshold  # [l, j]
        G[i] = lag.T
    return G, {
        "method": "varlingam",
        "causal_order": order,
        "units": len(xs),
        "n_instantaneous": n_inst,
        "seconds": time.perf_counter() - t0,
    }


# -------------------------------------------------------------------------- controlled error
def corrupt_graph(
    G: np.ndarray, missing: float = 0.0, spurious: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Delete ``missing`` × (true cross edges) and add ``spurious`` × (true cross edges) wrong ones,
    at random lags. Self-dependencies are kept (they are not part of the summary graph). Additions
    are capped by the number of absent cross edges, so dense worlds get fewer spurious edges than
    requested; ``discover`` records the achieved counts."""
    rng = np.random.default_rng(seed)
    out = np.asarray(G, bool).copy()
    k, _, L = out.shape
    cross = ~np.eye(k, dtype=bool)
    summary = out.any(axis=2) & cross
    edges = np.argwhere(summary)
    n_true = len(edges)
    for i, j in edges[rng.permutation(n_true)[: round(missing * n_true)]]:
        out[i, j, :] = False
    absent = np.argwhere(~summary & cross)
    for i, j in absent[rng.permutation(len(absent))[: round(spurious * n_true)]]:
        out[i, j, rng.integers(0, L)] = True
    return out


def discover(
    spec: dict[str, Any], x: np.ndarray, L: int, known_graph: np.ndarray | None, seed: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    kind = spec["estimator"]
    if kind == "pcmci":
        return discover_pcmci(
            x,
            L,
            alpha=float(spec.get("alpha", 0.01)),
            test=str(spec.get("test", "parcorr")),
            pc_alpha=spec.get("pc_alpha"),
            max_units=spec.get("max_units", 200),
            contemporaneous=bool(spec.get("contemporaneous", False)),
            seed=seed,
        )
    if kind == "varlingam":
        return discover_varlingam(
            x,
            L,
            threshold=float(spec.get("threshold", 1e-2)),
            max_units=spec.get("max_units", 400),
            seed=seed,
        )
    if kind == "corrupted":
        if known_graph is None:
            raise ValueError("graph corruption starts from the true graph")
        m, s = float(spec.get("missing", 0.0)), float(spec.get("spurious", 0.0))
        G = corrupt_graph(known_graph, m, s, seed=seed)
        cross = ~np.eye(G.shape[0], dtype=bool)
        true_s, est_s = np.asarray(known_graph, bool).any(2) & cross, G.any(2) & cross
        return G, {
            "method": "corrupted",
            "missing": m,
            "spurious": s,
            "n_true_edges": int(true_s.sum()),
            "n_deleted": int((true_s & ~est_s).sum()),
            "n_added": int((est_s & ~true_s).sum()),
            "n_instantaneous": 0,
        }
    raise ValueError(f"unknown discovery estimator {kind!r}")
