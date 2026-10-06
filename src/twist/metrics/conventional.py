"""Model-relative counterfactual quality: validity, proximity, sparsity, plausibility."""

from __future__ import annotations

import numpy as np

from twist.actions.trajectory_adapters import DEFAULT_CHANGE_TOL, changed_mask


def first_change(
    x: np.ndarray, x_hat: np.ndarray, tol: float = DEFAULT_CHANGE_TOL
) -> tuple[np.ndarray, np.ndarray]:
    """First changed timestep ``t0`` per unit and whether any entry changed.

    Batched ``(B, T, k)`` inputs. Following the draft, ``t0 = T-1`` when nothing changed.
    """
    m = np.asarray(np.any(changed_mask(x, x_hat, tol), axis=2))
    has = np.asarray(np.any(m, axis=1))
    T = x.shape[1]
    return np.where(has, m.argmax(axis=1), T - 1), has


def validity(pred: np.ndarray, target: np.ndarray | int) -> np.ndarray:
    return (np.asarray(pred) == np.asarray(target)).astype(float)


def proximity_lp(x: np.ndarray, x_hat: np.ndarray, p: int = 1) -> np.ndarray:
    d = (x_hat - x).reshape(x.shape[0], -1)
    return np.linalg.norm(d, ord=p, axis=1)


def normalized_proximity(
    x: np.ndarray, x_hat: np.ndarray, iqr: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """``d_norm = (1/Tk) Σ_{t,i} |x̂ - x| / (IQR(X_i) + ε)`` with per-channel IQR ``(k,)``."""
    return (np.abs(x_hat - x) / (np.asarray(iqr)[None, None, :] + eps)).mean(axis=(1, 2))


def channel_iqr(population: np.ndarray) -> np.ndarray:
    flat = population.reshape(-1, population.shape[-1])
    q75, q25 = np.percentile(flat, [75, 25], axis=0)
    return q75 - q25


def sparsity(
    x: np.ndarray, x_hat: np.ndarray, tol: float = DEFAULT_CHANGE_TOL
) -> dict[str, np.ndarray]:
    """Fractions left *unchanged* (higher = sparser), at entry, channel and timestep level."""
    m = changed_mask(x, x_hat, tol)
    return {
        "unchanged_entries": 1.0 - m.mean(axis=(1, 2)),
        "unchanged_channels": 1.0 - m.any(axis=1).mean(axis=1),
        "unchanged_timesteps": 1.0 - m.any(axis=2).mean(axis=1),
    }


def temporal_smoothness(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Mean squared second difference of the edit ``x̂ - x`` (lower = smoother)."""
    d = x_hat - x
    if d.shape[1] < 3:
        return np.zeros(d.shape[0])
    return (np.diff(d, n=2, axis=1) ** 2).mean(axis=(1, 2))


def ood_score(train: np.ndarray, x_hat: np.ndarray, seed: int = 0) -> np.ndarray:
    """Isolation Forest decision score on flattened trajectories (higher = more in-distribution)."""
    return OODScorer(train, seed).score(x_hat)


def logit_margin(logits: np.ndarray, target: np.ndarray | int) -> np.ndarray:
    """Binary decision margin ``z[y*] − z[1−y*]`` (positive ⇔ predicted target class)."""
    z = np.asarray(logits)
    y = np.broadcast_to(np.asarray(target), (len(z),)).astype(int)
    rows = np.arange(len(z))
    return z[rows, y] - z[rows, 1 - y]


def dtw_distance(
    x: np.ndarray, x_hat: np.ndarray, scale: np.ndarray | None = None, window: int | None = None
) -> np.ndarray:
    """Multivariate DTW between factual and proposal, per unit ``(B, T, k)``.

    Local cost is the mean absolute difference across channels (divided by ``scale`` per channel,
    e.g. the IQR); the accumulated path cost is divided by ``T`` so values are comparable with the
    normalized proximity. ``window`` is an optional Sakoe–Chiba band.
    """
    B, T, k = x.shape
    s = np.ones(k) if scale is None else np.asarray(scale) + 1e-8
    out = np.empty(B)
    w = T if window is None else max(int(window), 0)
    for b in range(B):
        a, c = x[b] / s, x_hat[b] / s
        cost = np.abs(a[:, None, :] - c[None, :, :]).mean(axis=2)
        acc = np.full((T + 1, T + 1), np.inf)
        acc[0, 0] = 0.0
        for i in range(1, T + 1):
            lo, hi = max(1, i - w), min(T, i + w)
            for j in range(lo, hi + 1):
                acc[i, j] = cost[i - 1, j - 1] + min(
                    acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
                )
        out[b] = acc[T, T] / T
    return out


def interval_sparsity(
    x: np.ndarray, x_hat: np.ndarray, tol: float = DEFAULT_CHANGE_TOL
) -> dict[str, np.ndarray]:
    """Number of contiguous edited intervals: per-channel runs and timestep-level runs."""
    m = changed_mask(x, x_hat, tol)
    pad = np.zeros_like(m[:, :1])
    onsets = m & ~np.concatenate([pad, m[:, :-1]], axis=1)
    mt = np.any(m, axis=2)
    t_onsets = mt & ~np.concatenate([np.zeros_like(mt[:, :1]), mt[:, :-1]], axis=1)
    return {"channel_segments": onsets.sum(axis=(1, 2)), "time_segments": t_onsets.sum(axis=1)}


def diversity(proposals: np.ndarray, scale: np.ndarray | None = None) -> float:
    """Mean pairwise normalized L1 distance within a set of proposals ``(M, T, k)`` for one unit.

    NaN for fewer than two proposals. Used across seeds/candidates of one explainer.
    """
    p = np.asarray(proposals, float)
    if len(p) < 2:
        return float("nan")
    s = np.ones(p.shape[-1]) if scale is None else np.asarray(scale) + 1e-8
    p = p / s
    d = [np.abs(p[i] - p[j]).mean() for i in range(len(p)) for j in range(i + 1, len(p))]
    return float(np.mean(d))


class OODScorer:
    """Isolation Forest fit once on training trajectories; ``score`` is higher = more typical."""

    def __init__(self, train: np.ndarray, seed: int = 0):
        from sklearn.ensemble import IsolationForest

        self.forest = IsolationForest(random_state=seed).fit(train.reshape(len(train), -1))

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.forest.decision_function(np.asarray(x).reshape(len(x), -1))
