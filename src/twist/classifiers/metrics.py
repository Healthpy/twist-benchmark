"""Classifier quality: accuracy and calibration (Brier, expected calibration error, NLL)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def expected_calibration_error(p1: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE over equal-width bins of the predicted-class confidence."""
    conf = np.maximum(p1, 1 - p1)
    pred = (p1 > 0.5).astype(int)
    bins = np.minimum((conf - 0.5) / 0.5 * n_bins, n_bins - 1).astype(int)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def classifier_report(clf: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    p1 = torch.softmax(torch.as_tensor(clf.logits(x)), -1).numpy()[:, 1]
    y = np.asarray(y)
    eps = 1e-12
    return {
        "n": len(y),
        "accuracy": float(((p1 > 0.5).astype(int) == y).mean()),
        "brier": float(((p1 - y) ** 2).mean()),
        "ece": expected_calibration_error(p1, y),
        "nll": float(-(y * np.log(p1 + eps) + (1 - y) * np.log(1 - p1 + eps)).mean()),
        "positive_rate": float((p1 > 0.5).mean()),
    }
