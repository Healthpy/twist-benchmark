"""Classifier training with a fixed seed and a recorded classifier card."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from twist.classifiers.metrics import classifier_report
from twist.classifiers.models import (
    FeatureLogisticClassifier,
    TorchClassifier,
    build_model,
    temporal_features,
)
from twist.utils.logging import get_logger
from twist.utils.seeds import seed_everything

log = get_logger(__name__)


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    architecture: str = "gru",
    hidden: int = 32,
    layers: int = 1,
    epochs: int = 10,
    lr: float = 3e-3,
    batch_size: int = 128,
    seed: int = 101,
    heads: int = 4,
    kernel: int = 3,
    l2: float = 1.0,
    **_: Any,
) -> TorchClassifier:
    """Train one classifier. ``logreg`` is fit by scikit-learn (inverse regularization ``l2``);
    every other architecture by Adam on cross-entropy."""
    seed_everything(seed)
    spec: dict[str, Any] = {
        "architecture": architecture,
        "k": int(x_train.shape[-1]),
        "T": int(x_train.shape[-2]),
        "hidden": hidden,
        "layers": layers,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "classifier_seed": seed,
        "heads": heads,
        "kernel": kernel,
        "l2": l2,
    }
    model = build_model(spec)
    model.set_normalization(x_train)
    start = time.perf_counter()
    if architecture == "logreg":
        _fit_logistic(model, x_train, y_train, l2, seed)  # type: ignore[arg-type]
        return _finish(model, spec, x_test, y_test, start)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(x_train, dtype=torch.float64)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i : i + batch_size]
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log.debug("epoch %d loss %.4f", epoch, total / len(xt))
    return _finish(model, spec, x_test, y_test, start)


def _fit_logistic(
    model: FeatureLogisticClassifier, x: np.ndarray, y: np.ndarray, l2: float, seed: int
) -> None:
    from sklearn.linear_model import LogisticRegression

    with torch.no_grad():
        f = temporal_features(model.normalize(torch.as_tensor(x, dtype=torch.float64)))
        model.f_mu.copy_(f.mean(0))
        model.f_sd.copy_(f.std(0) + 1e-6)
        fs = ((f - model.f_mu) / model.f_sd).numpy()
    lr = LogisticRegression(C=l2, max_iter=2000, random_state=seed).fit(fs, y)
    with torch.no_grad():
        model.linear.weight.copy_(torch.as_tensor(lr.coef_, dtype=torch.float64))
        model.linear.bias.copy_(torch.as_tensor(lr.intercept_, dtype=torch.float64))


def _finish(
    model: torch.nn.Module,
    spec: dict[str, Any],
    x_test: np.ndarray,
    y_test: np.ndarray,
    start: float,
) -> TorchClassifier:
    clf = TorchClassifier(model.eval(), spec)
    report = classifier_report(clf, x_test, y_test)
    spec["test_accuracy"] = report["accuracy"]
    spec["test_brier"] = report["brier"]
    spec["test_ece"] = report["ece"]
    spec["test_nll"] = report["nll"]
    spec["train_seconds"] = time.perf_counter() - start
    log.info(
        "%s classifier: test accuracy %.3f, ECE %.3f",
        spec["architecture"],
        report["accuracy"],
        report["ece"],
    )
    return clf
