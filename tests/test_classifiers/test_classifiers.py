"""Week 5: classifier suite (architectures, calibration metrics, persistence, gradients)."""

import numpy as np
import pytest
import torch

from twist.classifiers.metrics import classifier_report, expected_calibration_error
from twist.classifiers.models import FEATURES, TorchClassifier
from twist.classifiers.train import train_classifier

ARCHS = ["gru", "lstm", "tcn", "transformer", "logreg"]


@pytest.fixture(scope="module")
def data(linear_world):
    world, x, _ = linear_world
    y = world.label(x)
    return x[:240], y[:240], x[240:], y[240:]


@pytest.mark.parametrize("arch", ARCHS)
def test_architecture_trains_predicts_and_roundtrips(arch, data, tmp_path):
    xtr, ytr, xte, yte = data
    clf = train_classifier(
        xtr, ytr, xte, yte, architecture=arch, hidden=8, layers=2, epochs=2, seed=3
    )
    z = clf.logits(xte)
    assert z.shape == (len(xte), 2) and np.all(np.isfinite(z))
    assert {"test_accuracy", "test_ece", "test_brier"} <= set(clf.spec)
    clf.save(tmp_path)
    np.testing.assert_allclose(TorchClassifier.load(tmp_path).logits(xte), z)
    x = torch.tensor(xte[:2], dtype=torch.float64, requires_grad=True)
    clf.forward(x)[:, 1].sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0  # gradient-based explainers can query it


@pytest.mark.parametrize("arch", ["tcn", "transformer", "logreg"])
def test_training_is_deterministic(arch, data):
    xtr, ytr, xte, yte = data
    a = train_classifier(xtr, ytr, xte, yte, architecture=arch, hidden=8, epochs=1, seed=5)
    b = train_classifier(xtr, ytr, xte, yte, architecture=arch, hidden=8, epochs=1, seed=5)
    np.testing.assert_array_equal(a.logits(xte), b.logits(xte))


def test_logreg_feature_names(data):
    xtr, ytr, xte, yte = data
    clf = train_classifier(xtr, ytr, xte, yte, architecture="logreg")
    assert len(clf.model.feature_names) == xtr.shape[-1] * len(FEATURES)


def test_calibration_metrics_hand_case():
    p = np.array([0.9, 0.9, 0.1, 0.6])
    y = np.array([1, 0, 0, 1])
    # bins: conf .9 (2 units, acc .5), conf .9 for p=.1 → same bin; conf .6 (acc 1)
    assert expected_calibration_error(p, y, n_bins=5) == pytest.approx(
        0.75 * abs(2 / 3 - 0.9) + 0.25 * 0.4
    )

    class Fixed:
        def logits(self, x):
            return np.log(np.stack([1 - p, p], 1))

    r = classifier_report(Fixed(), np.zeros((4, 1, 1)), y)
    assert r["accuracy"] == 0.75 and r["brier"] == pytest.approx(((p - y) ** 2).mean())
