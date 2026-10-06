"""Week 5: third-party baseline adapters, budgets, structured failures, retrieval baseline."""

import numpy as np
import pytest
import torch

from twist.evaluation.result_schema import ExplanationStatus
from twist.explainers.adapters.cfts_baselines import cfts_available
from twist.explainers.budget import Budget, BudgetedModel, BudgetExceeded
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import EXPLAINERS, build_explainer

if not cfts_available():
    pytest.skip("cfts baselines extra not installed", allow_module_level=True)

FAST = {
    "Wachter": {"max_cfs": 40},
    "COMTE": {},
    "M-CELS": {"max_itr": 20},
    "CONFETTI": {"population_size": 8, "max_generations": 2},
    "NativeGuide": {},
    "LatentCF": {"ae_epochs": 2, "max_iter": 10},
    "NUN": {},
}


@pytest.fixture(scope="module")
def setup(linear_world, tiny_classifier):
    world, x, _ = linear_world
    y = world.label(x)
    idx = np.flatnonzero(tiny_classifier.predict(x[200:]) == 0)[:2] + 200
    ctx = build_context(
        "R1",
        tiny_classifier,
        world,
        train_x=x[:200],
        train_y=y[:200],
        budget={"max_queries": 100000, "time_limit_s": 60, "max_reference": 60},
    )
    return world, tiny_classifier, x[idx], [f"u{i}" for i in idx], np.ones(len(idx), int), ctx


def test_budgeted_model_layout_gradients_and_limits(tiny_classifier):
    m = BudgetedModel(tiny_classifier, Budget(max_queries=5, time_limit_s=None))
    x = torch.randn(3, 4, 20, requires_grad=True)  # (B, C, L) float32 as cfts passes it
    z = m(x)
    assert z.dtype == torch.float32 and z.shape == (3, 2) and m.queries == 3
    z.sum().backward()
    assert x.grad is not None
    expected = tiny_classifier.logits(x.detach().numpy().transpose(0, 2, 1))
    np.testing.assert_allclose(z.detach().numpy(), expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(BudgetExceeded):
        m(torch.randn(3, 4, 20))


@pytest.mark.parametrize("name", sorted(FAST))
def test_adapter_outputs_standardized_and_deterministic(setup, name):
    _, clf, xs, ids, tg, ctx = setup
    a = build_explainer(name, seed=7, **FAST[name]).explain(xs, ids, tg, ctx)
    b = build_explainer(name, seed=7, **FAST[name]).explain(xs, ids, tg, ctx)
    assert len(a) == len(xs)
    for e1, e2, x in zip(a, b, xs, strict=True):
        assert isinstance(e1.status, ExplanationStatus) and e1.explainer == name
        assert e1.action is None and not e1.declares_actions  # trajectory-only
        if e1.proposed_trajectory is not None:
            assert e1.proposed_trajectory.shape == x.shape
            np.testing.assert_allclose(e1.proposed_trajectory, e2.proposed_trajectory, atol=1e-6)
            flipped = clf.predict(e1.proposed_trajectory) == 1
            assert (e1.status == ExplanationStatus.OK) == bool(flipped)
        assert e1.status == e2.status


def test_reference_methods_need_r1(setup):
    world, clf, xs, ids, tg, _ = setup
    ctx0 = build_context("R0", clf, world)
    for name in ("COMTE", "NUN"):
        with pytest.raises(PermissionError):
            build_explainer(name).explain(xs, ids, tg, ctx0)


def test_budget_and_error_become_structured_records(setup, monkeypatch):
    world, clf, xs, ids, tg, ctx = setup
    tight = build_context(
        "R1",
        clf,
        world,
        train_x=ctx.train_x,
        train_y=ctx.train_y,
        budget={"max_queries": 1, "max_reference": 60},
    )
    ex = build_explainer("Wachter", seed=1, max_cfs=40).explain(xs, ids, tg, tight)
    assert {e.status for e in ex} == {ExplanationStatus.BUDGET_EXCEEDED}
    assert all(e.proposed_trajectory is None and "budget" in e.diagnostics["error"] for e in ex)

    def boom(*a, **k):
        raise ZeroDivisionError("upstream bug")

    monkeypatch.setattr(EXPLAINERS["COMTE"], "_call", boom)
    ex = build_explainer("COMTE").explain(xs, ids, tg, ctx)
    assert {e.status for e in ex} == {ExplanationStatus.METHOD_ERROR}
    assert "ZeroDivisionError: upstream bug" in ex[0].diagnostics["error"]


def test_nun_returns_a_reference_trajectory_of_the_target_class(setup):
    _, clf, xs, ids, tg, ctx = setup
    ex = build_explainer("NUN", seed=3).explain(xs, ids, tg, ctx)
    for e in ex:
        match = np.all(np.isclose(ctx.train_x, e.proposed_trajectory[None]), axis=(1, 2))
        assert match.any() and clf.predict(e.proposed_trajectory) == 1
