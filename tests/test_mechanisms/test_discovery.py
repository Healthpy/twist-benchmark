"""Causal discovery as an R4 mechanism source: PCMCI, VAR-LiNGAM, controlled graph corruption."""

import numpy as np
import pytest

from twist.mechanisms.discovery import corrupt_graph, direct_lingam_order, discover_varlingam
from twist.mechanisms.estimators import estimate_mechanism, mechanism_label
from twist.mechanisms.quality import graph_recovery
from twist.worlds import make_linear_var


@pytest.fixture(scope="module")
def lin():
    w = make_linear_var("disc", 7, k=5, T=40, density=0.3, spectral_radius=0.9, rescale="exact")
    x, _ = w.simulate(400, np.random.default_rng(0))
    w.label.fit(x)
    return w, x


def test_lag_metrics_hand_case():
    true = np.zeros((3, 3, 2), bool)
    true[0, 1, 0] = True  # x1_{t-1} → x0
    true[2, 0, 1] = True  # x0_{t-2} → x2
    true[1, 1, 0] = True  # self-dependence: ignored
    est = np.zeros((3, 3, 2), bool)
    est[0, 1, 0] = True  # right edge, right lag
    est[2, 0, 0] = True  # right edge, wrong lag
    est[1, 2, 0] = True  # spurious
    g = graph_recovery(est, true)
    assert g["precision"] == pytest.approx(2 / 3) and g["recall"] == 1.0 and g["shd"] == 1
    assert g["lag_precision"] == pytest.approx(1 / 3) and g["lag_recall"] == pytest.approx(0.5)
    assert g["lag_accuracy"] == pytest.approx(0.5)
    one_lag = est[:, :, :1]  # graphs with different max lags are padded
    assert graph_recovery(one_lag, true)["lag_accuracy"] == pytest.approx(0.5)


def test_corrupt_graph_is_controlled(lin):
    w, _ = lin
    G = w.dependency_graph()
    n_cross = int((G.any(2) & ~np.eye(w.k, dtype=bool)).sum())
    assert np.array_equal(corrupt_graph(G, 0, 0), G)
    drop = graph_recovery(corrupt_graph(G, missing=0.5, seed=1), G)
    assert drop["precision"] == 1.0 and drop["recall"] == pytest.approx(
        1 - round(0.5 * n_cross) / n_cross
    )
    add = graph_recovery(corrupt_graph(G, spurious=1.0, seed=1), G)
    assert add["recall"] == 1.0 and add["precision"] == pytest.approx(0.5)
    diag = np.eye(w.k, dtype=bool)[:, :, None]
    assert np.array_equal(corrupt_graph(G, 1.0, 0.0) & diag, G & diag)  # self-lags kept


def test_direct_lingam_recovers_a_causal_order():
    rng = np.random.default_rng(0)
    e = rng.laplace(size=(4000, 4))
    X = np.zeros_like(e)
    X[:, 2] = e[:, 2]
    X[:, 0] = 0.8 * X[:, 2] + e[:, 0]
    X[:, 3] = -0.7 * X[:, 0] + e[:, 3]
    X[:, 1] = 0.6 * X[:, 3] + 0.5 * X[:, 2] + e[:, 1]
    assert direct_lingam_order(X) == [2, 0, 3, 1]


def test_varlingam_recovers_the_lagged_graph(lin):
    w, x = lin
    G, info = discover_varlingam(x, w.L, seed=0)
    assert graph_recovery(G, w.dependency_graph())["f1"] >= 0.8
    assert info["n_instantaneous"] == 0  # TWIST worlds have no instantaneous effects


def test_pcmci_recovers_the_lagged_graph(lin):
    pytest.importorskip("tigramite")
    from twist.mechanisms.discovery import discover_pcmci

    w, x = lin
    G, info = discover_pcmci(x, w.L, alpha=0.01, max_units=150, seed=0)
    g = graph_recovery(G, w.dependency_graph())
    assert g["f1"] >= 0.8 and info["units"] == 150
    loose, _ = discover_pcmci(x, w.L, alpha=0.3, max_units=150, seed=0)
    assert loose.sum() >= G.sum()  # a looser significance level never removes links here


def test_discovery_estimator_fits_on_the_discovered_graph(lin):
    w, x = lin
    G = w.dependency_graph()
    spec = {"estimator": "corrupted", "missing": 0.5, "spurious": 0.5}
    mech = estimate_mechanism(spec, x, w, G, seed=3)
    assert mech.discovery_info["method"] == "corrupted"
    support = np.abs(mech.coef).sum(0) > 0  # coef (L, k, k) → (k, k) [i, j]
    assert not np.any(support & ~mech.graph.any(2))  # no coefficient outside Ĝ
    assert mechanism_label(spec) == "corrupted-m0.5-s0.5"
    assert (
        mechanism_label({"estimator": "pcmci", "alpha": 0.01, "fit": "neural"})
        == "pcmci-a0.01-neural"
    )
    assert mechanism_label({"estimator": "masked", "fit": "linear"}) == "masked-linear"
    assert mechanism_label({"estimator": "lasso"}) == "lasso"
    true_fit = estimate_mechanism({"estimator": "masked", "fit": "linear"}, x, w, G)
    same = estimate_mechanism({"estimator": "corrupted"}, x, w, G)
    np.testing.assert_allclose(same.coef, true_fit.coef)  # zero corruption = R3 masked-linear


@pytest.mark.smoke
def test_mini_discovery_report(tmp_path, monkeypatch):
    import json

    from twist.evaluation.discovery_report import DiscoveryConfig, run_discovery
    from twist.worlds.suite import SuiteConfig, build_suite

    monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    suite = build_suite(
        SuiteConfig(
            name="s9",
            output_dir=str(tmp_path / "suites"),
            families=["linear"],
            worlds_per_split={"dev": 2},
            k=4,
            T=20,
            n_samples=300,
        )
    )
    rec = {"steps": 10, "t0_fracs": [0.5, 0.9], "duration": 3}

    def row(regime, label, mech=None):
        e = {
            "name": "EstimatedActionRecourse",
            "regime": regime,
            "params": {**rec, "label": f"ActionRecourse[{label}]"},
        }
        return {**e, "mechanism": mech} if mech else e

    cfg = DiscoveryConfig.model_validate(
        {
            "name": "disc9",
            "output_dir": str(tmp_path / "disc"),
            "suite": str(suite),
            "worlds": ["linear_dev_000", "linear_dev_001"],
            "classifiers": [{"architecture": "logreg"}],
            "explainers": [
                row("R3", "R3 masked-linear", {"estimator": "masked", "fit": "linear"}),
                row("R4", "R4 varlingam", {"estimator": "varlingam"}),
                row("R4", "R4 corrupted m=1", {"estimator": "corrupted", "missing": 1.0}),
                row("R4", "R4 corrupted s=1", {"estimator": "corrupted", "spurious": 1.0}),
            ],
            "explainer_seeds": [1],
            "n_factuals": 5,
            "n_boot": 50,
            "hypothesis_row": "ActionRecourse[R4 varlingam]",
            "evaluation": {"readings": ["first_change_atomic"], "compute_dtw": False},
        }
    )
    path = run_discovery(cfg)
    text = path.read_text()
    for s in (
        "Graph recovery against the true graph",
        "cost of not knowing the graph",
        "every_discovery_graph_scored | **PASS**",
        "cost_of_graph_is_paired | **PASS**",
        "corruption_spans_graph_error | **PASS**",
    ):
        assert s in text, s
    res = json.loads(path.with_suffix(".json").read_text())
    rows = {r["method"]: r for r in res["rows"]}
    assert rows["ActionRecourse[R4 corrupted m=1]"]["graph_recall"][0] == 0.0
    assert rows["ActionRecourse[R3 masked-linear]"]["graph_f1"][0] == 1.0
    assert len(res["points"]) == 6  # 3 R4 rows × 2 worlds


def test_r5_estimated_recourse_skips_partially_observed_worlds(tiny_classifier):
    from twist.explainers.mechanism_access.regimes import build_context
    from twist.explainers.registry import build_explainer
    from twist.worlds import make_statespace

    ss = make_statespace("ss9", 0, base_params=dict(k=4, T=20), observation=dict(noise_scale=0.1))
    x, _ = ss.simulate_units(5, np.random.default_rng(0))
    ctx = build_context("R5", tiny_classifier, ss)
    with pytest.raises(NotImplementedError, match="posterior abduction"):
        build_explainer("EstimatedActionRecourse", steps=2).explain(
            x, list("abcde"), np.ones(5, int), ctx
        )
