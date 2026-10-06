"""Week 4: factors, integrity checks, family diversity, suite building and data IO."""

import json

import numpy as np
import pytest

from twist.evaluation.paired_runner import generate_world_data
from twist.worlds import make_linear_var, make_world
from twist.worlds.factors import APPLICABLE, DEFAULTS, FAMILIES, resolve_world
from twist.worlds.integrity import (
    ancestors,
    check_family_diversity,
    check_world,
    mechanism_distance,
)
from twist.worlds.io import load_world_data, save_world_data
from twist.worlds.suite import SuiteConfig, build_suite, load_manifest, load_suite_world


def _units(cfg):
    d = generate_world_data(cfg)
    x = np.concatenate([d["x_train"], d["x_test"]])
    exo = {k: np.concatenate([d["exo_train"][k], d["exo_test"][k]]) for k in d["exo_train"]}
    return d["world"], x, exo


@pytest.mark.parametrize("family", FAMILIES)
def test_default_factor_world_passes_integrity(family):
    world, x, exo = _units(resolve_world(family, {}, f"{family}_t", 7, k=6, T=40, n_samples=400))
    rep = check_world(world, x, exo)
    assert rep["passed"], [c for c in rep["checks"] if not c["passed"]]


def test_resolve_world_rejects_unknown_levels():
    with pytest.raises(ValueError):
        resolve_world("linear", {"lag": "L7"}, "w", 0)
    assert set(APPLICABLE["statespace"]) <= set(DEFAULTS)


def test_integrity_flags_routeless_and_unbalanced_worlds():
    w = make_linear_var("bad", 0, k=4, T=30, density=0.0)  # empty graph: no route into x0
    x, eta = w.simulate(300, np.random.default_rng(0))
    w.label.fit(x)
    rep = {c["check"]: c["passed"] for c in check_world(w, x, {"state": x, "eta": eta})["checks"]}
    assert not rep["causal_route_graph"] and not rep["causal_route_effect"]
    w.label.threshold = float(np.quantile(w.label.raw(x), 0.95))
    rep = {c["check"]: c["passed"] for c in check_world(w, x, {"state": x, "eta": eta})["checks"]}
    assert not rep["label_balance"]


def test_ancestors_hand_case():
    G = np.zeros((4, 4, 1), bool)
    G[0, 1, 0] = G[1, 2, 0] = True  # 2 → 1 → 0;  3 isolated
    assert ancestors(G, [0]) == {1, 2}


def test_mechanism_distance_and_diversity():
    a, b = make_world("linear", "a", 1, k=5), make_world("linear", "b", 2, k=5)
    assert mechanism_distance(a, a) == 0.0 and mechanism_distance(a, b) > 0.05
    assert mechanism_distance(a, make_world("linear", "c", 3, k=6)) == float("inf")
    assert check_family_diversity([a, b])["passed"]
    assert not check_family_diversity([a, make_world("linear", "a2", 1, k=5)])["passed"]


def test_world_data_io_roundtrip(tmp_path):
    x = np.ones((3, 4, 2))
    exo = {
        "state": np.zeros((3, 4, 3)),
        "eta": np.ones((3, 4, 3)),
        "obs_mask": np.ones((3, 4, 2), bool),
    }
    save_world_data(tmp_path, x, np.array([0, 1, 0]), exo, 2)
    d = load_world_data(tmp_path)
    assert d["n_train"] == 2 and set(d["exo"]) == {"state", "eta", "obs_mask"}
    np.testing.assert_array_equal(d["exo"]["state"], exo["state"])
    save_world_data(tmp_path, x, np.array([0, 1, 0]), {"state": x, "eta": exo["eta"]}, 2)
    np.testing.assert_array_equal(load_world_data(tmp_path)["exo"]["state"], x)


def _tiny_suite(tmp_path, name, monkeypatch=None, secret=None):
    if monkeypatch is not None:
        if secret:
            monkeypatch.setenv("TWIST_SUITE_SECRET", secret)
        else:
            monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    cfg = SuiteConfig(
        name=name,
        output_dir=str(tmp_path),
        families=["linear", "statespace"],
        worlds_per_split={"dev": 1, "val": 1, "test": 1, "challenge": 1},
        k=5,
        T=30,
        n_samples=300,
    )
    return build_suite(cfg)


def test_build_suite_layout_hiding_and_reports(tmp_path, monkeypatch):
    root = _tiny_suite(tmp_path, "s1", monkeypatch)
    m = load_manifest(root)
    assert m["passed"] and len(m["worlds"]) == 8 and m["hidden_seed_source"] == "public"
    assert (root / "splits" / "test.txt").read_text().split() == [
        "linear_test_000",
        "statespace_test_000",
    ]
    for e in m["worlds"]:
        hidden = e["split"] in ("test", "challenge")
        assert e["visibility"] == ("hidden" if hidden else "visible")
        assert ("seed" in e) != hidden and ("factors" in e) != hidden  # no seed leak
        card = json.loads((root / e["path"] / "world_card.json").read_text())
        assert ("roles" in card) != hidden
        assert (root / e["path"] / "data.npz").exists()
    with pytest.raises(PermissionError):
        load_suite_world(root, "linear_test_000")
    assert (
        load_suite_world(root, "linear_test_000", allow_hidden=True).world_id == "linear_test_000"
    )
    assert load_suite_world(root, "linear_dev_000").world_id == "linear_dev_000"
    for f in ("integrity_report.md", "DATASET_CARD.md", "hidden/manifest_private.json"):
        assert (root / f).exists()
    with pytest.raises(FileExistsError):
        _tiny_suite(tmp_path, "s1")


def test_secret_changes_only_hidden_seeds(tmp_path, monkeypatch):
    pub = json.loads(
        (_tiny_suite(tmp_path, "a", monkeypatch) / "hidden/manifest_private.json").read_text()
    )
    sec = json.loads(
        (
            _tiny_suite(tmp_path, "b", monkeypatch, "s3cret") / "hidden/manifest_private.json"
        ).read_text()
    )
    seeds = lambda m: {e["world_id"]: e["seed"] for e in m["worlds"]}  # noqa: E731
    a, b = seeds(pub), seeds(sec)
    assert a["linear_dev_000"] == b["linear_dev_000"]
    assert a["linear_test_000"] != b["linear_test_000"]
