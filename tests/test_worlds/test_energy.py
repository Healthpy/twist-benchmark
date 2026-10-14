"""Semi-synthetic energy worlds (plan §Week 9): real covariates, explicit controls, domain limits."""

import copy

import numpy as np
import pytest
import torch

from twist.actions import HighLevelControl
from twist.actions.feasibility import (
    MaxRateOfChange,
    StateBounds,
    build_constraints,
    check_feasibility,
)
from twist.actions.schemas import DenseSchedule
from twist.replay.abduction import abduct_noise
from twist.replay.pearl_replay import to_dense
from twist.worlds.semi_synthetic import (
    EXOGENOUS,
    GRID,
    HOT_SPOT,
    SOC,
    SOC_HI,
    SOC_LO,
    STRESS,
    ProfileInnovations,
    load_profiles,
    make_energy_grid,
)


@pytest.fixture(scope="module")
def energy():
    w = make_energy_grid("e_test", 11, battery="large", season="winter")
    x, eta = w.simulate(200, np.random.default_rng(0))
    w.label.fit(x)
    return w, x, eta


def test_profiles_are_verified_and_synthetic_matches_layout():
    real = load_profiles("opsd_nl_2016_2019")
    syn = load_profiles("synthetic")
    assert set(real) == set(syn) == {"load", "temp", "ghi", "hour_utc", "month"}
    assert len(real["load"]) == 35064 and abs(real["load"].mean() - 1) < 1e-4
    assert real["ghi"].min() >= 0 and real["hour_utc"].max() == 23


def test_exogenous_channels_are_real_next_hour_windows(energy):
    w, x, eta = energy
    p = load_profiles(w.noise.profiles)
    np.testing.assert_array_equal(x[..., list(EXOGENOUS)], eta[..., list(EXOGENOUS)])
    # each unit's load channel is a contiguous run of the profile starting at 01:00 UTC
    # (next-hour forecast of a window aligned at 00:00 UTC after the burn-in)
    u = x[0, :, 2]
    hits = np.flatnonzero(np.isclose(p["load"], u[0], atol=1e-6))
    starts = [h for h in hits if np.allclose(p["load"][h : h + w.T], u, atol=1e-6)]
    assert starts and p["hour_utc"][starts[0]] == 1
    assert set(p["month"][starts[0] : starts[0] + 1]) <= {11, 12, 1, 2}


def test_abduction_is_exact_and_actions_never_change_the_weather(energy):
    w, x, eta = energy
    np.testing.assert_allclose(abduct_noise(x, w)[:, w.L :], eta[:, w.L :], atol=1e-10)  # roots: x
    act = HighLevelControl("u_dispatch", 30, 34, [w.controls["u_dispatch"].upper] * 4)
    _, s = w.replay_units({"state": x[:5], "eta": eta[:5]}, [act] * 5)
    np.testing.assert_array_equal(s[..., list(EXOGENOUS)], x[:5][..., list(EXOGENOUS)])
    np.testing.assert_array_equal(s[:, :30], x[:5, :30])  # prefix kept
    E = w.phys["e_cap"]
    d_soc = s[:, 30, SOC] - x[:5, 30, SOC]
    np.testing.assert_allclose(d_soc, -w.controls["u_dispatch"].upper / E, atol=1e-12)
    assert np.all(s[:, 30, GRID] < x[:5, 30, GRID])


def test_policy_respects_battery_physics(energy):
    _, x, _ = energy
    soc = x[..., SOC]
    assert soc.min() > SOC_LO - 0.02 and soc.max() < SOC_HI + 0.02
    assert np.abs(np.diff(soc, axis=1)).max() < 0.55  # C-rate 0.5 / efficiency + noise


def test_controls_have_action_card_fields_and_delays(energy):
    w, _, _ = energy
    space = w.action_space()
    d, r = space.controls["u_dispatch"], space.controls["u_dr"]
    assert d.delay == 0 and r.delay == 1 and r.lower == 0.0
    assert d.cost > 0 and r.cost > d.cost and "rebound" in r.side_effects
    s = HighLevelControl("u_dr", 30, 32, [0.1, 0.1]).compile(space)
    assert s.shift[31, GRID] == -0.1 and s.shift[31, 5] == 0.1 and not s.shift[30].any()
    assert space.actionable == [GRID]
    assert "non-actionable" in " ".join(
        __import__("twist.actions", fromlist=["PointIntervention"])
        .PointIntervention(30, [0], [1.0])
        .validate(space)
    )


def test_mechanism_is_differentiable_and_matches_numpy(energy):
    w, x, _ = energy
    past = x[:16, 20:21][:, ::-1].copy()
    pt = torch.tensor(past, requires_grad=True)
    out = w.structural_mean(pt)
    np.testing.assert_allclose(out.detach().numpy(), w.structural_mean(past), atol=1e-10)
    out[:, HOT_SPOT].sum().backward()
    assert torch.isfinite(pt.grad).all() and pt.grad[:, 0, GRID].abs().sum() > 0


def test_graph_roles_and_distractor(energy):
    from twist.worlds.integrity import ancestors

    w, _, _ = energy
    anc = ancestors(w.dependency_graph(), [HOT_SPOT])
    assert GRID in anc and SOC in anc and STRESS not in anc
    assert w.roles["distractor"] == STRESS and w.roles["label_channel"] == HOT_SPOT


def test_save_load_and_noise_shifts_keep_real_covariates(energy, tmp_path):
    from twist.worlds.base_scm import TemporalSCM
    from twist.worlds.shifts import innovation_shift, perturb_mechanism

    w, _, _ = energy
    w.save(tmp_path)
    w2 = TemporalSCM.load(tmp_path)
    assert (
        isinstance(w2.noise, ProfileInnovations)
        and w2.card()["mechanism_hash"] == w.card()["mechanism_hash"]
    )
    a, _ = w.simulate(4, np.random.default_rng(3))
    b, _ = w2.simulate(4, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    s = innovation_shift(w, "uniform")
    assert isinstance(s.noise, ProfileInnovations) and s.noise.kind == "uniform"
    wrong = perturb_mechanism(w, 0.5, seed=1)
    assert wrong.phys != w.phys and set(wrong.phys) == set(w.phys)


def test_energy_factors_resolve_and_pass_integrity():
    from twist.evaluation.paired_runner import generate_world_data
    from twist.worlds.factors import resolve_world
    from twist.worlds.integrity import check_world

    cfg = resolve_world(
        "energy",
        {"pv_penetration": "high", "battery": "small", "energy_label": "evening_peak"},
        "energy_dev_t",
        5,
        n_samples=300,
    )
    assert cfg.label == {"type": "onset", "channel": HOT_SPOT, "t_start": 39, "t_end": 44}
    data = generate_world_data(cfg)
    x = np.concatenate([data["x_train"], data["x_test"]])
    exo = {
        n: np.concatenate([data["exo_train"][n], data["exo_test"][n]]) for n in data["exo_train"]
    }
    report = check_world(data["world"], x, exo)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]


def test_channel_constraints_and_ids():
    c = build_constraints(
        [
            {"name": "state_bounds", "id": "soc_capacity", "channels": [1], "lower": 0, "upper": 1},
            {"name": "max_rate_of_change", "id": "c_rate", "channels": [1], "max_rate": 0.5},
        ]
    )
    assert [k.name for k in c] == ["soc_capacity", "c_rate"]
    assert isinstance(c[0], StateBounds) and isinstance(c[1], MaxRateOfChange)
    traj = np.zeros((5, 2))
    traj[:, 0] = 5.0  # out of [0, 1] but not a selected channel
    traj[:, 1] = [0.2, 0.6, 0.9, 0.9, 0.9]
    empty = DenseSchedule.empty(5, 2)
    assert check_feasibility(c, traj, empty, traj).passed
    traj[3, 1] = 1.2
    rep = check_feasibility(c, traj, empty, traj)
    assert [r.name for r in rep.violations] == ["soc_capacity"]


def test_r6_penalty_respects_soc_capacity(energy):
    """Constraint-aware planning: the same optimizer with the SoC bound in its context keeps the
    battery within capacity where the constraint-blind optimizer does not."""
    from twist.classifiers.train import train_classifier
    from twist.explainers.mechanism_access.regimes import build_context
    from twist.explainers.registry import build_explainer

    w, x, eta = energy
    y = w.label(x)
    clf = train_classifier(x[:150], y[:150], x[150:], y[150:], architecture="logreg", seed=1)
    pick = np.flatnonzero(clf.predict(x) == 1)[:6]
    cons = build_constraints(
        [{"name": "state_bounds", "id": "soc_capacity", "channels": [SOC], "lower": 0, "upper": 1}]
    )
    params = dict(
        action_form="control",
        control="u_dispatch",
        duration=4,
        t0_fracs=[0.5],
        steps=40,
        lambda_prox=0.0,
    )
    space = w.action_space()
    viol = {}
    for name, regime in (("PearlSCMRecourse", "R5"), ("OracleActionRecourse", "R6")):
        ctx = build_context(regime, clf, w, constraints=copy.deepcopy(cons))
        ex = build_explainer(name, seed=1, **params)
        plans = [e.action for e in ex.explain(x[pick], list(map(str, pick)), np.zeros(6, int), ctx)]
        _, s = w.replay_units({"state": x[pick], "eta": eta[pick]}, plans)
        viol[regime] = sum(
            not check_feasibility(cons, x[i], to_dense(p, w.T, w.k, space), s[j]).passed
            for j, (i, p) in enumerate(zip(pick, plans, strict=True))
        )
    assert viol["R6"] <= viol["R5"]


def test_domain_action_card(energy):
    from twist.reporting.benchmark_cards import domain_action_card

    w, _, _ = energy
    card = domain_action_card(w)
    for s in ("`u_dispatch`", "`u_dr`", "rebound", "'hot_spot'", "exogenous weather", "1 h"):
        assert s in card, s


@pytest.mark.smoke
def test_mini_bridge_report(tmp_path, monkeypatch):
    import json

    from twist.evaluation.bridge_report import BridgeConfig, run_bridge
    from twist.worlds.suite import SuiteConfig, build_suite

    monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    suite = build_suite(
        SuiteConfig(
            name="e9",
            output_dir=str(tmp_path / "suites"),
            families=["energy"],
            worlds_per_split={"dev": 1},
            T=50,
            n_samples=300,
            pools={"standard": {"season": ["winter"], "energy_label": ["overload_event"]}},
        )
    )
    assert "CC-BY 4.0" in (suite / "DATASET_CARD.md").read_text()
    rec = {"steps": 15, "t0_fracs": [0.7, 0.76], "duration": 4}

    def row(name, regime, label, **p):
        return {"name": name, "regime": regime, "params": {**rec, **p, "label": label}}

    cfg = BridgeConfig.model_validate(
        {
            "name": "bridge9",
            "output_dir": str(tmp_path / "bridge"),
            "suite": str(suite),
            "worlds": ["energy_dev_000"],
            "classifiers": [{"architecture": "logreg"}],
            "explainers": [
                {"name": "NUN", "regime": "R1"},
                row("PearlSCMRecourse", "R5", "SetPoint[R5 grid]", action_form="interval"),
                row(
                    "PearlSCMRecourse",
                    "R5",
                    "Dispatch[R5 constraint-blind]",
                    action_form="control",
                    control="u_dispatch",
                    lambda_prox=0.0,
                ),
                row(
                    "OracleActionRecourse",
                    "R6",
                    "Dispatch[R6 constraint-aware]",
                    action_form="control",
                    control="u_dispatch",
                    lambda_prox=0.0,
                ),
            ],
            "explainer_seeds": [501],
            "n_factuals": 6,
            "target": 0,
            "n_boot": 50,
            "evaluation": {
                "readings": ["first_change_atomic"],
                "compute_dtw": False,
                "constraints": [
                    {"name": "amplitude_bounds", "max_abs": 3.0},
                    {
                        "name": "state_bounds",
                        "id": "soc_capacity",
                        "channels": [SOC],
                        "lower": 0.0,
                        "upper": 1.0,
                    },
                ],
            },
            "domain_constraints": ["soc_capacity"],
        }
    )
    path = run_bridge(cfg)
    text = path.read_text()
    for s in (
        "native_control_actions_scored | **PASS**",
        "control_actions_replayable | **PASS**",
        "domain_constraint_changes_acceptability",
        "HE1:",
        "HE2:",
        "Claim boundary",
        "### Action card — `energy_dev_000`",
    ):
        assert s in text, s
    res = json.loads(path.with_suffix(".json").read_text())
    rows = {r["method"]: r for r in res["rows"]}
    assert rows["NUN"]["action"] == "reading" and rows["NUN"]["invalid_itt"][0] == 1.0
    assert rows["Dispatch[R5 constraint-blind]"]["action"] == "control"
