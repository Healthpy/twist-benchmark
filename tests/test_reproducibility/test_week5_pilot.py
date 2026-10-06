"""Week 5: multi-world runner and pilot dashboard (miniature end-to-end)."""

import pytest

from twist.config import ExplainerConfig
from twist.evaluation.multiworld_runner import (
    MultiWorldConfig,
    expand_explainers,
    run_multiworld,
    suite_world_config,
)
from twist.evaluation.pilot_report import build_pilot_report
from twist.worlds.suite import SuiteConfig, build_suite


def test_seed_expansion_only_for_stochastic_methods():
    ex = expand_explainers(
        [
            ExplainerConfig(name="Wachter"),
            ExplainerConfig(name="COMTE"),
            ExplainerConfig(name="PearlSCMRecourse"),
        ],
        [1, 2, 3],
    )
    assert [(e.name, e.seed) for e in ex] == [
        ("Wachter", 1),
        ("Wachter", 2),
        ("Wachter", 3),
        ("COMTE", 1),
        ("PearlSCMRecourse", 1),
    ]


@pytest.mark.smoke
def test_mini_pilot_end_to_end(tmp_path, monkeypatch):
    from twist.explainers.adapters.cfts_baselines import cfts_available

    if not cfts_available():
        pytest.skip("cfts baselines extra not installed")
    monkeypatch.delenv("TWIST_SUITE_SECRET", raising=False)
    suite = build_suite(
        SuiteConfig(
            name="mini",
            output_dir=str(tmp_path / "suites"),
            families=["linear"],
            worlds_per_split={"dev": 1, "test": 1},
            k=4,
            T=24,
            n_samples=300,
        )
    )
    with pytest.raises(PermissionError):
        suite_world_config(suite, "linear_test_000")
    cfg = MultiWorldConfig.model_validate(
        {
            "name": "mini_pilot",
            "output_dir": str(tmp_path / "pilots"),
            "suite": str(suite),
            "worlds": ["linear_dev_000"],
            "classifiers": [{"architecture": "logreg"}],
            "explainers": [
                {"name": "NUN", "regime": "R1"},
                {"name": "COMTE", "regime": "R1"},
                {"name": "Wachter", "regime": "R1", "params": {"max_cfs": 30}},
                {
                    "name": "PearlSCMRecourse",
                    "regime": "R5",
                    "params": {"steps": 10, "t0_fracs": [0.5]},
                },
            ],
            "explainer_seeds": [1, 2],
            "n_factuals": 4,
            "jobs": 1,
            "evaluation": {"readings": ["first_change_atomic"], "compute_dtw": False},
        }
    )
    runs = run_multiworld(cfg)
    assert len(runs) == 1
    assert run_multiworld(cfg) == runs  # completed tasks are reused (resumable runs)
    import twist.utils.serialization as ser

    monkeypatch.setattr(ser, "code_fingerprint", lambda: "different-code")
    from twist.evaluation.multiworld_runner import completed_run, make_tasks

    assert completed_run(make_tasks(cfg)[0][0]) is None  # changed code invalidates reuse
    # every seed keeps its own arrays (regression: seeds used to overwrite each other)
    import json

    import numpy as np

    from twist.evaluation.audit import audit_run

    recs = json.loads((runs[0] / "records.json").read_text())
    wachter = [r for r in recs if r["explainer"] == "Wachter"]
    assert {r["explainer_seed"] for r in wachter} == {1, 2}
    for r in wachter:
        assert r["explanation_id"] in set(np.load(r["arrays_file"])["explanation_ids"])
    assert audit_run(runs[0])["passed"]
    report = build_pilot_report("mini_pilot", runs, tmp_path / "report")
    text = report.read_text()
    for section in (
        "Week-5 acceptance checks",
        "Failure taxonomy",
        "Seed stability",
        "Classifiers",
    ):
        assert section in text
    assert "standardized_output_or_failure_record | **PASS**" in text
