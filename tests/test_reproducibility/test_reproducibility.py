import json

import numpy as np
import pytest

from twist.config import ExperimentConfig, load_experiment
from twist.evaluation.result_schema import ProvenanceRecord, RunManifest
from twist.utils.logging import read_manifest
from twist.utils.seeds import derive_seed
from twist.utils.serialization import hash_arrays, hash_config


def test_derive_seed_is_stable_and_key_sensitive():
    assert derive_seed(42, "data") == derive_seed(42, "data")
    assert derive_seed(42, "data") != derive_seed(42, "model")
    assert derive_seed(42, "data") != derive_seed(43, "data")


def test_classifier_training_is_deterministic(linear_world):
    from twist.classifiers.train import train_classifier

    world, x, _ = linear_world
    y = world.label(x)
    a = train_classifier(x[:100], y[:100], x[100:150], y[100:150], hidden=4, epochs=1, seed=3)
    b = train_classifier(x[:100], y[:100], x[100:150], y[100:150], hidden=4, epochs=1, seed=3)
    np.testing.assert_array_equal(a.logits(x[:20]), b.logits(x[:20]))


def test_toy_config_validates():
    cfg = load_experiment("experiments/toy.yaml")
    assert cfg.world.family == "linear" and len(cfg.explainers) == 6
    assert hash_config(cfg) == hash_config(load_experiment("experiments/toy.yaml"))


@pytest.mark.smoke
def test_end_to_end_mini_run_artifact_schema(tmp_path):
    from twist.evaluation.paired_runner import run_paired_experiment

    cfg = ExperimentConfig.model_validate(
        {
            "name": "smoke",
            "output_dir": str(tmp_path),
            "world": {
                "family": "linear",
                "world_id": "smoke_w",
                "seed": 0,
                "n_samples": 300,
                "params": {
                    "k": 3,
                    "T": 16,
                    "density": 0.5,
                    "spectral_radius": 0.95,
                    "rescale": "exact",
                },
            },
            "classifier": {"hidden": 8, "epochs": 2},
            "explainers": [
                {
                    "name": "PearlSCMRecourse",
                    "regime": "R5",
                    "params": {"steps": 10, "t0_fracs": [0.5]},
                },
                {"name": "WachterReference", "params": {"steps": 10}},
                {"name": "NoActionAbstain"},
            ],
            "n_factuals": 4,
            "evaluation": {"constraints": [{"name": "amplitude_bounds", "max_abs": 5.0}]},
        }
    )
    run_dir = run_paired_experiment(cfg)
    manifests = read_manifest(tmp_path / "manifest.jsonl")
    assert len(manifests) == 4  # Pearl/native, Wachter×2 readings, Abstain/native
    for m in manifests:
        RunManifest.model_validate(m)
    records = json.loads((run_dir / "records.json").read_text())
    for r in records:
        ProvenanceRecord.model_validate(r["provenance"])
        assert r["provenance_completeness"] == 1.0 and r["action_roundtrip_ok"]
    summary = json.loads((run_dir / "summary.json").read_text())
    for row in summary:
        for key in (
            "proposal_model_success",
            "replay_model_success",
            "replay_world_success",
            "D_pearl",
            "reading",
            "n_scorable",
            "n_abstain",
        ):
            assert key in row
    certs = list((run_dir / "certificates").glob("*.json"))
    assert certs and all("recommended_action" in json.loads(c.read_text()) for c in certs)
    for r in records:
        assert not any(k.startswith("_") for k in r)
    from twist.evaluation.audit import audit_run

    audit = audit_run(run_dir)
    assert audit["passed"] and audit["n_certificates"] == len(certs)
    assert all(c.with_suffix(".md").exists() for c in certs)
    data = np.load(tmp_path / "worlds" / "smoke_w" / "data.npz")
    assert hash_arrays(data["x"], data["y"]) == manifests[0]["dataset_hash"]
