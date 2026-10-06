"""Traceability audit: can an auditor reproduce each certified explanation from the artifacts?

For a finished run directory, the audit reloads the world, data and classifier from the paths in
``run_info.json`` and, for every certificate, deserializes the recommended action, replays it on
the factual unit and checks it against what was recorded:

* config hash — the resolved config re-hashes to the recorded ``config_hash``;
* dataset and mechanism hashes — the stored world and data match the run manifest;
* action round-trip — the serialized action deserializes to an identical action;
* array integrity — the saved proposal/replay arrays hash to the recorded hashes;
* replay reproducibility — a fresh replay matches the saved replay within ``tol``;
* outcome reproducibility — world label, classifier decision and replay error match;
* provenance completeness, logged constraint status and a recorded selection objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from twist.actions.schemas import ActionPlan
from twist.classifiers.models import TorchClassifier
from twist.config import ExperimentConfig
from twist.metrics.mechanism import replay_error
from twist.utils.serialization import hash_arrays, hash_config, read_json, write_json
from twist.worlds.base_scm import TemporalSCM
from twist.worlds.io import load_world_data, subset_exo


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def audit_run(run_dir: str | Path, tol: float = 1e-9) -> dict[str, Any]:
    run_dir = Path(run_dir)
    info = read_json(run_dir / "run_info.json")
    cfg = ExperimentConfig.model_validate(read_json(run_dir / "config.resolved.json"))
    run_checks = [
        _check("config_hash", hash_config(cfg) == info["config_hash"], info["config_hash"]),
    ]
    world = TemporalSCM.load(info["world_dir"])
    data = load_world_data(info["world_dir"])
    n_train = data["n_train"]
    clf = TorchClassifier.load(info["model_dir"])
    certs = sorted((run_dir / "certificates").glob("*.json"))
    arrays_cache: dict[str, Any] = {}
    results = []
    for path in certs:
        c = read_json(path)
        manifest = c["run_manifest"]
        checks = []
        checks.append(
            _check("dataset_hash", hash_arrays(data["x"], data["y"]) == manifest["dataset_hash"])
        )
        checks.append(
            _check(
                "mechanism_hash", world.card()["mechanism_hash"] == c["mechanism"]["mechanism_hash"]
            )
        )
        plan = ActionPlan.from_dict(c["recommended_action"])
        checks.append(_check("action_roundtrip", plan.to_dict() == c["recommended_action"]))
        idx = int(c["unit"]["factual_id"].rsplit("/", 1)[1])
        fresh, fresh_state = world.replay_units(subset_exo(data["exo"], n_train + idx), plan)

        arrays = arrays_cache.setdefault(c["arrays_file"], dict(np.load(c["arrays_file"])))
        row = int(np.flatnonzero(arrays["explanation_ids"] == c["explanation_id"])[0])
        saved_rep, proposal = arrays["replays"][row], arrays["proposals"][row]
        checks.append(
            _check("replay_integrity", hash_arrays(saved_rep) == c["replay"]["replay_hash"])
        )
        checks.append(
            _check("proposal_integrity", hash_arrays(proposal) == c["proposal"]["proposal_hash"])
        )
        dev = float(np.max(np.abs(fresh - saved_rep)))
        checks.append(_check("replay_reproducible", dev <= tol, f"max |Δ| = {dev:.3g}"))
        checks.append(
            _check(
                "world_label_reproducible",
                int(world.outcome(fresh_state)) == c["replay"]["world_label_on_replay"],
            )
        )
        checks.append(
            _check(
                "classifier_reproducible",
                int(clf.predict(fresh)) == c["replay"]["classifier_on_replay"],
            )
        )
        err = float(replay_error(proposal[None], fresh[None])[0])
        rec_err = c["replay"]["replay_error_pearl"]
        checks.append(
            _check(
                "replay_error_reproducible",
                abs(err - rec_err) <= tol + 1e-9 * abs(rec_err),
                f"{err:.6g} vs {rec_err:.6g}",
            )
        )
        checks.append(_check("provenance_complete", c["provenance_completeness"] == 1.0))
        checks.append(_check("constraints_logged", c["constraints"] is not None))
        checks.append(_check("objective_recorded", bool(c["selection"]["objective"])))
        results.append(
            {
                "certificate": path.name,
                "explainer": c["explainer"],
                "reading": c["action_reading"],
                "passed": all(k["passed"] for k in checks),
                "checks": checks,
            }
        )

    report = {
        "run_id": info["run_id"],
        "n_certificates": len(results),
        "passed": all(k["passed"] for k in run_checks) and all(r["passed"] for r in results),
        "run_checks": run_checks,
        "certificates": results,
    }
    write_json(run_dir / "audit.json", report)
    (run_dir / "audit.md").write_text(audit_markdown(report), encoding="utf-8")
    return report


def audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Traceability audit — `{report['run_id']}`",
        "",
        f"**{'PASS' if report['passed'] else 'FAIL'}** — {report['n_certificates']} certificates.",
        "",
        "| scope | check | result | detail |",
        "|---|---|---|---|",
    ]
    for k in report["run_checks"]:
        lines.append(
            f"| run | {k['check']} | {'pass' if k['passed'] else '**FAIL**'} | {k['detail']} |"
        )
    for r in report["certificates"]:
        for k in r["checks"]:
            lines.append(
                f"| {r['explainer']} / {r['reading']} | {k['check']} | "
                f"{'pass' if k['passed'] else '**FAIL**'} | {k['detail']} |"
            )
    return "\n".join(lines) + "\n"
