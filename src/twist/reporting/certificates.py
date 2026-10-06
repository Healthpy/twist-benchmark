"""Explanation certificates: a self-contained, auditable record answering

what action is recommended; under which mechanism and assumptions; what trajectory it generates
for this unit; whether it achieves the target; which constraints it satisfies; how uncertain
the conclusion is; and why this explanation was selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from twist.utils.serialization import write_json
from twist.worlds.base_scm import TemporalSCM

CERTIFICATE_VERSION = "0.1"


def make_certificate(
    record: dict[str, Any], world: TemporalSCM, manifest: dict[str, Any]
) -> dict[str, Any]:
    prov = record["provenance"]
    return {
        "certificate_version": CERTIFICATE_VERSION,
        "explanation_id": record["explanation_id"],
        "explainer": record["explainer"],
        "action_reading": record["reading"],
        "recommended_action": record["action"],
        "native_action": record["native_action"],
        "mechanism": {
            "world_id": world.world_id,
            "world_family": world.family,
            "mechanism_hash": world.card()["mechanism_hash"],
            "assumptions": world.assumptions,
            "explainer_access_regime": record["mechanism_access_regime"],
        },
        "unit": {
            "factual_id": record["factual_id"],
            "target_outcome": record["target"],
            "h_factual": record["h_factual"],
        },
        "replay": {
            "semantics": "pearl (factual innovations retained)"
            if world.assumptions.get("exact_abduction", True)
            else "oracle: Pearl replay of the stored state (factual innovations), re-observed with "
            "the factual sensor noise and missingness",
            "replay_error_pearl": record["replay_error_pearl"],
            "replay_error_noiseless": record["replay_error_noiseless"],
            "classifier_on_replay": record["h_replay"],
            "world_label_on_replay": record["world_replay"],
            "achieves_world_target": record["scorable"]
            and record["world_replay"] == record["target"],
            "t_label": record["t_label"],
            "action_t_first": record["action_t_first"],
            "label_distance": record["label_distance"],
            "replay_hash": record.get("replay_hash"),
        },
        "proposal": {
            "classifier_on_proposal": record["h_proposal"],
            "cf_faith_pearl_hard": record["cf_faith_pearl_hard"],
            "cf_faith_noiseless_hard": record["cf_faith_noiseless_hard"],
            "D_pearl": record["D_pearl"],
            "proposal_hash": record.get("proposal_hash"),
            "ood_proposal": record.get("ood_proposal"),
        },
        "mechanism_consistency": {
            k: record.get(k)
            for k in (
                "replay_error_pearl_norm",
                "soft_faith_replay",
                "effect_corr",
                "support_overlap",
                "retroactive_fraction",
                "prefix_preservation",
                "causal_path_consistency",
            )
        },
        "arrays_file": record.get("arrays_file"),
        "action_validation": {
            "valid": not record["invalid_action"],
            "violations": record["action_violations"],
        },
        "constraints": record["feasibility"],
        "costs": record["costs"],
        "selection": {
            "objective": prov["selection_objective"],
            "candidate_count": prov["candidate_count"],
            "selection_rank": prov["selection_rank"],
        },
        "uncertainty": prov["uncertainty"],
        "known_failure_modes": prov["known_failure_modes"],
        "provenance_completeness": record["provenance_completeness"],
        "run_manifest": manifest,
    }


def write_certificate(path: str | Path, certificate: dict[str, Any]) -> Path:
    """Write the JSON certificate and a human-readable Markdown rendering next to it."""
    path = write_json(path, certificate)
    path.with_suffix(".md").write_text(certificate_markdown(certificate), encoding="utf-8")
    return path


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "–" if v is None else str(v)


def certificate_markdown(c: dict[str, Any]) -> str:
    """Render a certificate as the seven audit questions of the plan's definition of done."""
    rep, prop, mc = c["replay"], c["proposal"], c["mechanism_consistency"]
    act = c["recommended_action"] or {}
    validation = c["action_validation"]
    constraints = c["constraints"] or {"passed": None, "results": []}
    lines = [
        f"# Explanation certificate `{c['explanation_id']}`",
        "",
        f"Explainer **{c['explainer']}**, reading **{c['action_reading']}**, unit "
        f"`{c['unit']['factual_id']}`, target outcome **{c['unit']['target_outcome']}**.",
        "",
        "## 1. What action is recommended?",
        f"- {'Native (declared by the explainer)' if c['native_action'] else 'Reconstructed by benchmark adapter `' + str(act.get('adapter_name')) + '` — not a claim of the method'}",
        f"- semantics `{act.get('semantics')}`; action `{act.get('action')}`",
        "",
        "## 2. Under which mechanism and assumptions?",
        f"- world `{c['mechanism']['world_id']}` ({c['mechanism']['world_family']}), mechanism hash "
        f"`{c['mechanism']['mechanism_hash']}`",
        f"- assumptions: {c['mechanism']['assumptions']}",
        f"- explainer access regime: {c['mechanism']['explainer_access_regime']}",
        "",
        "## 3. What trajectory does the action generate for this unit?",
        f"- {rep['semantics']}; replay hash `{_fmt(rep['replay_hash'])}`",
        f"- replay error vs proposal: Pearl {_fmt(rep['replay_error_pearl'])}, "
        f"normalized {_fmt(mc['replay_error_pearl_norm'])}, soft faith {_fmt(mc['soft_faith_replay'])}",
        f"- effect correlation {_fmt(mc['effect_corr'])}, support overlap {_fmt(mc['support_overlap'])}, "
        f"causal-path consistency {_fmt(mc['causal_path_consistency'])}, retroactive fraction "
        f"{_fmt(mc['retroactive_fraction'])}",
        f"- proposal: classifier {prop['classifier_on_proposal']}, first-slice CF-faith "
        f"Pearl/noiseless {_fmt(prop['cf_faith_pearl_hard'])}/{_fmt(prop['cf_faith_noiseless_hard'])}, "
        f"D = {prop['D_pearl']}",
        "",
        "## 4. Does it achieve the target outcome?",
        f"- world label on replay: {rep['world_label_on_replay']}; classifier on replay: "
        f"{rep['classifier_on_replay']}; **achieves world target: {rep['achieves_world_target']}**",
        f"- label site t={rep['t_label']}, action starts t={_fmt(rep['action_t_first'])}, "
        f"distance {_fmt(rep['label_distance'])}",
        "",
        "## 5. Which constraints does it satisfy?",
        f"- action valid: {validation['valid']} {validation['violations'] or ''}",
        f"- feasibility passed: {constraints['passed']}",
        *[
            f"  - {r['name']}: {'pass' if r['passed'] else 'FAIL — ' + r['message']}"
            for r in constraints["results"]
        ],
        f"- costs: {c['costs']}",
        "",
        "## 6. How uncertain is this conclusion?",
        f"- reported uncertainty: {c['uncertainty'] or 'none reported by the explainer'}",
        f"- known failure modes: {', '.join(c['known_failure_modes']) or 'none detected'}",
        "",
        "## 7. Why was this explanation selected?",
        f"- objective: {c['selection']['objective'] or 'not reported'}",
        f"- candidates {c['selection']['candidate_count']}, rank {_fmt(c['selection']['selection_rank'])}",
        "",
        f"Provenance completeness {_fmt(c['provenance_completeness'])}; run "
        f"`{c['run_manifest']['run_id']}`, config `{c['run_manifest']['config_hash']}`, commit "
        f"`{c['run_manifest']['git_commit']}`.",
    ]
    return "\n".join(lines) + "\n"
