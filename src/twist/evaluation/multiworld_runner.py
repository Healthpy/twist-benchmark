"""Multi-world runner: worlds × classifiers × explainers × explainer seeds (Week 5 pilot; reused by
the Week 10 campaign).

Each (world, classifier) pair is one paired experiment (``run_paired_experiment``): every explainer
and seed is evaluated on the same factual units. Stochastic explainers run once per seed in
``explainer_seeds``; deterministic ones run once. Worlds come from a built suite (visible splits
only): they are regenerated from their stored ``world_config.json``, and the dataset hash is
checked against the suite manifest so the pilot provably uses the suite's data. Tasks run in
parallel processes (``jobs``), one thread each.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from twist.config import (
    BudgetConfig,
    ClassifierConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExplainerConfig,
    WorldConfig,
    load_yaml,
)
from twist.utils.logging import get_logger
from twist.utils.serialization import read_json

log = get_logger(__name__)


class MultiWorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: str = "artifacts/pilots"
    suite: str
    worlds: list[str]
    classifiers: list[ClassifierConfig]
    explainers: list[ExplainerConfig]
    explainer_seeds: list[int] = Field(default_factory=lambda: [501, 502, 503])
    n_factuals: int = 50
    target: int = 1
    evaluation_seed: int = 9001
    certificates: int = 1
    jobs: int = 1
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @model_validator(mode="before")
    @classmethod
    def _resolve_refs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            for key in ("classifiers", "explainers"):
                if key in data:
                    data[key] = [load_yaml(e) if isinstance(e, str) else e for e in data[key]]
        return data


def suite_world_config(suite: str | Path, world_id: str) -> tuple[WorldConfig, str]:
    """World config and expected dataset hash of a visible suite world."""
    root = Path(suite)
    manifest = read_json(root / "manifest.json")
    entry = next((e for e in manifest["worlds"] if e["world_id"] == world_id), None)
    if entry is None:
        raise KeyError(f"{world_id} not in suite {root}")
    if entry["visibility"] != "visible":
        raise PermissionError(
            f"{world_id} is a hidden ({entry['split']}) world; pilots use dev/val"
        )
    cfg = WorldConfig.model_validate(read_json(root / entry["path"] / "world_config.json"))
    return cfg, entry["dataset_hash"]


def expand_explainers(explainers: list[ExplainerConfig], seeds: list[int]) -> list[ExplainerConfig]:
    from twist.explainers.registry import EXPLAINERS

    out = []
    for e in explainers:
        if EXPLAINERS[e.name].stochastic:
            out += [e.model_copy(update={"seed": s}) for s in seeds]
        else:
            out.append(e.model_copy(update={"seed": seeds[0]}))
    return out


def make_tasks(cfg: MultiWorldConfig) -> list[tuple[ExperimentConfig, str]]:
    tasks = []
    explainers = expand_explainers(cfg.explainers, cfg.explainer_seeds)
    for world_id in cfg.worlds:
        wcfg, dataset_hash = suite_world_config(cfg.suite, world_id)
        for ccfg in cfg.classifiers:
            exp = ExperimentConfig(
                name=f"{cfg.name}__{world_id}__{ccfg.architecture}",
                output_dir=str(Path(cfg.output_dir) / cfg.name),
                evaluation_seed=cfg.evaluation_seed,
                world=wcfg,
                classifier=ccfg,
                explainers=explainers,
                n_factuals=cfg.n_factuals,
                target=cfg.target,
                evaluation=cfg.evaluation,
                budget=cfg.budget,
                certificates=cfg.certificates,
            )
            tasks.append((exp, dataset_hash))
    return tasks


def completed_run(exp: ExperimentConfig, same_code: bool = True) -> Path | None:
    """The newest finished run of exactly this task (same config hash and code fingerprint).

    ``run_info.json`` is written last, so its presence marks a complete run. Interrupted runs
    are ignored and recomputed, which makes multi-world runs resumable at task granularity.
    ``same_code=False`` (report rebuilds only) accepts runs made by other code, with a warning.
    """
    from twist.utils.serialization import code_fingerprint, hash_config

    target, code = hash_config(exp), code_fingerprint()
    for info_path in sorted(
        Path(exp.output_dir, "evaluations").glob(f"{exp.name}__*/run_info.json"), reverse=True
    ):
        info = read_json(info_path)
        if info.get("config_hash") == target and (
            not same_code or info.get("code_fingerprint") == code
        ):
            if info.get("code_fingerprint") != code:
                log.warning(
                    "%s was produced by different code (report rebuild only)", info_path.parent
                )
            return info_path.parent
    return None


def _run_task(payload: tuple[dict[str, Any], str]) -> str:
    from twist.evaluation.paired_runner import run_paired_experiment

    exp_dict, dataset_hash = payload
    exp = ExperimentConfig.model_validate(exp_dict)
    done = completed_run(exp)
    if done is not None:
        log.info("reusing completed run %s", done)
        return str(done)
    try:
        run_dir = run_paired_experiment(exp)
    except NotImplementedError as exc:  # e.g. a shift that does not apply to this world family
        log.warning("skipping task %s: %s", exp.name, exc)
        return ""
    split = read_json(Path(exp.output_dir) / "worlds" / exp.world.world_id / "split.json")
    if split["dataset_hash"] != dataset_hash:
        raise RuntimeError(
            f"{exp.world.world_id}: regenerated data hash {split['dataset_hash']} "
            f"differs from the suite manifest {dataset_hash}"
        )
    return str(run_dir)


def run_multiworld(cfg: MultiWorldConfig) -> list[Path]:
    tasks = make_tasks(cfg)
    payloads = [(t.model_dump(), h) for t, h in tasks]
    log.info("%d tasks, %d jobs", len(tasks), cfg.jobs)
    if cfg.jobs <= 1:
        return [Path(r) for r in map(_run_task, payloads) if r]
    with ProcessPoolExecutor(max_workers=cfg.jobs, mp_context=mp.get_context("spawn")) as pool:
        return [Path(r) for r in pool.map(_run_task, payloads) if r]
