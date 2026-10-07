"""Paired experiment runner: every explainer is evaluated on the same factual units.

Pipeline: generate world + data → train classifier → select flip candidates → run explainers →
evaluate under registered readings → write records, summary, certificates and run manifests.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from twist import __version__
from twist.actions.feasibility import build_constraints
from twist.classifiers.models import TorchClassifier
from twist.classifiers.train import train_classifier
from twist.config import ExperimentConfig, WorldConfig
from twist.evaluation.protocol import evaluate_explanations, summarize
from twist.explainers.mechanism_access.regimes import build_context
from twist.explainers.registry import build_explainer
from twist.metrics.conventional import OODScorer, channel_iqr
from twist.reporting.benchmark_cards import world_card_markdown
from twist.reporting.certificates import make_certificate, write_certificate
from twist.reporting.figures import plot_case_study
from twist.reporting.tables import markdown_table
from twist.utils.logging import append_manifest, get_logger
from twist.utils.seeds import configure_threads, derive_seed, seed_everything
from twist.utils.serialization import (
    code_fingerprint,
    git_commit,
    hardware_summary,
    hash_arrays,
    hash_config,
    utc_timestamp,
    write_json,
)
from twist.worlds.base_scm import TemporalSCM
from twist.worlds.io import save_world_data, subset_exo
from twist.worlds.registry import make_world
from twist.worlds.shifts import apply_shift, perturb_units, shift_id

log = get_logger(__name__)


def generate_world_data(cfg: WorldConfig, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Build a world, simulate its population, fit the label threshold on the training split."""
    world = make_world(
        cfg.family,
        cfg.world_id,
        cfg.seed,
        noise=cfg.noise,
        label=cfg.label,
        action_space=cfg.action_space,
        **cfg.params,
    )
    x, exo = world.simulate_units(
        cfg.n_samples, np.random.default_rng(derive_seed(cfg.seed, "data"))
    )
    n_train = int(cfg.train_frac * cfg.n_samples)
    world.label.fit(exo["state"][:n_train])
    y = world.outcome(exo["state"])
    data = {
        "world": world,
        "x_train": x[:n_train],
        "y_train": y[:n_train],
        "exo_train": subset_exo(exo, slice(0, n_train)),
        "x_test": x[n_train:],
        "y_test": y[n_train:],
        "exo_test": subset_exo(exo, slice(n_train, None)),
        "dataset_hash": hash_arrays(x, y),
    }
    if out_dir is not None:
        d = Path(out_dir)
        world.save(d)
        save_world_data(d, x, y, exo, n_train)
        (d / "WORLD_CARD.md").write_text(world_card_markdown(world), encoding="utf-8")
        write_json(
            d / "split.json",
            {
                "n_train": n_train,
                "n_test": cfg.n_samples - n_train,
                "dataset_hash": data["dataset_hash"],
            },
        )
    return data


def _prepare_mechanism(
    ecfg: Any,
    world: Any,
    data: dict[str, Any],
    cache: dict[str, tuple[Any, str]],
    quality: dict[str, dict[str, Any]],
) -> tuple[Any, str]:
    """Build (or reuse) the R3/R4 mechanism an explainer config asks for, and score it against the
    true world on held-out units (the explainer never sees the score)."""
    import json

    from twist.mechanisms.estimators import estimate_mechanism, mechanism_label
    from twist.mechanisms.misspecification import misspecify
    from twist.mechanisms.quality import mechanism_quality

    spec = dict(ecfg.mechanism or {"estimator": "masked" if ecfg.regime == "R3" else "lasso"})
    if ecfg.regime == "R3" and "misspecify" not in spec and spec.get("estimator") != "masked":
        raise ValueError("R3 gives the true graph and family: use estimator 'masked'")
    key = json.dumps(spec, sort_keys=True)
    if key not in cache:
        if "misspecify" in spec:
            m = spec["misspecify"]
            mech = misspecify(world, m["kind"], float(m["level"]), seed=int(m.get("seed", 0)))
            label = f"misspecified:{m['kind']}-{float(m['level']):g}"
        else:
            obs = list(getattr(world, "obs_index", range(world.k)))
            G = world.dependency_graph()[np.ix_(obs, obs)]
            mech = estimate_mechanism(
                spec, data["x_train"], world, G, seed=derive_seed(world.seed, "mechanism", key)
            )
            label = f"estimated:{mechanism_label(spec)}"
        cache[key] = (mech, label)
        quality[label] = mechanism_quality(mech, world, data["x_test"], data["exo_test"])
        quality[label]["spec"] = spec
        if (info := getattr(mech, "discovery_info", None)) is not None:
            quality[label]["discovery"] = info
        log.info("mechanism %s: replay error %.4f", label, quality[label]["replay_error"])
    return cache[key]


def classifier_reports(
    clf: TorchClassifier,
    world: Any,
    data: dict[str, Any],
    cfg: ExperimentConfig,
    n_shift: int = 500,
) -> dict[str, Any]:
    """Test-split accuracy/calibration and robustness under an innovation shift (uniform
    innovations of matched variance, same mechanism and label; plan §Week 5, task 1)."""
    from twist.classifiers.metrics import classifier_report
    from twist.worlds.shifts import innovation_shift

    shifted = innovation_shift(world, "uniform")
    xs, exo = shifted.simulate_units(
        n_shift, np.random.default_rng(derive_seed(cfg.world.seed, "shift-uniform"))
    )
    return {
        "architecture": cfg.classifier.architecture,
        "test": classifier_report(clf, data["x_test"], data["y_test"]),
        "shift_uniform_innovations": classifier_report(clf, xs, shifted.outcome(exo["state"])),
        "spec": clf.spec,
    }


def select_factuals(
    clf: TorchClassifier, x: np.ndarray, target: int | str, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """First ``n`` held-out units predicted outside the target class (draft §4.3)."""
    pred = clf.predict(x)
    targets = 1 - pred if target == "flip" else np.full(len(x), int(target))
    idx = np.flatnonzero(pred != targets)[:n]
    return idx, targets[idx]


def run_paired_experiment(cfg: ExperimentConfig, repo_root: str | Path = ".") -> Path:
    t_start = time.perf_counter()
    code_fp = code_fingerprint()  # the code this process loaded (cached per process)
    configure_threads()  # library callers get the same thread policy as the CLI
    seed_everything(cfg.evaluation_seed)
    config_hash = hash_config(cfg)
    commit = git_commit(repo_root)
    run_id = f"{cfg.name}__{utc_timestamp().replace(':', '').replace('-', '')}__{config_hash[:8]}"
    out = Path(cfg.output_dir)
    run_dir = out / "evaluations" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.resolved.json", cfg)

    log.info("generating world %s", cfg.world.world_id)
    data = generate_world_data(cfg.world, out / "worlds" / cfg.world.world_id)
    world: TemporalSCM = data["world"]

    log.info("training %s classifier", cfg.classifier.architecture)
    clf = train_classifier(
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
        **cfg.classifier.model_dump(),
    )
    clf_id = f"{cfg.world.world_id}__{cfg.classifier.architecture}_seed_{cfg.classifier.seed}"
    clf.save(out / "models" / clf_id)
    clf_reports = classifier_reports(clf, world, data, cfg)

    # evaluation population: the base test split, a shifted deployment world, and/or perturbed units
    eval_world, eval_clf = world, clf
    pop_x, pop_exo = data["x_test"], data["exo_test"]
    pop_dir, model_dir = out / "worlds" / cfg.world.world_id, out / "models" / clf_id
    if cfg.shift:
        eval_world = apply_shift(world, cfg.shift, data["exo_train"]["state"])
        if cfg.shift["kind"] == "retrain":
            eval_clf = train_classifier(
                data["x_train"],
                data["y_train"],
                data["x_test"],
                data["y_test"],
                **{**cfg.classifier.model_dump(), "seed": cfg.classifier.seed + 1000},
            )
            model_dir = out / "models" / f"{clf_id}__retrained"
            eval_clf.save(model_dir)
        else:
            pop_x, pop_exo = eval_world.simulate_units(
                len(pop_x),
                np.random.default_rng(derive_seed(world.seed, "shift", shift_id(cfg.shift))),
            )
    idx, targets = select_factuals(clf, pop_x, cfg.target, cfg.n_factuals)
    if cfg.perturb:
        px, pexo = perturb_units(
            eval_world,
            subset_exo(pop_exo, idx),
            float(cfg.perturb.get("scale", 0.05)),
            np.random.default_rng(derive_seed(world.seed, "perturb", cfg.evaluation_seed)),
        )
        pop_x, pop_exo = pop_x.copy(), {k: v.copy() for k, v in pop_exo.items()}
        pop_x[idx] = px
        for k in pop_exo:
            pop_exo[k][idx] = pexo[k]
    if cfg.shift or cfg.perturb:  # persist the evaluated population so audits can replay it
        tag = eval_world.world_id + (
            f"__perturb-{cfg.perturb.get('scale', 0.05):g}" if cfg.perturb else ""
        )
        pop_dir = out / "worlds" / tag
        eval_world.save(pop_dir)
        pop_y = eval_world.outcome(pop_exo["state"])
        save_world_data(pop_dir, pop_x, pop_y, pop_exo, 0)
        unit_prefix = tag
        eval_dataset_hash = hash_arrays(pop_x, pop_y)
    else:
        unit_prefix = cfg.world.world_id
        eval_dataset_hash = data["dataset_hash"]
    factuals = pop_x[idx]
    exo_f = subset_exo(pop_exo, idx)
    factual_ids = [f"{unit_prefix}/test/{i}" for i in idx]
    log.info(
        "selected %d flip candidates%s", len(idx), f" (shift {cfg.shift})" if cfg.shift else ""
    )

    constraints = build_constraints(cfg.evaluation.constraints)
    iqr = channel_iqr(data["x_train"])
    ood = OODScorer(data["x_train"], seed=derive_seed(cfg.evaluation_seed, "ood"))
    expl_dir = out / "explanations" / run_id
    all_records: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    mechanisms: dict[str, tuple[Any, str]] = {}
    mech_quality: dict[str, dict[str, Any]] = {}
    for ecfg in cfg.explainers:
        explainer = build_explainer(ecfg.name, seed=ecfg.seed, **ecfg.params)
        mech, mech_label = None, ""
        if ecfg.regime in ("R3", "R4"):
            try:
                mech, mech_label = _prepare_mechanism(ecfg, world, data, mechanisms, mech_quality)
            except (NotImplementedError, ValueError) as exc:
                log.warning("skipping %s on %s: %s", explainer.name, world.world_id, exc)
                skipped.append(
                    {
                        "explainer": explainer.name,
                        "regime": ecfg.regime,
                        "reason": f"mechanism: {exc}",
                    }
                )
                continue
        ctx = build_context(
            ecfg.regime,
            clf,
            world,
            train_x=data["x_train"],
            train_y=data["y_train"],
            constraints=constraints,
            budget=cfg.budget.model_dump(),
            mechanism=mech,
            mechanism_label=mech_label,
        )
        log.info("running %s under %s", explainer.name, ecfg.regime)
        t0 = time.perf_counter()
        try:
            expls = explainer.explain(factuals, factual_ids, targets, ctx)
        except (PermissionError, NotImplementedError) as exc:
            # the method cannot run on this world/regime: a structured skip, never a silent drop
            log.warning("skipping %s on %s: %s", explainer.name, world.world_id, exc)
            skipped.append({"explainer": explainer.name, "regime": ecfg.regime, "reason": str(exc)})
            continue
        runtime = time.perf_counter() - t0
        recs = evaluate_explanations(
            eval_world,
            eval_clf,
            factuals,
            expls,
            cfg.evaluation,
            constraints,
            iqr,
            ood,
            exogenous=exo_f,
        )
        q = mech_quality.get(mech_label, {})
        for r in recs:
            r["is_control"] = explainer.is_control
            r["mechanism_replay_error"] = q.get("replay_error")
            r["mechanism_graph_f1"] = q.get("graph_f1")
        _save_arrays(recs, expl_dir)
        all_records.extend(recs)
        for reading in sorted({r["reading"] for r in recs}):
            manifest = {
                "run_id": run_id,
                "world_id": world.world_id,
                "world_family": world.family,
                "world_seed": world.seed,
                "classifier_architecture": cfg.classifier.architecture,
                "classifier_seed": cfg.classifier.seed,
                "classifier_test_accuracy": clf.spec["test_accuracy"],
                "explainer": explainer.name,
                "explainer_class": ecfg.name,
                "explainer_seed": ecfg.seed,
                "mechanism_access_regime": ecfg.regime,
                "action_semantics": reading,
                "evaluation_seed": cfg.evaluation_seed,
                "dataset_hash": eval_dataset_hash,
                "evaluation_world_id": eval_world.world_id,
                "config_hash": config_hash,
                "git_commit": commit,
                "timestamp_utc": utc_timestamp(),
                "hardware": hardware_summary(),
                "runtime_seconds": runtime,
                "package_version": __version__,
                "n_factuals": len(idx),
            }
            manifests[f"{explainer.name}/{reading}"] = manifest
            append_manifest(out / "manifest.jsonl", manifest)
        _export_certificates(recs, eval_world, manifests, run_dir, cfg.certificates)
        _export_case_study(recs, expls, factuals, exo_f, eval_world, run_dir)

    summary = summarize(all_records, cfg.evaluation.faith_tol)
    write_json(run_dir / "records.json", all_records)
    write_json(run_dir / "summary.json", summary)
    table = markdown_table(summary)
    (run_dir / "summary.md").write_text(
        f"# {cfg.name}\n\nworld `{world.world_id}`, classifier {cfg.classifier.architecture} "
        f"(test acc {clf.spec['test_accuracy']:.3f}), {len(idx)} flip candidates, "
        f"target {cfg.target}, label site t={world.label.site(world.T)}.\n\n{table}\n",
        encoding="utf-8",
    )
    write_json(
        run_dir / "run_info.json",
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "code_fingerprint": code_fp,
            "git_commit": commit,
            "total_seconds": time.perf_counter() - t_start,
            "output_dir": str(out),
            "world_dir": str(pop_dir),
            "model_dir": str(model_dir),
            "shift": cfg.shift,
            "perturb": cfg.perturb,
            "explanations_dir": str(expl_dir),
            "skipped_explainers": skipped,
            "classifier": clf_reports,
            "mechanisms": mech_quality,
        },
    )
    log.info("wrote %s", run_dir)
    print(table)
    return run_dir


def _save_arrays(recs: list[dict[str, Any]], expl_dir: Path) -> None:
    """Persist proposals and replays per (explainer, reading) and strip them from the records."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        # one file per explainer, seed and reading: seeds of one method must not overwrite each other
        groups.setdefault(
            f"{r['explainer']}__s{r.get('explainer_seed')}__{r['reading']}", []
        ).append(r)
    expl_dir.mkdir(parents=True, exist_ok=True)
    for key, rs in groups.items():
        shape = next((r["_proposal"].shape for r in rs if r["_proposal"] is not None), (0, 0))

        def stack(field: str, rs: list[dict[str, Any]] = rs, shape: tuple = shape) -> np.ndarray:
            return np.stack(
                [r[field] if r[field] is not None else np.full(shape, np.nan) for r in rs]
            )

        np.savez_compressed(
            expl_dir / f"{key}.npz",
            explanation_ids=np.array([r["explanation_id"] for r in rs]),
            factual_ids=np.array([r["factual_id"] for r in rs]),
            proposals=stack("_proposal"),
            replays=stack("_replay"),
        )
        for r in rs:
            r["arrays_file"] = str(expl_dir / f"{key}.npz")
            del r["_proposal"], r["_replay"]


def _export_certificates(recs, world, manifests, run_dir: Path, n: int) -> None:
    by_key: dict[str, int] = {}
    for r in recs:
        key = f"{r['explainer']}/{r['reading']}"
        if not r["scorable"] or by_key.get(key, 0) >= n:
            continue
        by_key[key] = by_key.get(key, 0) + 1
        cert = make_certificate(r, world, manifests[key])
        write_certificate(
            run_dir / "certificates" / f"{r['explainer']}__{r['reading']}__"
            f"{r['explanation_id']}.json",
            cert,
        )


def _export_case_study(recs, expls, factuals, exo, world, run_dir: Path) -> None:
    from twist.actions.schemas import ActionPlan
    from twist.replay.pearl_replay import to_dense

    by_id = {e.explanation_id: (i, e) for i, e in enumerate(expls)}
    obs_index = list(getattr(world, "obs_index", range(world.k)))
    channels = [obs_index.index(c) for c in world.label.channels if c in obs_index] or [0]
    done: set[str] = set()
    for r in recs:
        if not r["scorable"] or r["reading"] in done:
            continue
        done.add(r["reading"])
        i, e = by_id[r["explanation_id"]]
        plan = ActionPlan.from_dict(r["action"])
        sched = to_dense(plan, world.T, world.k_state, world.action_space())
        rep, _ = world.replay_units(subset_exo(exo, i), plan)
        plot_case_study(
            factuals[i],
            e.proposed_trajectory,
            rep,
            channels,
            sched.active_timesteps.tolist()[:50],
            world.label.site(world.T),
            run_dir / "figures" / f"case__{r['explainer']}__{r['reading']}.png",
            title=f"{r['explainer']} [{r['reading']}] unit {r['factual_id']}",
        )
