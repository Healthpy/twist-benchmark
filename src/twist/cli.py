"""Command-line entry point: ``twist <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from twist.config import load_experiment, load_world_config


def _toy(args: argparse.Namespace) -> None:
    from twist.evaluation.paired_runner import run_paired_experiment

    cfg = load_experiment(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_paired_experiment(cfg)


def _generate_world(args: argparse.Namespace) -> None:
    from twist.evaluation.paired_runner import generate_world_data

    for path in args.configs:
        cfg = load_world_config(path)
        out = Path(args.output_dir) / "worlds" / cfg.world_id
        generate_world_data(cfg, out)
        print(f"wrote {out}")


def _train_classifier(args: argparse.Namespace) -> None:
    import numpy as np

    from twist.classifiers.train import train_classifier
    from twist.config import ClassifierConfig, load_yaml

    d = Path(args.world_dir)
    data = np.load(d / "data.npz")
    n = int(data["n_train"])
    cfg = ClassifierConfig.model_validate(load_yaml(args.config))
    clf = train_classifier(
        data["x"][:n], data["y"][:n], data["x"][n:], data["y"][n:], **cfg.model_dump()
    )
    out = Path(args.output_dir) / "models" / f"{d.name}__{cfg.architecture}_seed_{cfg.seed}"
    clf.save(out)
    print(f"wrote {out} (test accuracy {clf.spec['test_accuracy']:.3f})")


def _build_suite(args: argparse.Namespace) -> None:
    from twist.config import load_yaml
    from twist.worlds.suite import SuiteConfig, build_suite

    cfg = SuiteConfig.model_validate(load_yaml(args.config))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    root = build_suite(cfg)
    print((root / "integrity_report.md").read_text().split("## Worlds")[0])
    print(f"wrote {root}")


def _check_worlds(args: argparse.Namespace) -> None:
    from twist.worlds.integrity import check_world
    from twist.worlds.io import load_world_data
    from twist.worlds.suite import load_manifest, load_suite_world

    manifest = load_manifest(args.suite_dir)
    failed = []
    for e in manifest["worlds"]:
        world = load_suite_world(args.suite_dir, e["world_id"], allow_hidden=True)
        data = load_world_data(f"{args.suite_dir}/{e['path']}")
        rep = check_world(world, data["x"], data["exo"])
        status = "pass" if rep["passed"] else "FAIL"
        print(f"{e['world_id']:28s} {status}")
        if not rep["passed"]:
            failed.append(e["world_id"])
    if failed:
        raise SystemExit(f"{len(failed)} worlds failed: {failed}")


def _pilot(args: argparse.Namespace) -> None:
    from twist.config import load_yaml
    from twist.evaluation.multiworld_runner import MultiWorldConfig, run_multiworld
    from twist.evaluation.pilot_report import build_pilot_report

    cfg = MultiWorldConfig.model_validate(load_yaml(args.config))
    if args.jobs:
        cfg.jobs = args.jobs
    run_dirs = run_multiworld(cfg)
    path = build_pilot_report(cfg.name, run_dirs, Path(cfg.output_dir) / cfg.name / "report")
    print(f"wrote {path}")


def _pilot_report(args: argparse.Namespace) -> None:
    from twist.evaluation.pilot_report import build_pilot_report

    dirs = [Path(d) for d in args.run_dirs]
    print(build_pilot_report(args.name, dirs, dirs[0].parent.parent / "report"))


def _ladder(args: argparse.Namespace) -> None:
    from twist.config import load_yaml
    from twist.evaluation.ladder_report import build_ladder_report
    from twist.evaluation.multiworld_runner import MultiWorldConfig, run_multiworld

    cfg = MultiWorldConfig.model_validate(load_yaml(args.config))
    if args.jobs:
        cfg.jobs = args.jobs
    run_dirs = run_multiworld(cfg)
    path = build_ladder_report(cfg.name, run_dirs, Path(cfg.output_dir) / cfg.name / "report")
    print(f"wrote {path}")


def _robustness(args: argparse.Namespace) -> None:
    from twist.evaluation.robustness import (
        build_robustness_report,
        load_robustness_config,
        run_robustness,
    )

    cfg = load_robustness_config(args.config)
    if args.jobs:
        cfg.jobs = args.jobs
    print(f"wrote {build_robustness_report(cfg, run_robustness(cfg))}")


def _validity(args: argparse.Namespace) -> None:
    from twist.evaluation.validity import load_validity_config, run_validity

    print(f"wrote {run_validity(load_validity_config(args.config))}")


def _study(args: argparse.Namespace) -> None:
    from twist.evaluation.study import build_study_report, load_study_config, run_study

    cfg = load_study_config(args.config)
    if args.jobs:
        cfg.jobs = args.jobs
    if args.report_only:
        from twist.evaluation.multiworld_runner import completed_run, make_tasks

        # rebuilding a report reuses finished runs of this config even if the code changed since
        runs = [
            d
            for exp, _ in make_tasks(cfg)
            if (d := completed_run(exp, same_code=False)) is not None
        ]
        print(f"wrote {build_study_report(cfg, runs)}")
    else:
        print(f"wrote {run_study(cfg)}")


def _discovery(args: argparse.Namespace) -> None:
    from twist.evaluation.discovery_report import (
        build_discovery_report,
        load_discovery_config,
        run_discovery,
    )

    cfg = load_discovery_config(args.config)
    if args.jobs:
        cfg.jobs = args.jobs
    if args.report_only:
        from twist.evaluation.multiworld_runner import completed_run, make_tasks

        runs = [
            d
            for exp, _ in make_tasks(cfg)
            if (d := completed_run(exp, same_code=False)) is not None
        ]
        print(f"wrote {build_discovery_report(cfg, runs)}")
    else:
        print(f"wrote {run_discovery(cfg)}")


def _compat(args: argparse.Namespace) -> None:
    from twist.reporting.compatibility import compatibility_markdown

    Path(args.output).write_text(compatibility_markdown(), encoding="utf-8")
    print(f"wrote {args.output}")


def _audit(args: argparse.Namespace) -> None:
    from twist.evaluation.audit import audit_markdown, audit_run

    report = audit_run(args.run_dir)
    print(audit_markdown(report))
    if not report["passed"]:
        raise SystemExit(1)


def _sanity(args: argparse.Namespace) -> None:
    from twist.evaluation.sanity import run_sanity

    path = run_sanity(args.config, args.output_dir)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="twist", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a paired experiment from an experiment config")
    p.add_argument("--config", default="experiments/toy.yaml")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_toy)

    p = sub.add_parser(
        "toy", help="end-to-end toy run (alias of `run --config experiments/toy.yaml`)"
    )
    p.add_argument("--config", default="experiments/toy.yaml")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_toy)

    p = sub.add_parser("generate-world", help="generate world(s) and their data splits")
    p.add_argument("configs", nargs="+")
    p.add_argument("--output-dir", default="artifacts")
    p.set_defaults(func=_generate_world)

    p = sub.add_parser("train-classifier", help="train a classifier on a generated world")
    p.add_argument("--world-dir", required=True)
    p.add_argument("--config", default="classifiers/gru_small.yaml")
    p.add_argument("--output-dir", default="artifacts")
    p.set_defaults(func=_train_classifier)

    p = sub.add_parser("build-suite", help="generate a multi-world suite with splits and checks")
    p.add_argument("--config", default="suites/dev.yaml")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_build_suite)

    p = sub.add_parser("check-worlds", help="re-run integrity checks on every world of a suite")
    p.add_argument("suite_dir")
    p.set_defaults(func=_check_worlds)

    p = sub.add_parser("pilot", help="multi-world pilot: worlds × classifiers × explainers × seeds")
    p.add_argument("--config", default="experiments/pilot.yaml")
    p.add_argument("--jobs", type=int, default=None)
    p.set_defaults(func=_pilot)

    p = sub.add_parser("pilot-report", help="rebuild the pilot dashboard from run directories")
    p.add_argument("name")
    p.add_argument("run_dirs", nargs="+")
    p.set_defaults(func=_pilot_report)

    p = sub.add_parser("ladder", help="mechanism-access ladder + misspecification sweep + report")
    p.add_argument("--config", default="experiments/access_ladder.yaml")
    p.add_argument("--jobs", type=int, default=None)
    p.set_defaults(func=_ladder)

    p = sub.add_parser("robustness", help="shifts, perturbation stability, uncertainty, tolerance")
    p.add_argument("--config", default="experiments/robustness.yaml")
    p.add_argument("--jobs", type=int, default=None)
    p.set_defaults(func=_robustness)

    p = sub.add_parser("validity", help="construct and discriminant validity of the metrics")
    p.add_argument("--config", default="experiments/validity.yaml")
    p.set_defaults(func=_validity)

    p = sub.add_parser("study", help="pilot multi-world study: paired statistics, controls, report")
    p.add_argument("--config", default="experiments/study.yaml")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument(
        "--report-only", action="store_true", help="rebuild the report from finished runs"
    )
    p.set_defaults(func=_study)

    p = sub.add_parser(
        "discovery", help="causal discovery as a mechanism source (graph vs recourse)"
    )
    p.add_argument("--config", default="experiments/discovery.yaml")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument(
        "--report-only", action="store_true", help="rebuild the report from finished runs"
    )
    p.set_defaults(func=_discovery)

    p = sub.add_parser("compat", help="write the explainer compatibility matrix")
    p.add_argument("--output", default="docs/compatibility_matrix.md")
    p.set_defaults(func=_compat)

    p = sub.add_parser("audit", help="traceability audit of a finished run directory")
    p.add_argument("run_dir")
    p.set_defaults(func=_audit)

    p = sub.add_parser("sanity", help="metric sanity report over the control suite and worlds")
    p.add_argument("--config", default="experiments/sanity.yaml")
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=_sanity)

    args = parser.parse_args(argv)
    from twist.utils.seeds import configure_threads

    configure_threads()
    args.func(args)


if __name__ == "__main__":
    main()
