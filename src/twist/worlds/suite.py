"""Multi-world benchmark suite builder (plan §Week 4, task 5).

A suite config lists families, worlds per split, global sizes and the factor pools to sample
from. ``build_suite`` then, for each family × split × index:

1. derives the world seed from ``(suite seed, family, split, index, attempt)`` and samples one
   level per applicable factor from the split's pool;
2. generates the world and its population and runs ``check_world``. It also requires a relative
   functional distance ≥ ``min_family_distance`` to every world of the family accepted so far.
   Candidates that fail are rejected and resampled with the next attempt (pre-declared rejection
   sampling). Every rejection and its reasons are kept in the manifest;
3. writes the world directory. For splits listed in ``hidden``, the mechanism and the full card
   go to ``hidden/<world_id>/`` and the public directory keeps only data plus a redacted card.

It finally checks family diversity (distinct mechanisms) and writes ``manifest.json``,
``splits/<split>.txt``, ``integrity_report.md`` and ``DATASET_CARD.md``.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from twist.reporting.benchmark_cards import world_card_markdown
from twist.utils.logging import get_logger
from twist.utils.seeds import derive_seed
from twist.utils.serialization import read_json, utc_timestamp, write_json
from twist.worlds.factors import APPLICABLE, FAMILIES, LEVELS, resolve_world
from twist.worlds.integrity import check_family_diversity, check_world, mechanism_distance

log = get_logger(__name__)
SPLITS = ("dev", "val", "test", "challenge")
PUBLIC_CARD_FIELDS = (
    "world_id",
    "world_family",
    "k",
    "T",
    "label",
    "variable_names",
    "assumptions",
)


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 2026
    output_dir: str = "artifacts/suites"
    families: list[str] = Field(default_factory=lambda: list(FAMILIES))
    worlds_per_split: dict[str, int] = Field(
        default_factory=lambda: {"dev": 10, "val": 10, "test": 20, "challenge": 10}
    )
    k: int = 6
    T: int = 50
    n_samples: int = 1000
    train_frac: float = 0.8
    max_attempts: int = 5
    min_family_distance: float = 0.05  # acceptance: relative functional distance to accepted worlds
    hidden: list[str] = Field(default_factory=lambda: ["test", "challenge"])
    # hidden splits derive their seeds from this env var when set; otherwise from the public seed
    # (then hiding is a no-peeking discipline, not protection — disclosed in the dataset card)
    secret_env: str = "TWIST_SUITE_SECRET"
    # factor pools: {"standard": {factor: [levels]}, "challenge": {...}}; missing → all levels
    pools: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


def _pool(cfg: SuiteConfig, split: str) -> dict[str, list[str]]:
    base = {name: list(levels) for name, levels in LEVELS.items()}
    base.update(cfg.pools.get("standard", {}))
    if split == "challenge":
        base.update(cfg.pools.get("challenge", {}))
    return base


def _sample_factors(
    rng: np.random.Generator, family: str, pool: dict[str, list[str]]
) -> dict[str, str]:
    return {name: str(rng.choice(pool[name])) for name in APPLICABLE[family]}


def build_suite(cfg: SuiteConfig) -> Path:
    from twist.evaluation.paired_runner import generate_world_data

    final = Path(cfg.output_dir) / cfg.name
    if final.exists():
        raise FileExistsError(f"{final} exists; suites are immutable — choose a new name")
    root = final.parent / f"{final.name}.partial"  # renamed only after a complete, checked build
    if root.exists():
        shutil.rmtree(root)
    secret = os.environ.get(cfg.secret_env)
    hidden_seed_source = "secret" if secret else "public"
    entries: list[dict[str, Any]] = []
    built: dict[str, list[Any]] = defaultdict(list)
    for family in cfg.families:
        for split in SPLITS:
            for idx in range(cfg.worlds_per_split.get(split, 0)):
                world_id = f"{family}_{split}_{idx:03d}"
                rejections = []
                for attempt in range(cfg.max_attempts):
                    hide = bool(secret) and split in cfg.hidden
                    base_seed = derive_seed(0, "suite-secret", str(secret)) if hide else cfg.seed
                    seed = derive_seed(base_seed, family, split, idx, attempt)
                    factors = _sample_factors(
                        np.random.default_rng(seed), family, _pool(cfg, split)
                    )
                    wcfg = resolve_world(
                        family, factors, world_id, seed, cfg.k, cfg.T, cfg.n_samples, cfg.train_frac
                    )
                    data = generate_world_data(wcfg)
                    x = np.concatenate([data["x_train"], data["x_test"]])
                    exo = {
                        n: np.concatenate([data["exo_train"][n], data["exo_test"][n]])
                        for n in data["exo_train"]
                    }
                    integrity = check_world(data["world"], x, exo)
                    failed = [
                        c["check"] + ": " + c["detail"]
                        for c in integrity["checks"]
                        if not c["passed"]
                    ]
                    if not failed:  # diversity: must differ from every accepted world of the family
                        for other in built[family]:
                            dist = mechanism_distance(data["world"], other)
                            if dist < cfg.min_family_distance:
                                failed.append(f"too_similar: d={dist:.3g} to {other.world_id}")
                                break
                    if not failed:
                        break
                    rejections.append(
                        {"attempt": attempt, "seed": seed, "factors": factors, "failed": failed}
                    )
                else:
                    raise RuntimeError(
                        f"{world_id}: no world passed integrity in "
                        f"{cfg.max_attempts} attempts: {rejections}"
                    )
                world = data["world"]
                entry = _write_world(root, split, world, wcfg, data, x, exo, integrity, cfg)
                entry.update(
                    family=family,
                    split=split,
                    index=idx,
                    attempt=attempt,
                    seed=seed,
                    factors=factors,
                    rejections=rejections,
                )
                entries.append(entry)
                built[family].append(world)
                log.info("%s: attempt %d, %s", world_id, attempt, factors)

    diversity = {
        fam: check_family_diversity(ws, cfg.min_family_distance) for fam, ws in built.items()
    }
    private_keys = ("seed", "factors", "rejections")
    public_entries = [
        {k: v for k, v in e.items() if not (e["visibility"] == "hidden" and k in private_keys)}
        for e in entries
    ]
    write_json(root / "hidden" / "manifest_private.json", {"suite": cfg.name, "worlds": entries})
    manifest = {
        "suite": cfg.name,
        "created_utc": utc_timestamp(),
        "config": cfg.model_dump(),
        "hidden_seed_source": hidden_seed_source,
        "worlds": public_entries,
        "family_diversity": diversity,
        "passed": all(e["integrity_passed"] for e in entries)
        and all(d["passed"] for d in diversity.values()),
    }
    write_json(root / "manifest.json", manifest)
    for split in SPLITS:
        ids = [e["world_id"] for e in entries if e["split"] == split]
        (root / "splits").mkdir(parents=True, exist_ok=True)
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    full = {**manifest, "worlds": entries}
    (root / "integrity_report.md").write_text(integrity_markdown(full), encoding="utf-8")
    (root / "DATASET_CARD.md").write_text(dataset_card_markdown(full), encoding="utf-8")
    root.rename(final)
    return final


def _write_world(
    root: Path,
    split: str,
    world: Any,
    wcfg: Any,
    data: dict[str, Any],
    x: np.ndarray,
    exo: dict[str, np.ndarray],
    integrity: dict[str, Any],
    cfg: SuiteConfig,
) -> dict[str, Any]:
    from twist.worlds.io import save_world_data

    hidden = split in cfg.hidden
    public = root / "worlds" / world.world_id
    private = root / "hidden" / world.world_id if hidden else public
    public.mkdir(parents=True, exist_ok=True)
    world.save(private)
    y = np.concatenate([data["y_train"], data["y_test"]])
    n_train = len(data["x_train"])
    save_world_data(public, x, y, exo, n_train)
    write_json(private / "integrity.json", integrity)
    write_json(private / "world_config.json", wcfg)
    (private / "WORLD_CARD.md").write_text(world_card_markdown(world, integrity), encoding="utf-8")
    card = world.card()
    if hidden:
        redacted = {key: card[key] for key in PUBLIC_CARD_FIELDS if key in card}
        redacted["visibility"] = "hidden"
        redacted["mechanism_location"] = str(private.relative_to(root))
        write_json(public / "world_card.json", redacted)
    exact = bool(world.assumptions.get("exact_abduction", world.assumptions.get("exact_replay")))
    return {
        "world_id": world.world_id,
        "path": str(public.relative_to(root)),
        "visibility": "hidden" if hidden else "visible",
        "mechanism_hash": card["mechanism_hash"],
        "dataset_hash": data["dataset_hash"],
        "k": world.k,
        "k_state": world.k_state,
        "T": world.T,
        "n_samples": len(x),
        "n_train": n_train,
        "label_type": card["label"]["type"],
        "label_balance": integrity["label_balance"],
        "route_effect": integrity["route_effect"],
        "exact_abduction": exact,
        "integrity_passed": integrity["passed"],
    }


def load_suite_world(suite_dir: str | Path, world_id: str, allow_hidden: bool = False) -> Any:
    """Load a world's mechanism. Hidden (test/challenge) mechanisms require ``allow_hidden``."""
    from twist.worlds.base_scm import TemporalSCM

    root = Path(suite_dir)
    hidden = root / "hidden" / world_id
    if hidden.exists():
        if not allow_hidden:
            raise PermissionError(
                f"{world_id} has a hidden mechanism; pass allow_hidden=True "
                "only for final evaluation"
            )
        return TemporalSCM.load(hidden)
    return TemporalSCM.load(root / "worlds" / world_id)


# ----------------------------------------------------------------------------- reports
def integrity_markdown(manifest: dict[str, Any]) -> str:
    worlds = manifest["worlds"]
    n_rej = sum(len(e["rejections"]) for e in worlds)
    reasons = Counter(f.split(":")[0] for e in worlds for r in e["rejections"] for f in r["failed"])
    lines = [
        f"# Integrity report — suite `{manifest['suite']}`",
        "",
        f"**{'PASS' if manifest['passed'] else 'FAIL'}** — {len(worlds)} worlds accepted, "
        f"{n_rej} candidate worlds rejected before acceptance.",
        "",
        "Rejection reasons (failed checks across rejected candidates): "
        + (", ".join(f"`{k}` ×{v}" for k, v in reasons.most_common()) or "none"),
        "",
        "## Family diversity",
        "",
        "| family | worlds | distinct mechanisms | min relative functional distance | result |",
        "|---|---|---|---|---|",
    ]
    for fam, d in manifest["family_diversity"].items():
        lines.append(
            f"| {fam} | {d['n_worlds']} | {d['distinct_hashes']} | "
            f"{d['min_relative_distance']:.3g} | {'pass' if d['passed'] else '**FAIL**'} |"
        )
    lines += [
        "",
        "## Worlds",
        "",
        "| world | split | attempt | label | P(y=1) | route effect | exact abduction | integrity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in worlds:
        lines.append(
            f"| {e['world_id']} | {e['split']} | {e['attempt']} | {e['label_type']} | "
            f"{e['label_balance']:.2f} | {e['route_effect']:.3g} | {e['exact_abduction']} | "
            f"{'pass' if e['integrity_passed'] else '**FAIL**'} |"
        )
    return "\n".join(lines) + "\n"


def dataset_card_markdown(manifest: dict[str, Any]) -> str:
    worlds = manifest["worlds"]
    cfg = manifest["config"]
    fams = sorted({e["family"] for e in worlds})
    lines = [
        f"# Dataset card — TWIST suite `{manifest['suite']}`",
        "",
        f"Created {manifest['created_utc']} from suite seed {cfg['seed']}. {len(worlds)} worlds; "
        f"each has {cfg['n_samples']} units of length T={cfg['T']} "
        f"({int(cfg['train_frac'] * 100)}% train / rest test), k={cfg['k']} state channels.",
        "",
        "## Worlds per family and split",
        "",
        "| family | " + " | ".join(SPLITS) + " |",
        "|---|" + "---|" * len(SPLITS),
    ]
    for fam in fams:
        counts = Counter(e["split"] for e in worlds if e["family"] == fam)
        lines.append(f"| {fam} | " + " | ".join(str(counts.get(s, 0)) for s in SPLITS) + " |")
    lines += [
        "",
        "## Visibility",
        f"- splits with hidden mechanisms: {cfg['hidden']}. Their public directories contain the "
        "data and a redacted card. Mechanisms, roles, configs, seeds and full cards are under "
        "`hidden/` and load only with `load_suite_world(..., allow_hidden=True)`.",
        f"- hidden-split seeds derive from the **{manifest['hidden_seed_source']}** seed"
        + (
            " (a secret held outside the repository)."
            if manifest["hidden_seed_source"] == "secret"
            else ". Anyone with the public suite seed and the code can regenerate them, so hiding "
            "is a no-peeking discipline for development, not protection."
        ),
        "",
        "## Replay exactness",
    ]
    exact = Counter(e["exact_abduction"] for e in worlds)
    lines += [
        f"- {exact.get(True, 0)} worlds allow **exact abduction** from observations "
        "(additive-noise, fully observed, unconfounded): all mechanism metrics are defined.",
        f"- {exact.get(False, 0)} worlds are **partially observed** (`statespace`). The oracle "
        "replays from stored exogenous variables; abduction-based metrics are NaN; explainers "
        "that abduct from `x` need posterior replay.",
        "",
        "## Factor distribution",
        "",
        "| factor | level | " + " | ".join(SPLITS) + " |",
        "|---|---|" + "---|" * len(SPLITS),
    ]
    tallies: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for e in worlds:
        for name, level in e["factors"].items():
            tallies[(name, level)][e["split"]] += 1
    for (name, level), c in sorted(tallies.items()):
        lines.append(f"| {name} | {level} | " + " | ".join(str(c.get(s, 0)) for s in SPLITS) + " |")
    lines += [
        "",
        "## Label types",
        "- " + ", ".join(f"{k}: {v}" for k, v in Counter(e["label_type"] for e in worlds).items()),
        "",
        "## Generation and acceptance",
        "- Every world passed `twist.worlds.integrity.check_world` (stability, graph, action "
        "space, label balance in [0.2, 0.8], graph and empirical causal route, distractor, replay, "
        "exactness disclosure). Rejected candidates are listed in `manifest.json`.",
        "- Family diversity: distinct mechanism hashes and pairwise relative functional distance "
        "≥ 0.05 (`integrity_report.md`).",
        "",
        "## Claim boundary",
        "- These worlds define causal truth by construction. They are evaluation oracles, not "
        "models of any real domain.",
    ]
    return "\n".join(lines) + "\n"


def load_manifest(suite_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(suite_dir) / "manifest.json")
