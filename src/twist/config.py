"""Pydantic-validated YAML configuration.

An experiment config may reference component configs by path relative to the ``configs/``
directory (e.g. ``world: worlds/linear_toy.yaml``); they are resolved and validated on load, and
the fully resolved config is what gets hashed into ``config_hash``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorldConfig(_Strict):
    family: Literal["linear", "nonlinear", "spring", "regime", "statespace", "energy"]
    world_id: str
    seed: int
    n_samples: int = 10_000
    train_frac: float = 0.8
    params: dict[str, Any] = Field(default_factory=dict)
    noise: dict[str, Any] = Field(default_factory=lambda: {"kind": "laplace", "scale": 0.1})
    label: dict[str, Any] = Field(default_factory=lambda: {"channel": 0, "t_label": -1})
    # actionable channels, value bounds and high-level controls (see docs/action_schema.md)
    action_space: dict[str, Any] = Field(default_factory=dict)


class ClassifierConfig(_Strict):
    architecture: Literal["gru", "lstm", "tcn", "transformer", "logreg"] = "gru"
    hidden: int = 32
    layers: int = 1
    epochs: int = 10
    lr: float = 3e-3
    batch_size: int = 128
    seed: int = 101
    heads: int = 4  # transformer
    kernel: int = 3  # tcn
    l2: float = 1.0  # logreg inverse regularization (scikit-learn C)


class ExplainerConfig(_Strict):
    name: str
    seed: int = 501
    regime: str = "R0"
    params: dict[str, Any] = Field(default_factory=dict)
    # R3/R4: how the runner builds the explainer's mechanism from the observational data, e.g.
    # {estimator: lasso} | {estimator: neural, epochs: 40} | {estimator: masked} (R3) |
    # {misspecify: {kind: delay, level: 0.5}} (controlled wrong mechanism)
    mechanism: dict[str, Any] | None = None


class EvaluationConfig(_Strict):
    readings: list[str] = Field(
        default_factory=lambda: ["first_change_atomic", "full_changed_schedule"]
    )
    change_tol: float = 1e-3
    faith_tol: float = 1e-3
    soft_sigma: float = 1.0
    compute_dtw: bool = True
    # >0: also estimate P(world target | do(action)) with this many fresh-noise rollouts
    interventional_samples: int = 0
    dtw_window: int | None = None
    constraints: list[dict[str, Any]] = Field(default_factory=list)


class BudgetConfig(_Strict):
    """Per-unit limits shared by every explainer (plan §Week 5, task 4)."""

    max_queries: int | None = 200_000
    time_limit_s: float | None = 120.0
    max_reference: int = 500


class ExperimentConfig(_Strict):
    name: str
    output_dir: str = "artifacts"
    evaluation_seed: int = 9001
    world: WorldConfig
    classifier: ClassifierConfig
    explainers: list[ExplainerConfig]
    n_factuals: int = 100
    target: int | Literal["flip"] = 1
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    # Week 7: evaluate in a shifted deployment world ({level, kind, amount}; see worlds.shifts) and/or
    # on innovation-perturbed copies of the factual units ({scale}); training stays on the base world
    shift: dict[str, Any] | None = None
    perturb: dict[str, Any] | None = None
    certificates: int = 1  # number of explanation certificates to export per explainer/reading

    @model_validator(mode="before")
    @classmethod
    def _resolve_refs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key in ("world", "classifier"):
            if isinstance(data.get(key), str):
                data[key] = load_yaml(data[key])
        if "explainers" in data:
            data["explainers"] = [
                load_yaml(e) if isinstance(e, str) else e for e in data["explainers"]
            ]
        return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_ROOT / p
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_world_config(path: str | Path) -> WorldConfig:
    return WorldConfig.model_validate(load_yaml(path))
