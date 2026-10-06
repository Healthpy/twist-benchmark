"""Persisting and loading a world's unit-level data (observations, labels, exogenous variables)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_world_data(
    directory: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    exo: dict[str, np.ndarray],
    n_train: int,
) -> Path:
    """``data.npz`` holds ``x``, ``y``, ``n_train``, ``eps`` (= η) and, for partially observed
    worlds, every other exogenous array as ``exo_<name>``. The state equals ``x`` when absent."""
    arrays: dict[str, Any] = {"x": x, "y": y, "n_train": n_train, "eps": exo["eta"]}
    for name, arr in exo.items():
        if name == "eta" or (name == "state" and arr is x):
            continue
        arrays[f"exo_{name}"] = arr
    path = Path(directory) / "data.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_world_data(directory: str | Path) -> dict[str, Any]:
    d = dict(np.load(Path(directory) / "data.npz"))
    exo = {k[4:]: v for k, v in d.items() if k.startswith("exo_")}
    exo.setdefault("state", d["x"])
    exo["eta"] = d["eps"]
    return {"x": d["x"], "y": d["y"], "n_train": int(d["n_train"]), "exo": exo}


def subset_exo(exo: dict[str, np.ndarray], idx: Any) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in exo.items()}
