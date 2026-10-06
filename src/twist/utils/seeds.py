"""Random-seed policy.

Every stochastic component receives its own integer seed recorded in the run manifest
(``world_seed``, ``classifier_seed``, ``explainer_seed``, ``evaluation_seed``). Child seeds are
derived deterministically from a parent seed plus string keys so that adding a component never
perturbs the seeds of existing components.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(parent: int, *keys: str | int) -> int:
    """Derive a stable 32-bit child seed from ``parent`` and an ordered tuple of keys."""
    payload = ":".join([str(parent), *map(str, keys)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and torch. Call once per process entry point."""
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002 — seeds third-party code using the legacy global RNG
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    import torch

    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure_threads(default: int = 1) -> int:
    """Pin intra-op threads. Small recurrent models are much slower when oversubscribed on shared
    nodes, so default to ``TWIST_NUM_THREADS``, else ``SLURM_CPUS_PER_TASK``, else ``default``."""
    import torch

    n = int(os.environ.get("TWIST_NUM_THREADS") or os.environ.get("SLURM_CPUS_PER_TASK") or default)
    torch.set_num_threads(n)
    try:  # BLAS / OpenMP pools of numpy and scikit-learn oversubscribe the same way
        from threadpoolctl import threadpool_limits

        threadpool_limits(n)
    except ImportError:
        pass
    return n
