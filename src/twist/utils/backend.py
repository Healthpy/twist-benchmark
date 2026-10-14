"""Minimal NumPy/PyTorch dispatch so mechanisms and replay share one implementation.

World mechanisms and the replay recursion are written once and run on either NumPy arrays
(benchmark oracle) or torch tensors (differentiable structural controls).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # torch is a hard dependency, but keep the import local to this module.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

Array = Any  # np.ndarray | torch.Tensor


def is_torch(x: Array) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def stack(xs: list[Array], axis: int) -> Array:
    if is_torch(xs[0]):
        return torch.stack(xs, dim=axis)
    return np.stack(xs, axis=axis)


def where(mask: Array, a: Array, b: Array) -> Array:
    if is_torch(a) or is_torch(b):
        mask_t = mask if is_torch(mask) else torch.as_tensor(mask)
        a_t = a if is_torch(a) else torch.as_tensor(a, dtype=b.dtype)
        b_t = b if is_torch(b) else torch.as_tensor(b, dtype=a_t.dtype)
        return torch.where(mask_t, a_t, b_t)
    return np.where(mask, a, b)


def tanh(x: Array) -> Array:
    return torch.tanh(x) if is_torch(x) else np.tanh(x)


def as_like(x: Any, ref: Array) -> Array:
    """Convert a NumPy constant to the backend/dtype of ``ref``."""
    if is_torch(ref):
        return torch.as_tensor(np.asarray(x), dtype=ref.dtype, device=ref.device)
    return np.asarray(x, dtype=ref.dtype)


def zeros_like(x: Array) -> Array:
    return torch.zeros_like(x) if is_torch(x) else np.zeros_like(x)


def einsum(spec: str, *ops: Array) -> Array:
    if any(is_torch(o) for o in ops):
        ref = next(o for o in ops if is_torch(o))
        return torch.einsum(spec, *[o if is_torch(o) else as_like(o, ref) for o in ops])
    return np.einsum(spec, *ops)


def softplus(x: Array) -> Array:
    """``log(1 + exp(x))``, overflow-safe on both backends."""
    return torch.nn.functional.softplus(x) if is_torch(x) else np.logaddexp(0.0, x)


def sigmoid(x: Array) -> Array:
    return 0.5 * (1.0 + tanh(0.5 * x))


def relu(x: Array) -> Array:
    return 0.5 * (x + abs(x))
