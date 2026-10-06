"""Shape and value checks shared across modules."""

from __future__ import annotations

import numpy as np


class TwistValidationError(ValueError):
    """Raised when an input violates a benchmark contract."""


def check_trajectory(x: np.ndarray, T: int | None = None, k: int | None = None) -> np.ndarray:
    """Validate a single ``(T, k)`` or batched ``(B, T, k)`` trajectory array."""
    x = np.asarray(x)
    if x.ndim not in (2, 3):
        raise TwistValidationError(f"trajectory must be (T,k) or (B,T,k); got shape {x.shape}")
    if T is not None and x.shape[-2] != T:
        raise TwistValidationError(f"expected T={T}, got {x.shape[-2]}")
    if k is not None and x.shape[-1] != k:
        raise TwistValidationError(f"expected k={k}, got {x.shape[-1]}")
    if not np.all(np.isfinite(x)):
        raise TwistValidationError("trajectory contains non-finite values")
    return x


def as_batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(B,T,k)`` view and whether the input was a single unit."""
    x = np.asarray(x)
    return (x[None], True) if x.ndim == 2 else (x, False)
