"""Generator factors (plan §Week 4, task 3) and their mapping onto family parameters.

A world is specified by a family and one level per applicable factor. ``resolve_world`` turns
that into a :class:`~twist.config.WorldConfig`. Factor levels are named, so suite manifests stay
readable and the mapping lives in one place.

| Factor | Levels | Applies to |
|---|---|---|
| ``lag`` | L1, L3, L5 | linear, nonlinear, regime (spring is first-order) |
| ``density`` | sparse, moderate, dense | all |
| ``strength`` | weak, medium, strong | all (entry range / gain / spring constant) |
| ``stability`` | contractive, weakly_stable, oscillatory | all (radius cap / decay / damping) |
| ``noise`` | gaussian, laplace, heteroscedastic, correlated | all |
| ``label`` | terminal, interior, exposure, onset, multi | all |
| ``regime_driver`` | state, exogenous | regime |
| ``state_family`` | linear, nonlinear, spring, regime | statespace |
| ``observation`` | identity, noisy, nonlinear, partial | statespace |
| ``missingness`` | none, mcar, block | statespace |
| ``sampling`` | regular, irregular | statespace |
| ``confounding`` | none, latent | statespace (latent forces a linear state) |

Exact-replay families never get latent confounding, missingness or non-identity observation.
Those factors exist only for the partially observed ``statespace`` family, whose worlds are
flagged ``exact_abduction: false``.
"""

from __future__ import annotations

from typing import Any

from twist.config import WorldConfig

FAMILIES = ("linear", "nonlinear", "spring", "regime", "statespace")

LEVELS: dict[str, tuple[str, ...]] = {
    "lag": ("L1", "L3", "L5"),
    "density": ("sparse", "moderate", "dense"),
    "strength": ("weak", "medium", "strong"),
    "stability": ("contractive", "weakly_stable", "oscillatory"),
    "noise": ("gaussian", "laplace", "heteroscedastic", "correlated"),
    "label": ("terminal", "interior", "exposure", "onset", "multi"),
    "regime_driver": ("state", "exogenous"),
    "state_family": ("linear", "nonlinear", "spring", "regime"),
    "observation": ("identity", "noisy", "nonlinear", "partial"),
    "missingness": ("none", "mcar", "block"),
    "sampling": ("regular", "irregular"),
    "confounding": ("none", "latent"),
}

APPLICABLE: dict[str, tuple[str, ...]] = {
    "linear": ("lag", "density", "strength", "stability", "noise", "label"),
    "nonlinear": ("lag", "density", "strength", "stability", "noise", "label"),
    "spring": ("density", "strength", "stability", "noise", "label"),
    "regime": ("lag", "density", "strength", "stability", "noise", "label", "regime_driver"),
    "statespace": (
        "state_family",
        "density",
        "strength",
        "stability",
        "noise",
        "label",
        "observation",
        "missingness",
        "sampling",
        "confounding",
    ),
}

DEFAULTS: dict[str, str] = {
    "lag": "L1",
    "density": "moderate",
    "strength": "medium",
    "stability": "weakly_stable",
    "noise": "laplace",
    "label": "terminal",
    "regime_driver": "state",
    "state_family": "linear",
    "observation": "noisy",
    "missingness": "none",
    "sampling": "regular",
    "confounding": "none",
}

_DENSITY = {"sparse": 0.15, "moderate": 0.25, "dense": 0.5}
_NOISE = {
    "gaussian": {"kind": "gaussian", "scale": 0.1},
    "laplace": {"kind": "laplace", "scale": 0.1},
    "heteroscedastic": {"kind": "laplace", "scale": 0.1, "heteroscedastic": 1.0},
    "correlated": {"kind": "gaussian", "scale": 0.1, "correlation": 0.5},
}


def _family_params(family: str, f: dict[str, str], k: int, T: int) -> dict[str, Any]:
    density = _DENSITY[f["density"]]
    if family in ("linear", "regime"):
        p: dict[str, Any] = {
            "k": k,
            "T": T,
            "L": int(f.get("lag", "L1")[1:]),
            "density": density,
            "strength": {"weak": 0.2, "medium": 0.5, "strong": 1.0}[f["strength"]],
            "spectral_radius": {"contractive": 0.6, "weakly_stable": 0.95, "oscillatory": 0.95}[
                f["stability"]
            ],
            "rescale": "cap",
            "ensure_route": True,
            "distractor": True,
        }
        if family == "linear":
            p["oscillatory"] = f["stability"] == "oscillatory"
        else:
            p["driver"] = f.get("regime_driver", "state")
        return p
    if family == "nonlinear":
        return {
            "k": k,
            "T": T,
            "L": int(f.get("lag", "L1")[1:]),
            "density": density,
            "gain": {"weak": 0.75, "medium": 1.5, "strong": 3.0}[f["strength"]],
            "decay_range": {
                "contractive": [0.1, 0.4],
                "weakly_stable": [0.6, 0.9],
                "oscillatory": [-0.9, -0.6],
            }[f["stability"]],
            "ensure_route": True,
            "distractor": True,
        }
    if family == "spring":
        return {
            "n_particles": max(k // 2, 5),
            "T": T,
            "density": max(density, 0.3),
            "heterogeneous": True,
            "spring_constant": {"weak": 0.1, "medium": 0.3, "strong": 0.6}[f["strength"]],
            "damping": {"contractive": 0.5, "weakly_stable": 0.1, "oscillatory": 0.0}[
                f["stability"]
            ],
            "anchor": {"contractive": 0.2, "weakly_stable": 0.05, "oscillatory": 0.0}[
                f["stability"]
            ],
            "n_exogenous": 1,
            "ensure_route": True,
            "distractor": True,
        }
    raise KeyError(family)


def _label_spec(level: str, T: int) -> dict[str, Any]:
    if level == "terminal":
        return {"type": "threshold", "channel": 0, "t_label": -1}
    if level == "interior":
        return {"type": "threshold", "channel": 0, "t_label": int(0.6 * T)}
    if level == "exposure":
        return {"type": "exposure", "channel": 0, "t_start": -max(T // 4, 2), "t_end": -1}
    if level == "onset":
        return {"type": "onset", "channel": 0, "t_start": T // 2, "t_end": -1}
    if level == "multi":  # label channel at the end AND its route source late in the window
        return {
            "type": "multi",
            "combine": "and",
            "conditions": [
                {"type": "threshold", "channel": 0, "t_label": -1},
                {"type": "exposure", "channel": 1, "t_start": -max(T // 4, 2), "t_end": -2},
            ],
        }
    raise KeyError(level)


def resolve_world(
    family: str,
    factors: dict[str, str],
    world_id: str,
    seed: int,
    k: int = 6,
    T: int = 50,
    n_samples: int = 1000,
    train_frac: float = 0.8,
) -> WorldConfig:
    """Map named factor levels to a validated ``WorldConfig`` (missing factors use DEFAULTS)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}")
    f = {name: factors.get(name, DEFAULTS[name]) for name in APPLICABLE[family]}
    for name, level in f.items():
        if level not in LEVELS[name]:
            raise ValueError(f"factor {name!r} has no level {level!r}; levels: {LEVELS[name]}")
    noise = dict(_NOISE[f["noise"]])
    label = _label_spec(f["label"], T)
    if family != "statespace":
        params = _family_params(family, f, k, T)
    else:
        state_family = "linear" if f["confounding"] == "latent" else f["state_family"]
        sf = {**DEFAULTS, **f}
        obs: dict[str, Any] = {}
        if f["observation"] == "noisy":
            obs["noise_scale"] = 0.1
        elif f["observation"] == "nonlinear":
            obs.update(transform="saturating", saturation=1.0, noise_scale=0.05)
        elif f["observation"] == "partial":
            obs.update(hidden=[k - 2], noise_scale=0.05)  # hide a non-label, non-route channel
        if f["missingness"] == "mcar":
            obs.update(missing="mcar", missing_rate=0.1)
        elif f["missingness"] == "block":
            obs.update(missing="block", missing_rate=0.3, block_length=max(T // 10, 2))
        if f["sampling"] == "irregular":
            obs["sampling_rate"] = 0.8
        params = {
            "base_family": state_family,
            "base_params": _family_params(state_family, sf, k, T),
            "observation": obs,
            "latent_confounder": f["confounding"] == "latent",
        }
    return WorldConfig(
        family=family,  # type: ignore[arg-type]
        world_id=world_id,
        seed=seed,
        n_samples=n_samples,
        train_frac=train_frac,
        params=params,
        noise=noise,
        label=label,
    )
