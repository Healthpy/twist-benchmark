"""Explainer registry for config-driven construction."""

from __future__ import annotations

from typing import Any

from twist.explainers.action_native.recourse import (
    EstimatedActionRecourse,
    GraphActionRecourse,
    OracleActionRecourse,
)
from twist.explainers.adapters.cfts_baselines import (
    CftsCOMTE,
    CftsCONFETTI,
    CftsLatentCF,
    CftsMCELS,
    CftsNativeGuide,
    CftsWachter,
    NearestUnlikeNeighbor,
)
from twist.explainers.adapters.wachter import WachterTrajectoryCF
from twist.explainers.base import Explainer
from twist.explainers.structural_controls.controls import (
    AdversarialTrajectoryEdit,
    FullScheduleOracle,
    NoActionAbstain,
)
from twist.explainers.structural_controls.scm_recourse import (
    NoiselessSCMRecourse,
    PearlSCMRecourse,
    PlausibleButMechanismWrongEdit,
)

EXPLAINERS: dict[str, type[Explainer]] = {
    cls.name: cls
    for cls in (
        CftsWachter,
        CftsCOMTE,
        CftsMCELS,
        CftsCONFETTI,
        CftsNativeGuide,
        CftsLatentCF,
        NearestUnlikeNeighbor,
        WachterTrajectoryCF,
        GraphActionRecourse,
        EstimatedActionRecourse,
        OracleActionRecourse,
        PearlSCMRecourse,
        NoiselessSCMRecourse,
        PlausibleButMechanismWrongEdit,
        FullScheduleOracle,
        AdversarialTrajectoryEdit,
        NoActionAbstain,
    )
}


def build_explainer(name: str, seed: int = 0, **params: Any) -> Explainer:
    if name not in EXPLAINERS:
        raise KeyError(f"unknown explainer {name!r}; available: {sorted(EXPLAINERS)}")
    return EXPLAINERS[name](seed=seed, **params)
