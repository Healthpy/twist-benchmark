"""Adapters for third-party baselines from the ``cfts`` library (Schlegel & Seidl).

Upstream: https://github.com/visual-xai-for-time-series/counterfactual-explanations-for-time-series
(MIT), pinned to commit ``CFTS_COMMIT`` through the optional ``baselines`` extra. The draft's
reported baselines used this library too.

Every adapter:

* receives the same classifier (through :class:`~twist.explainers.budget.BudgetedModel`), the same
  raw-unit inputs and the same reference data (training trajectories and labels, regime R1),
  subsampled to ``budget.max_reference`` with the explainer seed;
* is seeded per unit (``derive_seed(explainer_seed, factual_id)``) for NumPy, Python and torch;
* converts ``(T, k)`` ↔ cfts' channel-first ``(C, L)`` float32;
* returns a trajectory-only :class:`Explanation` (no action — the benchmark applies registered
  readings), or a structured failure record: ``budget_exceeded``, ``timeout``, ``method_error``
  (exception message kept), ``invalid_output``, ``numerical_instability`` or ``no_target_flip``.
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import sys
import time
import traceback
import types
from typing import Any

import numpy as np
import torch

from twist.evaluation.result_schema import Explanation, ExplanationStatus
from twist.explainers.base import Explainer, ExplainerContext
from twist.explainers.budget import Budget, BudgetedModel, BudgetExceeded
from twist.utils.seeds import derive_seed

CFTS_COMMIT = "86c9e6e2ba3e894a3d5288f7a36f5908cea28232"
CFTS_SOURCE = f"cfts@{CFTS_COMMIT[:7]}"


def import_cfts(module: str) -> Any:
    """Import one cfts submodule without executing ``cfts/__init__.py``.

    The upstream package ``__init__`` eagerly imports all 35 methods, including ``pyts``, whose
    numba functions JIT-compile at import without caching (~55 s per process here). None of the
    adapted modules need that, so a path-only stub of the ``cfts`` package is registered and only
    the requested submodule (and its own imports) is loaded. Upstream code is unchanged. If the
    real package is already imported, it is used as is.
    """
    if "cfts" not in sys.modules:
        spec = importlib.util.find_spec("cfts")
        if spec is None or not spec.submodule_search_locations:
            raise NotImplementedError(
                "third-party baselines need the optional extra: uv pip install -e '.[baselines]'"
            )
        stub = types.ModuleType("cfts")
        stub.__path__ = list(spec.submodule_search_locations)
        stub.__package__ = "cfts"
        sys.modules["cfts"] = stub
    return importlib.import_module(module)


def cfts_available() -> bool:
    return "cfts" in sys.modules or importlib.util.find_spec("cfts") is not None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 — cfts uses the legacy global RNG
    torch.manual_seed(seed)


class CftsExplainer(Explainer):
    """Base adapter; subclasses implement :meth:`_call` for one cfts function."""

    native_actions = False
    min_regime = "R1"
    source = CFTS_SOURCE
    uses_reference_data = True
    stochastic = True

    def _call(
        self,
        sample_cl: np.ndarray,
        model: BudgetedModel,
        target: int,
        ref: dict[str, Any],
        seed: int,
    ) -> np.ndarray | None:
        """Run the method on one ``(C, L)`` float32 sample; return a ``(C, L)`` CF or ``None``."""
        raise NotImplementedError

    def _prepare(self, ref: dict[str, Any], model: BudgetedModel) -> None:
        """Optional per-call setup shared by all units (e.g. training an autoencoder)."""

    def explain(self, factuals, factual_ids, targets, ctx: ExplainerContext) -> list[Explanation]:
        # import the upstream module before any unit's clock starts
        import_cfts(self.upstream.split(":")[0])
        if ctx.train_x is None or ctx.train_y is None:
            raise PermissionError(f"{self.name} needs reference trajectories and labels (R1+)")
        budget = Budget.from_dict(ctx.budget)
        model = BudgetedModel(ctx.classifier, budget)
        rng = np.random.default_rng(derive_seed(self.seed, "reference"))
        n_ref = min(budget.max_reference, len(ctx.train_x))
        idx = np.sort(rng.choice(len(ctx.train_x), n_ref, replace=False))
        x_ref = np.ascontiguousarray(ctx.train_x[idx].transpose(0, 2, 1), dtype=np.float32)
        y_ref = np.asarray(ctx.train_y[idx], dtype=np.int64)
        ref = {"x": x_ref, "y": y_ref, "pairs": [(x_ref[i], int(y_ref[i])) for i in range(n_ref)]}
        _seed_all(derive_seed(self.seed, "prepare"))
        model.start_unit()
        model.budget = Budget(None, None, budget.max_reference)  # setup is not a unit's budget
        self._prepare(ref, model)
        model.budget = budget
        T, k = factuals.shape[1:]
        out = []
        for x, fid, y in zip(factuals, factual_ids, targets, strict=True):
            seed = derive_seed(self.seed, fid)
            _seed_all(seed)
            model.start_unit()
            t0 = time.perf_counter()
            proposal, status, error = None, ExplanationStatus.OK, None
            try:
                cf = self._call(
                    np.ascontiguousarray(x.T, dtype=np.float32), model, int(y), ref, seed
                )
                if cf is None:
                    status = ExplanationStatus.NO_TARGET_FLIP
                else:
                    cf = np.asarray(cf, dtype=np.float64)
                    if cf.shape != (k, T):
                        status, error = ExplanationStatus.INVALID_OUTPUT, f"shape {cf.shape}"
                    elif not np.all(np.isfinite(cf)):
                        status = ExplanationStatus.NUMERICAL_INSTABILITY
                    else:
                        proposal = cf.T.copy()
                        if int(ctx.classifier.predict(proposal)) != int(y):
                            status = ExplanationStatus.NO_TARGET_FLIP
            except BudgetExceeded as exc:
                status = (
                    ExplanationStatus.TIMEOUT
                    if exc.kind == "time"
                    else ExplanationStatus.BUDGET_EXCEEDED
                )
                error = str(exc)
            except Exception as exc:  # third-party failure → structured record, never a crash
                status = ExplanationStatus.METHOD_ERROR
                error = f"{type(exc).__name__}: {exc}"[:300]
                tb = traceback.format_exc(limit=3)
                error += " | " + tb.strip().splitlines()[-2][:200] if tb else ""
            out.append(
                Explanation(
                    explainer=self.name,
                    factual_id=fid,
                    target=int(y),
                    proposed_trajectory=proposal,
                    mechanism_access_regime=ctx.regime,
                    status=status,
                    selection_objective={
                        "upstream": self.upstream,
                        "params": self._public_params(),
                    },
                    diagnostics={
                        "queries": model.queries,
                        "error": error,
                        "unit_seed": seed,
                        "n_reference": n_ref,
                        "source": self.source,
                    },
                    explainer_seed=self.seed,
                    runtime_seconds=time.perf_counter() - t0,
                )
            )
        return out

    def _public_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.params.items() if not k.startswith("_")}


class CftsWachter(CftsExplainer):
    """Wachter et al. (2018), gradient variant (``wachter_gradient_cf``)."""

    name = "Wachter"
    upstream = "cfts.cf_wachter.wachter:wachter_gradient_cf"
    uses_gradients = True

    def _call(self, sample_cl, model, target, ref, seed):
        wachter_gradient_cf = import_cfts("cfts.cf_wachter.wachter").wachter_gradient_cf

        cf, _ = wachter_gradient_cf(
            sample_cl,
            model,
            target_class=target,
            dataset=ref["pairs"],
            max_cfs=int(self.params.get("max_cfs", 1000)),
            distance=self.params.get("distance", "euclidean"),
        )
        return cf


class CftsCOMTE(CftsExplainer):
    """CoMTE (Ates et al., 2021): greedy channel swaps from target-class distractors."""

    name = "COMTE"
    upstream = "cfts.cf_comte.comte:comte_cf"
    stochastic = False

    def _call(self, sample_cl, model, target, ref, seed):
        comte_cf = import_cfts("cfts.cf_comte.comte").comte_cf

        cf, _ = comte_cf(
            sample_cl,
            model,
            target_class=target,
            dataset=ref["pairs"],
            num_distractors=int(self.params.get("num_distractors", 2)),
            n_segments=int(self.params.get("n_segments", 10)),
        )
        return cf


class CftsMCELS(CftsExplainer):
    """M-CELS (Li et al., 2024): learned saliency mask blending toward the nearest unlike
    neighbour (faithful reimplementation ``mcels_reimp``)."""

    name = "M-CELS"
    upstream = "cfts.cf_cels.mcels_reimp:mcels_generate"
    uses_gradients = True

    def _call(self, sample_cl, model, target, ref, seed):
        mcels_generate = import_cfts("cfts.cf_cels.mcels_reimp").mcels_generate

        _, cf, _, cf_label = mcels_generate(
            sample_cl,
            model,
            ref["x"],
            ref["y"],
            max_itr=int(self.params.get("max_itr", 1000)),
            lr=float(self.params.get("lr", 1e-3)),
            random_state=seed % 2**32,
        )
        if int(cf_label) != target:  # binary: the second-most-probable class must be the target
            raise ValueError(f"M-CELS chose class {cf_label}, target is {target}")
        return cf


class CftsCONFETTI(CftsExplainer):
    """CONFETTI (Paredes Cetina et al., 2025): NSGA-II multi-objective search
    (``confetti_nsga_cf``)."""

    name = "CONFETTI"
    upstream = "cfts.cf_confetti.confetti:confetti_nsga_cf"

    def _call(self, sample_cl, model, target, ref, seed):
        confetti_nsga_cf = import_cfts("cfts.cf_confetti.confetti").confetti_nsga_cf

        cf, _ = confetti_nsga_cf(
            sample_cl,
            model,
            target_class=target,
            dataset=ref["pairs"],
            population_size=int(self.params.get("population_size", 60)),
            max_generations=int(self.params.get("max_generations", 40)),
            seed=seed,
        )
        return cf


class CftsNativeGuide(CftsExplainer):
    """Native Guide (Delaney et al., 2021): transplant attribution-important windows of the
    nearest unlike neighbour (``native_guide_uni_cf``; GradientShap attributions)."""

    name = "NativeGuide"
    upstream = "cfts.cf_native_guide.native_guide:native_guide_uni_cf"
    uses_gradients = True

    def _call(self, sample_cl, model, target, ref, seed):
        native_guide_uni_cf = import_cfts("cfts.cf_native_guide.native_guide").native_guide_uni_cf

        cf, _ = native_guide_uni_cf(
            sample_cl,
            model,
            target_class=target,
            dataset=ref["pairs"],
            sub_len=int(self.params.get("sub_len", 1)),
        )
        return cf


class CftsLatentCF(CftsExplainer):
    """Latent-CF (Balasubramanian et al., 2020): gradient search in an autoencoder latent space.

    Generative/plausibility baseline. The autoencoder is trained once per run on the reference data
    (``train_autoencoder``) and shared by all units.
    """

    name = "LatentCF"
    upstream = "cfts.cf_latent_cf.latent_cf:latent_cf_generate"
    uses_gradients = True

    def _prepare(self, ref, model):
        train_autoencoder = import_cfts("cfts.cf_latent_cf.latent_cf").train_autoencoder

        self._ae = train_autoencoder(
            ref["pairs"],
            latent_dim=int(self.params.get("latent_dim", 8)),
            epochs=int(self.params.get("ae_epochs", 100)),
        )

    def _call(self, sample_cl, model, target, ref, seed):
        latent_cf_generate = import_cfts("cfts.cf_latent_cf.latent_cf").latent_cf_generate

        cf, _ = latent_cf_generate(
            sample_cl,
            model,
            target_class=target,
            autoencoder=self._ae,
            max_iter=int(self.params.get("max_iter", 100)),
            lambda_dist=float(self.params.get("lambda_dist", 0.1)),
        )
        return None if cf is None else np.asarray(cf).reshape(sample_cl.shape)


class NearestUnlikeNeighbor(Explainer):
    """Retrieval baseline (in-house): the reference trajectory closest to the factual (per-channel
    standardized L2) among those the classifier assigns to the target class. Proposals are real
    trajectories from the world, hence plausible by construction."""

    name = "NUN"
    role = "baseline"
    native_actions = False
    min_regime = "R1"
    uses_reference_data = True

    def explain(self, factuals, factual_ids, targets, ctx):
        if ctx.train_x is None:
            raise PermissionError(f"{self.name} needs reference trajectories (R1+)")
        budget = Budget.from_dict(ctx.budget)
        rng = np.random.default_rng(derive_seed(self.seed, "reference"))
        n_ref = min(budget.max_reference, len(ctx.train_x))
        ref = ctx.train_x[np.sort(rng.choice(len(ctx.train_x), n_ref, replace=False))]
        pred = ctx.classifier.predict(ref)
        sd = ref.reshape(-1, ref.shape[-1]).std(0) + 1e-8
        out = []
        for x, fid, y in zip(factuals, factual_ids, targets, strict=True):
            t0 = time.perf_counter()
            pool = ref[pred == y]
            if len(pool) == 0:
                out.append(
                    Explanation(
                        self.name,
                        fid,
                        int(y),
                        None,
                        status=ExplanationStatus.NO_TARGET_FLIP,
                        mechanism_access_regime=ctx.regime,
                        explainer_seed=self.seed,
                    )
                )
                continue
            d = np.sqrt((((pool - x[None]) / sd) ** 2).sum(axis=(1, 2)))
            out.append(
                Explanation(
                    explainer=self.name,
                    factual_id=fid,
                    target=int(y),
                    proposed_trajectory=pool[int(np.argmin(d))].copy(),
                    mechanism_access_regime=ctx.regime,
                    selection_objective={
                        "rule": "nearest reference trajectory predicted as target",
                        "distance": "standardized L2",
                    },
                    candidate_count=len(pool),
                    diagnostics={"queries": n_ref, "n_reference": n_ref},
                    explainer_seed=self.seed,
                    runtime_seconds=time.perf_counter() - t0,
                )
            )
        return out
