"""Explainers: baseline adapters, action-native methods, structural controls, and
mechanism-access gating. Every explainer returns :class:`twist.evaluation.result_schema.Explanation`."""

from twist.explainers.base import Explainer, ExplainerContext
from twist.explainers.registry import EXPLAINERS, build_explainer

__all__ = ["EXPLAINERS", "Explainer", "ExplainerContext", "build_explainer"]
