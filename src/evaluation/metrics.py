"""Config-driven evaluation metrics.

Computes metrics declared in project YAML. One module, no separate
files per metric. One registry dict maps metric names to functions.

Used in two contexts:
  1. Candidate validation — compare candidate vs production model
  2. Monitoring — track prediction quality over time (when ground truth available)

Usage:
    evaluator = Evaluator(config.evaluation)
    results = evaluator.compute(y_true, y_pred, y_proba)
    # → {"auc_roc": 0.94, "f1_macro": 0.82, "precision_macro": 0.88, ...}

YAML config example:
    evaluation:
      metrics:
        - auc_roc
        - f1_macro
        - precision_macro
        - recall_macro
        - log_loss
      thresholds:
        auc_roc: 0.80
        f1_macro: 0.60
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("mlops.evaluation")

# ═══════════════════════════════════════════════════════════
# METRIC REGISTRY — all supported metrics in one place
# ═══════════════════════════════════════════════════════════
# To add a new metric:
#   1. Write a function: def _my_metric(y_true, y_pred, y_proba) -> float
#   2. Add it to METRIC_REGISTRY: "my_metric": _my_metric
#   3. Teams can now use it in their YAML: metrics: [my_metric]


def _auc_roc(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import roc_auc_score
    if y_proba is None:
        raise ValueError("auc_roc requires y_proba (probability scores)")
    if y_proba.ndim == 2:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return roc_auc_score(y_true, y_proba)


def _f1_macro(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def _f1_weighted(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _precision_macro(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def _recall_macro(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def _accuracy(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def _log_loss(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import log_loss
    if y_proba is None:
        raise ValueError("log_loss requires y_proba")
    return log_loss(y_true, y_proba)


def _mse(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)


def _rmse(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred, squared=False)


def _r2(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)


def _mae(y_true, y_pred, y_proba) -> float:
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def _ks_statistic(y_true, y_pred, y_proba) -> float:
    """Kolmogorov-Smirnov statistic — common in credit risk."""
    if y_proba is None:
        raise ValueError("ks_statistic requires y_proba")
    scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    from scipy.stats import ks_2samp
    stat, _ = ks_2samp(pos, neg)
    return stat


def _gini(y_true, y_pred, y_proba) -> float:
    """Gini coefficient — 2 * AUC - 1. Common in credit risk."""
    auc = _auc_roc(y_true, y_pred, y_proba)
    return 2 * auc - 1


METRIC_REGISTRY: dict[str, callable] = {
    # Classification
    "auc_roc": _auc_roc,
    "f1_macro": _f1_macro,
    "f1_weighted": _f1_weighted,
    "precision_macro": _precision_macro,
    "recall_macro": _recall_macro,
    "accuracy": _accuracy,
    "log_loss": _log_loss,
    "ks_statistic": _ks_statistic,
    "gini": _gini,
    # Regression
    "mse": _mse,
    "rmse": _rmse,
    "r2": _r2,
    "mae": _mae,
}


# ═══════════════════════════════════════════════════════════
# EVALUATOR — config-driven metric computation
# ═══════════════════════════════════════════════════════════


@dataclass
class EvaluationConfig:
    """Parsed from the evaluation section of project YAML."""
    metrics: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.metrics) - set(METRIC_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. "
                f"Available: {sorted(METRIC_REGISTRY.keys())}"
            )


class Evaluator:
    """Computes metrics declared in project config.

    Args:
        config: EvaluationConfig from project YAML.

    Usage:
        evaluator = Evaluator(config.evaluation)
        results = evaluator.compute(y_true, y_pred, y_proba)
        passed = evaluator.check_thresholds(results)
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all declared metrics.

        Args:
            y_true: Ground truth labels or values.
            y_pred: Predicted labels or values.
            y_proba: Predicted probabilities (required for AUC, log_loss, etc.).

        Returns:
            Dict mapping metric name to computed value.
        """
        results = {}
        for metric_name in self.config.metrics:
            fn = METRIC_REGISTRY[metric_name]
            try:
                results[metric_name] = round(fn(y_true, y_pred, y_proba), 6)
            except Exception as e:
                logger.warning(f"Metric '{metric_name}' failed: {e}")
                results[metric_name] = None
        return results

    def check_thresholds(self, results: dict[str, float]) -> dict[str, dict[str, Any]]:
        """Check computed metrics against declared thresholds.

        Returns:
            Dict with pass/fail status for each threshold.
            Example: {"auc_roc": {"value": 0.94, "threshold": 0.80, "passed": True}}
        """
        checks = {}
        for metric_name, threshold in self.config.thresholds.items():
            value = results.get(metric_name)
            if value is None:
                checks[metric_name] = {
                    "value": None,
                    "threshold": threshold,
                    "passed": False,
                    "reason": "metric computation failed",
                }
            else:
                # For loss metrics (lower is better), flip the comparison
                lower_is_better = metric_name in {"log_loss", "mse", "rmse", "mae"}
                passed = value <= threshold if lower_is_better else value >= threshold
                checks[metric_name] = {
                    "value": value,
                    "threshold": threshold,
                    "passed": passed,
                }
        return checks

    def summary(self, results: dict[str, float]) -> str:
        """Human-readable summary of evaluation results."""
        lines = []
        threshold_checks = self.check_thresholds(results)
        for metric_name in self.config.metrics:
            value = results.get(metric_name)
            val_str = f"{value:.4f}" if value is not None else "FAILED"
            if metric_name in threshold_checks:
                check = threshold_checks[metric_name]
                status = "✅" if check["passed"] else "❌"
                lines.append(f"  {status} {metric_name}: {val_str} (threshold: {check['threshold']})")
            else:
                lines.append(f"  ℹ️  {metric_name}: {val_str}")
        return "\n".join(lines)
