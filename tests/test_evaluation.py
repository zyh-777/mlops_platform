"""Tests for src/evaluation/metrics.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    METRIC_REGISTRY,
    EvaluationConfig,
    Evaluator,
)


class TestEvaluationConfig:
    def test_accepts_valid_metrics(self):
        config = EvaluationConfig(metrics=["auc_roc", "f1_macro"])
        assert len(config.metrics) == 2

    def test_raises_on_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metrics"):
            EvaluationConfig(metrics=["auc_roc", "nonexistent_metric"])

    def test_empty_metrics_is_valid(self):
        config = EvaluationConfig(metrics=[])
        assert config.metrics == []


class TestEvaluator:
    @pytest.fixture
    def binary_data(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.6, 0.85, 0.3, 0.15, 0.05])
        return y_true, y_pred, y_proba

    def test_compute_returns_all_metrics(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        config = EvaluationConfig(
            metrics=["auc_roc", "f1_macro", "accuracy"],
        )
        evaluator = Evaluator(config)
        results = evaluator.compute(y_true, y_pred, y_proba)

        assert "auc_roc" in results
        assert "f1_macro" in results
        assert "accuracy" in results
        assert all(isinstance(v, float) for v in results.values())

    def test_auc_roc_range(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        config = EvaluationConfig(metrics=["auc_roc"])
        evaluator = Evaluator(config)
        results = evaluator.compute(y_true, y_pred, y_proba)

        assert 0.0 <= results["auc_roc"] <= 1.0

    def test_check_thresholds_pass(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        config = EvaluationConfig(
            metrics=["accuracy"],
            thresholds={"accuracy": 0.5},
        )
        evaluator = Evaluator(config)
        results = evaluator.compute(y_true, y_pred, y_proba)
        checks = evaluator.check_thresholds(results)

        assert checks["accuracy"]["passed"] is True

    def test_check_thresholds_fail(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        config = EvaluationConfig(
            metrics=["accuracy"],
            thresholds={"accuracy": 0.99},
        )
        evaluator = Evaluator(config)
        results = evaluator.compute(y_true, y_pred, y_proba)
        checks = evaluator.check_thresholds(results)

        assert checks["accuracy"]["passed"] is False

    def test_loss_metric_lower_is_better(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        config = EvaluationConfig(
            metrics=["log_loss"],
            thresholds={"log_loss": 5.0},  # Very generous threshold
        )
        evaluator = Evaluator(config)
        results = evaluator.compute(y_true, y_pred, y_proba)
        checks = evaluator.check_thresholds(results)

        # log_loss should be well under 5.0 → should pass
        assert checks["log_loss"]["passed"] is True

    def test_summary_produces_string(self, binary_data):
        y_true, y_pred, y_proba = binary_data
        config = EvaluationConfig(
            metrics=["auc_roc", "f1_macro"],
            thresholds={"auc_roc": 0.5},
        )
        evaluator = Evaluator(config)
        results = evaluator.compute(y_true, y_pred, y_proba)
        summary = evaluator.summary(results)

        assert isinstance(summary, str)
        assert "auc_roc" in summary


class TestMetricRegistry:
    def test_all_registered_metrics_are_callable(self):
        for name, fn in METRIC_REGISTRY.items():
            assert callable(fn), f"Metric '{name}' is not callable"

    def test_registry_has_expected_metrics(self):
        expected = [
            "auc_roc", "f1_macro", "f1_weighted", "precision_macro",
            "recall_macro", "accuracy", "log_loss", "ks_statistic", "gini",
            "mse", "rmse", "r2", "mae",
        ]
        for name in expected:
            assert name in METRIC_REGISTRY, f"Missing metric: {name}"
