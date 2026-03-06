"""Tests for src/monitoring/drift_detector.py and src/monitoring/alerter.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift_detector import DriftDetector, compute_psi
from src.monitoring.alerter import Alerter, _severity_prefix


class TestComputePsi:
    def test_identical_distributions_return_near_zero(self):
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distributions_return_high_psi(self):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # Shifted mean
        psi = compute_psi(ref, cur)
        assert psi > 0.25  # Significant shift

    def test_handles_empty_arrays(self):
        psi = compute_psi(np.array([]), np.array([1.0, 2.0]))
        assert psi == 0.0

    def test_handles_nan_values(self):
        ref = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        cur = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        psi = compute_psi(ref, cur)
        assert psi >= 0.0  # Should not crash

    def test_returns_float(self):
        ref = np.random.uniform(0, 1, 500)
        cur = np.random.uniform(0, 1, 500)
        psi = compute_psi(ref, cur)
        assert isinstance(psi, float)


class TestDriftDetectorClassify:
    def test_classify_none(self):
        from src.core.project_config import MonitoringConfig, ProjectConfig, ModelConfig, SchemaConfig, OutputConfig
        config = ProjectConfig(
            project_id="test",
            model=ModelConfig(mlflow_name="m"),
            schema=SchemaConfig(feature_columns=["f1"]),
            output=OutputConfig(target_table="r.t"),
            monitoring=MonitoringConfig(
                drift={"enabled": True, "psi_warn": 0.10, "psi_critical": 0.25},
            ),
        )
        detector = DriftDetector(config)
        assert detector._classify(0.05) == "none"
        assert detector._classify(0.10) == "warn"
        assert detector._classify(0.20) == "warn"
        assert detector._classify(0.25) == "critical"
        assert detector._classify(0.50) == "critical"


class TestAlerter:
    def test_send_returns_true_without_webhook(self, monkeypatch):
        monkeypatch.delenv("TEAMS_WEBHOOK_URL", raising=False)
        alerter = Alerter()
        assert alerter.send("#test", "hello") is True

    def test_severity_prefix(self):
        assert _severity_prefix("critical") == "[!!!]"
        assert _severity_prefix("warning") == "[!]"
        assert _severity_prefix("info") == "[i]"
        assert _severity_prefix("unknown") == "[?]"
