"""Tests for src/inference/pipeline.py and src/core/project_config.py"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.project_config import (
    InputConfig,
    ModelConfig,
    MonitoringConfig,
    OutputConfig,
    ProjectConfig,
    SchemaConfig,
    ScheduleConfig,
)
from src.evaluation.metrics import EvaluationConfig
from src.inference.pipeline import InferencePipeline


# ═══════════════════════════════════════════════════════════
# ProjectConfig tests
# ═══════════════════════════════════════════════════════════

EXAMPLE_YAML = Path(__file__).resolve().parents[1] / "project_registry" / "projects" / "_example_fraud_detection.yaml"


class TestProjectConfigFromYaml:
    def test_loads_example_yaml(self):
        """Validates that the example YAML loads without errors."""
        config = ProjectConfig.from_yaml(EXAMPLE_YAML)

        assert config.project_id == "fraud_detection_alpha"
        assert config.status == "active"
        assert config.model.mlflow_name == "fraud_detection_alpha"
        assert config.model.version == "Production"

    def test_schema_columns(self):
        config = ProjectConfig.from_yaml(EXAMPLE_YAML)

        assert "txn_id" in config.schema.id_columns
        assert "amount" in config.schema.feature_columns
        assert "fraud_probability" in config.schema.output_columns

    def test_output_config(self):
        config = ProjectConfig.from_yaml(EXAMPLE_YAML)

        assert config.output.target_table == "results.fraud_scores"
        assert config.output.write_mode == "append"
        assert "txn_id" in config.output.dedup_key

    def test_evaluation_config(self):
        config = ProjectConfig.from_yaml(EXAMPLE_YAML)

        assert "auc_roc" in config.evaluation.metrics
        assert config.evaluation.thresholds.get("auc_roc") == 0.80

    def test_monitoring_config(self):
        config = ProjectConfig.from_yaml(EXAMPLE_YAML)

        assert config.monitoring.row_count_change_max_pct == 20
        assert "fraud_probability" in config.monitoring.prediction_range

    def test_schedule_config(self):
        config = ProjectConfig.from_yaml(EXAMPLE_YAML)

        assert config.schedule.cron == "0 8 * * *"
        assert config.schedule.timezone == "Asia/Singapore"
        assert config.schedule.retries == 2

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ProjectConfig.from_yaml("/nonexistent/path.yaml")

    def test_env_var_resolution(self, tmp_path, monkeypatch):
        """Tests that ${ENV_VAR} references are resolved."""
        monkeypatch.setenv("TEST_HOST", "db.example.com")
        monkeypatch.setenv("TEST_PORT", "3307")

        yaml_content = """
project_id: test_project
model:
  mlflow_name: test_model
  version: Production
input:
  source: mysql
  connection:
    host: ${TEST_HOST}
    port: ${TEST_PORT}
    database: testdb
  query: "SELECT 1"
schema:
  id_columns: [id]
  feature_columns: [f1]
  output_columns: [pred]
output:
  target_table: results.test
  write_mode: append
"""
        yaml_file = tmp_path / "test_project.yaml"
        yaml_file.write_text(yaml_content)

        config = ProjectConfig.from_yaml(yaml_file)
        assert config.input.connection["host"] == "db.example.com"
        assert config.input.connection["port"] == "3307"


# ═══════════════════════════════════════════════════════════
# InferencePipeline tests
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def mock_config() -> ProjectConfig:
    """Creates a ProjectConfig for testing without YAML."""
    return ProjectConfig(
        project_id="test_project",
        display_name="Test Project",
        model=ModelConfig(mlflow_name="test_model", version="Production"),
        input=InputConfig(
            source="mysql",
            connection={"host": "localhost", "database": "test"},
            query="SELECT * FROM t WHERE date = '{run_date}'",
        ),
        schema=SchemaConfig(
            id_columns=["id"],
            feature_columns=["f1", "f2", "f3"],
            output_columns=["prediction"],
        ),
        output=OutputConfig(
            target_table="results.test_output",
            write_mode="append",
            dedup_key=["id", "run_date"],
        ),
        schedule=ScheduleConfig(),
        evaluation=EvaluationConfig(),
        monitoring=MonitoringConfig(
            prediction_range={"prediction": [0.0, 1.0]},
        ),
    )


@pytest.fixture
def sample_input_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["A", "B", "C"],
        "f1": [1.0, 2.0, 3.0],
        "f2": [4.0, 5.0, 6.0],
        "f3": [7.0, 8.0, 9.0],
    })


class TestInferencePipeline:
    @patch("src.inference.pipeline.save")
    @patch("src.inference.pipeline.log_run")
    @patch("src.inference.pipeline.ensure_result_table")
    @patch("src.inference.pipeline.get_engine")
    @patch("src.inference.pipeline.load_model")
    @patch("src.inference.pipeline.DataConnectorFactory")
    def test_full_run(
        self,
        mock_factory,
        mock_load_model,
        mock_get_engine,
        mock_ensure_table,
        mock_log_run,
        mock_save,
        mock_config,
        sample_input_df,
    ):
        """Tests the full pipeline with mocked dependencies."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.8, 0.3])
        mock_load_model.return_value = mock_model

        # Mock data connector
        mock_connector = MagicMock()
        mock_connector.fetch.return_value = sample_input_df
        mock_factory.create.return_value = mock_connector

        # Mock save
        mock_save.return_value = 3

        # Run pipeline
        pipeline = InferencePipeline(mock_config)
        pipeline.run("2026-03-05")

        # Verify steps were called
        mock_load_model.assert_called_once_with("test_model", "Production")
        mock_connector.fetch.assert_called_once_with("2026-03-05")
        mock_model.predict.assert_called_once()
        mock_save.assert_called_once()
        mock_log_run.assert_called_once()

        # Verify log_run was called with success
        log_call = mock_log_run.call_args
        assert log_call[0][1] == "2026-03-05"  # run_date
        assert log_call[0][2] == "success"  # status

    @patch("src.inference.pipeline.log_run")
    @patch("src.inference.pipeline.load_model")
    def test_logs_failure_on_model_load_error(
        self, mock_load_model, mock_log_run, mock_config
    ):
        """Tests that model load failure is logged."""
        mock_load_model.side_effect = Exception("MLflow unreachable")

        pipeline = InferencePipeline(mock_config)
        with pytest.raises(Exception, match="MLflow unreachable"):
            pipeline.run("2026-03-05")

        # Verify failure was logged
        mock_log_run.assert_called_once()
        log_call = mock_log_run.call_args
        assert log_call[0][2] == "failed"
        assert "MLflow unreachable" in log_call[1]["error_message"]

    def test_predict_normalizes_1d_array(self, mock_config, sample_input_df):
        """Tests that 1D numpy array predictions are normalized to DataFrame."""
        pipeline = InferencePipeline(mock_config)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.5, 0.9])

        result = pipeline._predict(mock_model, sample_input_df)
        assert isinstance(result, pd.DataFrame)
        assert "prediction" in result.columns
        assert len(result) == 3

    def test_predict_normalizes_2d_array(self, mock_config, sample_input_df):
        """Tests that 2D numpy array predictions are normalized to DataFrame."""
        mock_config.schema.output_columns = ["pred_a", "pred_b"]
        pipeline = InferencePipeline(mock_config)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])

        result = pipeline._predict(mock_model, sample_input_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["pred_a", "pred_b"]

    def test_assemble_output_includes_run_date(self, mock_config, sample_input_df):
        """Tests that the assembled output has id_columns + predictions + run_date."""
        pipeline = InferencePipeline(mock_config)
        predictions = pd.DataFrame({"prediction": [0.1, 0.5, 0.9]})

        result = pipeline._assemble_output(sample_input_df, predictions, "2026-03-05")

        assert "id" in result.columns
        assert "prediction" in result.columns
        assert "run_date" in result.columns
        assert list(result["run_date"]) == ["2026-03-05"] * 3
