"""Project configuration loader and validator.

Reads YAML files from project_registry/projects/ and converts them into
strongly-typed dataclass objects that the rest of the platform uses.

This is the foundation — every other module depends on ProjectConfig.

Phase 1: Load YAML, resolve env vars, basic validation.
Phase 2: Add schema validation with pydantic or custom checks.
"""

from __future__ import annotations

# TODO: Implement ProjectConfig dataclass with nested configs:
#   - ModelConfig (mlflow_name, version)
#   - InputConfig (source, connection, query/bucket/url)
#   - SchemaConfig (id_columns, feature_columns, output_columns)
#   - OutputConfig (target_table, write_mode, dedup_key)
#   - ScheduleConfig (cron, timezone, timeout_minutes, retries)
#   - EvaluationConfig (metrics, thresholds, ground_truth)
#   - MonitoringConfig (row_count thresholds, prediction_range, drift, alert_channel)

# TODO: Implement from_yaml(path) classmethod
#   - Load YAML with yaml.safe_load
#   - Resolve ${ENV_VAR} references from os.environ
#   - Validate required fields are present
#   - Return ProjectConfig instance

# TODO: Implement load_all_active() classmethod
#   - Scan project_registry/projects/*.yaml
#   - Load each, filter by status == "active"
#   - Return list of ProjectConfig
