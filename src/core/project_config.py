"""Project configuration loader and validator.

Reads YAML files from project_registry/projects/ and converts them into
strongly-typed dataclass objects that the rest of the platform uses.

This is the foundation — every other module depends on ProjectConfig.

Universal pattern: Configuration as Code
    - All project-specific values live in YAML, not Python.
    - This module parses YAML → nested dataclasses with type safety.
    - Adding a new project = adding a YAML file. Zero code changes.
    - ${ENV_VAR} references in YAML get resolved from os.environ.

Usage:
    config = ProjectConfig.from_yaml("project_registry/projects/fraud_detection.yaml")
    print(config.model.mlflow_name)     # "fraud_detection_alpha"
    print(config.input.source)          # "mysql"
    print(config.schema.feature_columns)  # ["amount", "merchant_risk", ...]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.metrics import EvaluationConfig
from src.logger.logger import get_logger

logger = get_logger("project_config")

# Root of the project — used to locate project_registry/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════
# NESTED CONFIG DATACLASSES
# ═══════════════════════════════════════════════════════════
# Each section of the YAML becomes its own dataclass.
# Why? So you get autocomplete, type checking, and clear structure.
# config.model.mlflow_name is much clearer than config["model"]["mlflow_name"].


@dataclass
class ModelConfig:
    """Which model to load from MLflow."""
    mlflow_name: str                       # registered model name in MLflow
    version: str = "Production"            # alias ("Production") or version number ("3")


@dataclass
class InputConfig:
    """Where to read input data from."""
    source: str = "mysql"                  # "mysql", "s3", "api"
    connection: dict[str, Any] = field(default_factory=dict)  # host, port, user, etc.
    query: str = ""                        # SQL query with {run_date} placeholder
    bucket: str = ""                       # S3 bucket (for source=s3)
    key_template: str = ""                 # S3 key template (for source=s3)


@dataclass
class SchemaConfig:
    """Declares expected columns — used for validation."""
    id_columns: list[str] = field(default_factory=list)       # e.g., [txn_id, txn_date]
    feature_columns: list[str] = field(default_factory=list)  # fed to model.predict()
    output_columns: list[str] = field(default_factory=list)   # what predict() returns


@dataclass
class OutputConfig:
    """Where to write prediction results."""
    target_table: str = ""                 # e.g., "results.fraud_scores"
    write_mode: str = "append"             # "append" or "replace_date"
    dedup_key: list[str] = field(default_factory=list)  # prevents duplicate rows


@dataclass
class ScheduleConfig:
    """When and how to run the pipeline."""
    cron: str = "0 10 * * *"               # default: 10am daily
    timezone: str = "Asia/Singapore"
    timeout_minutes: int = 120
    retries: int = 2


@dataclass
class MonitoringConfig:
    """Health checks after each inference run."""
    row_count_change_max_pct: float = 20.0
    null_output_max_pct: float = 0.0
    prediction_range: dict[str, list[float]] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    alert_channel: str = ""


# ═══════════════════════════════════════════════════════════
# MAIN CONFIG CLASS
# ═══════════════════════════════════════════════════════════


@dataclass
class ProjectConfig:
    """Complete project configuration parsed from YAML.

    This is the single object passed to InferencePipeline, validators,
    data connectors, and result store. Everything reads from this.
    """
    project_id: str
    display_name: str = ""
    owner: str = ""
    contact: str = ""
    status: str = "active"

    model: ModelConfig = field(default_factory=lambda: ModelConfig(mlflow_name=""))
    input: InputConfig = field(default_factory=InputConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        """Loads a project config from a YAML file.

        Steps:
            1. Read YAML file
            2. Resolve ${ENV_VAR} references (e.g., ${MYSQL_HOST} → "127.0.0.1")
            3. Convert nested dicts into typed dataclasses
            4. Validate required fields

        Args:
            path: Path to the YAML file.

        Returns:
            A fully-parsed ProjectConfig instance.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            ValueError: If required fields are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        # Resolve ${ENV_VAR} references throughout the entire dict
        raw = _resolve_env_vars(raw)

        # Build nested dataclasses from the raw dict
        config = cls(
            project_id=raw["project_id"],
            display_name=raw.get("display_name", ""),
            owner=raw.get("owner", ""),
            contact=raw.get("contact", ""),
            status=raw.get("status", "active"),
            model=ModelConfig(**raw.get("model", {})),
            input=InputConfig(**raw.get("input", {})),
            schema=SchemaConfig(**raw.get("schema", {})),
            output=OutputConfig(**_normalize_output(raw.get("output", {}))),
            schedule=ScheduleConfig(**raw.get("schedule", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            monitoring=MonitoringConfig(**raw.get("monitoring", {})),
        )

        _validate(config, path)
        logger.info(f"[{config.project_id}] Loaded config from {path.name}")
        return config

    @classmethod
    def load_all_active(cls) -> list[ProjectConfig]:
        """Scans project_registry/projects/ and loads all active configs.

        Skips files starting with '_' (examples/templates).

        Returns:
            List of ProjectConfig where status == "active".
        """
        registry_dir = PROJECT_ROOT / "project_registry" / "projects"
        configs = []
        for yaml_path in sorted(registry_dir.glob("*.yaml")):
            if yaml_path.name.startswith("_"):
                continue
            try:
                config = cls.from_yaml(yaml_path)
                if config.status == "active":
                    configs.append(config)
            except Exception as e:
                logger.error(f"Failed to load {yaml_path.name}: {e}")
        logger.info(f"Loaded {len(configs)} active project(s)")
        return configs


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively replaces ${VAR_NAME} with os.environ values.

    This lets YAML files reference secrets without hardcoding them:
        host: ${MYSQL_HOST}  →  host: "127.0.0.1"

    Unresolved variables (not in env) are left as-is with a warning.
    """
    if isinstance(obj, str):
        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                logger.warning(f"Environment variable ${{{var_name}}} is not set")
                return match.group(0)  # leave as-is
            return value
        return re.sub(r"\$\{(\w+)}", _replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def _normalize_output(raw: dict) -> dict:
    """Ensures dedup_key is always a list (YAML allows inline list syntax)."""
    if "dedup_key" in raw and isinstance(raw["dedup_key"], str):
        raw["dedup_key"] = [raw["dedup_key"]]
    return raw


def _validate(config: ProjectConfig, path: Path) -> None:
    """Checks that essential fields are present.

    This is a lightweight check — not full schema validation.
    Catches the most common mistakes early.
    """
    errors = []
    if not config.project_id:
        errors.append("project_id is required")
    if not config.model.mlflow_name:
        errors.append("model.mlflow_name is required")
    if not config.output.target_table:
        errors.append("output.target_table is required")
    if not config.schema.feature_columns:
        errors.append("schema.feature_columns is required (at least one)")

    if errors:
        raise ValueError(f"Invalid config in {path.name}: {'; '.join(errors)}")
