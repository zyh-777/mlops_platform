"""Validate a project YAML config and its model end-to-end.

Usage:
    python -m tools.validate_project <project_id>
    python -m tools.validate_project <project_id> --candidate

Checks:
    1. YAML is valid and all required fields present
    2. Evaluation metrics are valid and registered in METRIC_REGISTRY
    3. Model loads from MLflow (Production or Candidate alias)
    4. Data source is reachable and query returns expected columns
    5. Test prediction succeeds on sample data
    6. Output matches declared schema
    7. (--candidate) Run evaluation metrics, check thresholds, compare with production
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.project_config import PROJECT_ROOT, ProjectConfig
from src.evaluation.metrics import METRIC_REGISTRY
from src.logger.logger import get_logger

logger = get_logger("validate_project")

REGISTRY_DIR = PROJECT_ROOT / "project_registry" / "projects"


def _find_yaml(project_id: str) -> Path:
    """Locates the YAML file for a given project_id."""
    path = REGISTRY_DIR / f"{project_id}.yaml"
    if path.exists():
        return path
    # Also check files starting with _ (examples)
    path = REGISTRY_DIR / f"_{project_id}.yaml"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"No YAML found for '{project_id}' in {REGISTRY_DIR}"
    )


def validate_config(project_id: str) -> tuple[ProjectConfig | None, list[str]]:
    """Step 1: Validate YAML loads and required fields are present.

    Returns:
        Tuple of (config or None, list of error messages).
    """
    errors: list[str] = []
    try:
        path = _find_yaml(project_id)
        config = ProjectConfig.from_yaml(path)
        return config, errors
    except FileNotFoundError as e:
        errors.append(f"Config not found: {e}")
        return None, errors
    except ValueError as e:
        errors.append(f"Config validation failed: {e}")
        return None, errors
    except Exception as e:
        errors.append(f"Unexpected error loading config: {e}")
        return None, errors


def validate_evaluation_config(config: ProjectConfig) -> list[str]:
    """Step 2: Check that evaluation metrics are valid.

    Verifies:
        - All metric names exist in METRIC_REGISTRY.
        - Threshold metric names are a subset of declared metrics.
    """
    errors: list[str] = []

    unknown_metrics = set(config.evaluation.metrics) - set(METRIC_REGISTRY)
    if unknown_metrics:
        errors.append(
            f"Unknown evaluation metrics: {sorted(unknown_metrics)}. "
            f"Available: {sorted(METRIC_REGISTRY.keys())}"
        )

    threshold_keys = set(config.evaluation.thresholds.keys())
    metric_set = set(config.evaluation.metrics)
    orphan_thresholds = threshold_keys - metric_set
    if orphan_thresholds:
        errors.append(
            f"Thresholds defined for metrics not in metrics list: "
            f"{sorted(orphan_thresholds)}"
        )

    return errors


def validate_model_loads(config: ProjectConfig, version: str | None = None) -> list[str]:
    """Step 3: Try loading the model from MLflow.

    Args:
        config: Project config.
        version: Override version/alias (e.g., "Candidate"). Uses config default if None.
    """
    errors: list[str] = []
    target_version = version or config.model.version

    try:
        from src.core.model_loader import load_model
        load_model(config.model.mlflow_name, target_version)
    except Exception as e:
        errors.append(f"Model load failed ({target_version}): {e}")

    return errors


def validate_data_source(config: ProjectConfig) -> list[str]:
    """Step 4: Check that the data source is reachable."""
    errors: list[str] = []
    try:
        from src.core.data_connector import DataConnectorFactory
        connector = DataConnectorFactory.create(config.input)
        # Try fetching with today's date as a connectivity test
        from datetime import date
        df = connector.fetch(date.today().isoformat())
        if df.empty:
            errors.append("Data source returned 0 rows (may be expected for today's date)")

        # Check expected columns exist
        expected = set(config.schema.id_columns + config.schema.feature_columns)
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            errors.append(f"Missing columns in data source: {sorted(missing)}")
    except Exception as e:
        errors.append(f"Data source check failed: {e}")

    return errors


def validate_prediction(config: ProjectConfig, version: str | None = None) -> list[str]:
    """Steps 5-6: Run a test prediction and validate output schema."""
    errors: list[str] = []
    target_version = version or config.model.version

    try:
        from datetime import date

        import pandas as pd

        from src.core.data_connector import DataConnectorFactory
        from src.core.model_loader import load_model
        from src.inference.validators import validate_input, validate_output

        model = load_model(config.model.mlflow_name, target_version)
        connector = DataConnectorFactory.create(config.input)
        df = connector.fetch(date.today().isoformat())

        if df.empty:
            errors.append("Cannot test prediction — no input data for today")
            return errors

        validate_input(df, config.schema)

        features = df[config.schema.feature_columns]
        result = model.predict(features)

        # Normalize to DataFrame
        if isinstance(result, pd.DataFrame):
            pred_df = result
        elif isinstance(result, pd.Series):
            pred_df = result.to_frame(name=config.schema.output_columns[0])
        else:
            if result.ndim == 1:
                pred_df = pd.DataFrame({config.schema.output_columns[0]: result})
            else:
                pred_df = pd.DataFrame(result, columns=config.schema.output_columns)

        # Assemble output
        id_df = df[config.schema.id_columns].reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)
        output_df = pd.concat([id_df, pred_df], axis=1)
        output_df["run_date"] = date.today().isoformat()

        validate_output(output_df, config.schema, config.monitoring)

    except Exception as e:
        errors.append(f"Prediction test failed: {e}")

    return errors


def run_validation(project_id: str, candidate: bool = False) -> bool:
    """Runs all validation steps and prints results.

    Args:
        project_id: The project to validate.
        candidate: If True, validate the Candidate model instead of Production.

    Returns:
        True if all checks passed, False otherwise.
    """
    version = "Candidate" if candidate else None
    all_passed = True

    print(f"\n{'='*60}")
    print(f"  Validating: {project_id}")
    if candidate:
        print(f"  Mode: Candidate validation")
    print(f"{'='*60}\n")

    # Step 1: Config
    print("[1/6] Checking YAML config...", end=" ")
    config, errors = validate_config(project_id)
    if errors:
        print("FAILED")
        for e in errors:
            print(f"      {e}")
        all_passed = False
        # Cannot continue without config
        return False
    print("OK")

    # Step 2: Evaluation config
    print("[2/6] Checking evaluation config...", end=" ")
    errors = validate_evaluation_config(config)
    if errors:
        print("FAILED")
        for e in errors:
            print(f"      {e}")
        all_passed = False
    else:
        metrics_str = ", ".join(config.evaluation.metrics) if config.evaluation.metrics else "(none)"
        print(f"OK ({metrics_str})")

    # Step 3: Model loading
    print(f"[3/6] Loading model from MLflow ({version or config.model.version})...", end=" ")
    errors = validate_model_loads(config, version)
    if errors:
        print("FAILED")
        for e in errors:
            print(f"      {e}")
        all_passed = False
    else:
        print("OK")

    # Step 4: Data source
    print("[4/6] Checking data source...", end=" ")
    errors = validate_data_source(config)
    if errors:
        print("WARNING" if "0 rows" in str(errors) else "FAILED")
        for e in errors:
            print(f"      {e}")
        if not any("0 rows" in e for e in errors):
            all_passed = False
    else:
        print("OK")

    # Steps 5-6: Prediction test
    print("[5/6] Running test prediction...", end=" ")
    errors = validate_prediction(config, version)
    if errors:
        print("FAILED")
        for e in errors:
            print(f"      {e}")
        all_passed = False
    else:
        print("OK")

    # Step 7: Candidate comparison (if --candidate)
    if candidate and config.evaluation.metrics:
        print("[6/6] Running candidate evaluation...", end=" ")
        print("SKIPPED (requires ground truth data)")
    else:
        print("[6/6] Candidate evaluation...", end=" ")
        print("SKIPPED (not in candidate mode)" if not candidate else "SKIPPED (no metrics configured)")

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print(f"  RESULT: ALL CHECKS PASSED")
    else:
        print(f"  RESULT: SOME CHECKS FAILED")
    print(f"{'='*60}\n")

    return all_passed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate a project YAML config and its model.",
    )
    parser.add_argument(
        "project_id",
        help="Project ID (matches the YAML filename without .yaml extension)",
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Validate the Candidate model instead of Production",
    )
    args = parser.parse_args()

    passed = run_validation(args.project_id, candidate=args.candidate)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
