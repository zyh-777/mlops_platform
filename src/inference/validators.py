"""Input and output schema validation.

Validates DataFrames against the schema declared in project YAML.
Catches mismatches BEFORE they cause silent prediction errors.

Universal pattern: Fail Fast with Clear Errors
    - Check data quality at two gates: before predict and after predict.
    - Raise specific exceptions so the pipeline knows what went wrong.
    - Every error message includes what was expected vs what was found.

Two custom exceptions:
    - ValidationError: structural problems (missing columns, empty data).
    - QualityError: data quality issues (NaN predictions, out-of-range values).

Phase 1: Column presence checks, null checks, range checks.
Phase 2: Add dtype validation, statistical checks.

Usage:
    validate_input(df, config.schema)
    predictions = model.predict(features)
    validate_output(predictions, config.schema, config.monitoring)
"""

from __future__ import annotations

import pandas as pd

from src.core.project_config import MonitoringConfig, SchemaConfig
from src.logger.logger import get_logger

logger = get_logger("validators")


# ═══════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════
# Why two exceptions? So the pipeline can react differently:
#   - ValidationError → data is structurally wrong, cannot proceed
#   - QualityError → predictions exist but look suspicious


class ValidationError(Exception):
    """Raised when input/output data doesn't match the expected schema.

    Examples: missing columns, empty DataFrame, wrong column names.
    This means the pipeline CANNOT proceed — the data is structurally wrong.
    """


class QualityError(Exception):
    """Raised when predictions exist but fail quality checks.

    Examples: too many NaN values, predictions outside valid range.
    The pipeline ran, but the results are suspicious and shouldn't be saved.
    """


# ═══════════════════════════════════════════════════════════
# INPUT VALIDATION — run BEFORE model.predict()
# ═══════════════════════════════════════════════════════════


def validate_input(df: pd.DataFrame, schema: SchemaConfig) -> None:
    """Validates that input data matches the expected schema.

    Checks (in order):
        1. DataFrame is not empty
        2. All id_columns exist
        3. All feature_columns exist

    Args:
        df: Input DataFrame fetched by the data connector.
        schema: SchemaConfig from project YAML.

    Raises:
        ValidationError: If any check fails.
    """
    # Check 1: Not empty
    if df.empty:
        raise ValidationError("Input DataFrame is empty — no rows returned from data source")

    # Check 2: ID columns present
    missing_ids = set(schema.id_columns) - set(df.columns)
    if missing_ids:
        raise ValidationError(
            f"Missing id_columns: {sorted(missing_ids)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # Check 3: Feature columns present
    missing_features = set(schema.feature_columns) - set(df.columns)
    if missing_features:
        raise ValidationError(
            f"Missing feature_columns: {sorted(missing_features)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    logger.info(
        f"Input validation passed: {len(df)} rows, "
        f"{len(schema.feature_columns)} features"
    )


# ═══════════════════════════════════════════════════════════
# OUTPUT VALIDATION — run AFTER model.predict()
# ═══════════════════════════════════════════════════════════


def validate_output(
    df: pd.DataFrame,
    schema: SchemaConfig,
    monitoring: MonitoringConfig,
) -> None:
    """Validates that prediction output is valid before saving.

    Checks (in order):
        1. DataFrame is not empty
        2. All declared output_columns exist
        3. NaN percentage is within tolerance
        4. Values are within declared prediction_range (if configured)

    Args:
        df: Output DataFrame from model.predict().
        schema: SchemaConfig from project YAML.
        monitoring: MonitoringConfig with quality thresholds.

    Raises:
        ValidationError: If output_columns are missing.
        QualityError: If NaN or range checks fail.
    """
    # Check 1: Not empty
    if df.empty:
        raise ValidationError("Output DataFrame is empty — model returned no predictions")

    # Check 2: Output columns present
    missing_outputs = set(schema.output_columns) - set(df.columns)
    if missing_outputs:
        raise ValidationError(
            f"Missing output_columns: {sorted(missing_outputs)}. "
            f"Model returned columns: {sorted(df.columns.tolist())}"
        )

    # Check 3: NaN check on output columns
    max_null_pct = monitoring.null_output_max_pct
    for col in schema.output_columns:
        null_pct = df[col].isna().mean() * 100
        if null_pct > max_null_pct:
            raise QualityError(
                f"Column '{col}' has {null_pct:.1f}% NaN values "
                f"(threshold: {max_null_pct}%)"
            )

    # Check 4: Prediction range check (if configured)
    for col, (lo, hi) in monitoring.prediction_range.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_range.empty:
            raise QualityError(
                f"Column '{col}' has {len(out_of_range)} values "
                f"outside range [{lo}, {hi}]"
            )

    logger.info(
        f"Output validation passed: {len(df)} rows, "
        f"{len(schema.output_columns)} output columns"
    )
