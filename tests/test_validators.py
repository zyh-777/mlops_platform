"""Tests for src/inference/validators.py"""

from __future__ import annotations

import pytest
import pandas as pd

from src.core.project_config import MonitoringConfig, SchemaConfig
from src.inference.validators import (
    QualityError,
    ValidationError,
    validate_input,
    validate_output,
)


@pytest.fixture
def schema() -> SchemaConfig:
    return SchemaConfig(
        id_columns=["txn_id", "txn_date"],
        feature_columns=["amount", "merchant_risk", "hour_of_day"],
        output_columns=["fraud_probability"],
    )


@pytest.fixture
def monitoring() -> MonitoringConfig:
    return MonitoringConfig(
        null_output_max_pct=0.0,
        prediction_range={"fraud_probability": [0.0, 1.0]},
    )


@pytest.fixture
def good_input_df() -> pd.DataFrame:
    return pd.DataFrame({
        "txn_id": ["T1", "T2", "T3"],
        "txn_date": ["2026-03-05"] * 3,
        "amount": [100.0, 250.0, 50.0],
        "merchant_risk": [0.3, 0.8, 0.1],
        "hour_of_day": [14, 22, 8],
    })


class TestValidateInput:
    def test_passes_with_valid_data(self, good_input_df, schema):
        validate_input(good_input_df, schema)  # Should not raise

    def test_raises_on_empty_df(self, schema):
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            validate_input(df, schema)

    def test_raises_on_missing_id_columns(self, schema):
        df = pd.DataFrame({
            "amount": [1.0],
            "merchant_risk": [0.5],
            "hour_of_day": [10],
        })
        with pytest.raises(ValidationError, match="id_columns"):
            validate_input(df, schema)

    def test_raises_on_missing_feature_columns(self, schema):
        df = pd.DataFrame({
            "txn_id": ["T1"],
            "txn_date": ["2026-03-05"],
            "amount": [100.0],
            # missing merchant_risk, hour_of_day
        })
        with pytest.raises(ValidationError, match="feature_columns"):
            validate_input(df, schema)

    def test_extra_columns_are_ok(self, good_input_df, schema):
        df = good_input_df.copy()
        df["extra_col"] = "ignored"
        validate_input(df, schema)  # Should not raise


class TestValidateOutput:
    def test_passes_with_valid_output(self, schema, monitoring):
        df = pd.DataFrame({
            "txn_id": ["T1", "T2"],
            "fraud_probability": [0.1, 0.9],
            "run_date": ["2026-03-05"] * 2,
        })
        validate_output(df, schema, monitoring)

    def test_raises_on_empty_df(self, schema, monitoring):
        df = pd.DataFrame()
        with pytest.raises(ValidationError, match="empty"):
            validate_output(df, schema, monitoring)

    def test_raises_on_missing_output_columns(self, schema, monitoring):
        df = pd.DataFrame({
            "txn_id": ["T1"],
            "wrong_column": [0.5],
        })
        with pytest.raises(ValidationError, match="output_columns"):
            validate_output(df, schema, monitoring)

    def test_raises_on_nan_output(self, schema, monitoring):
        df = pd.DataFrame({
            "fraud_probability": [0.1, None, 0.8],
            "run_date": ["2026-03-05"] * 3,
        })
        with pytest.raises(QualityError, match="NaN"):
            validate_output(df, schema, monitoring)

    def test_allows_nan_when_threshold_permits(self, schema):
        mon = MonitoringConfig(null_output_max_pct=50.0, prediction_range={})
        df = pd.DataFrame({
            "fraud_probability": [0.1, None, 0.8],
            "run_date": ["2026-03-05"] * 3,
        })
        validate_output(df, schema, mon)  # 33% NaN < 50% threshold

    def test_raises_on_out_of_range(self, schema, monitoring):
        df = pd.DataFrame({
            "fraud_probability": [0.1, 1.5, 0.8],
            "run_date": ["2026-03-05"] * 3,
        })
        with pytest.raises(QualityError, match="outside range"):
            validate_output(df, schema, monitoring)

    def test_passes_when_no_range_configured(self, schema):
        mon = MonitoringConfig(null_output_max_pct=0.0, prediction_range={})
        df = pd.DataFrame({
            "fraud_probability": [0.1, 999.0, -5.0],
            "run_date": ["2026-03-05"] * 3,
        })
        validate_output(df, schema, mon)  # No range check configured
