"""Inference pipeline — the core engine of the platform.

One class, one run() method, works for ALL projects.
The only thing that varies is the ProjectConfig passed in.

Universal pattern: Orchestrator
    - Each step is a private method with a clear responsibility.
    - The run() method calls them in sequence.
    - Any step failure is caught, logged to run_log, and re-raised.
    - This makes debugging easy: the run_log tells you which step failed.

Pipeline steps:
    1. Load model from MLflow (via pyfunc)
    2. Fetch input data (via DataConnector)
    3. Validate input schema
    4. Run model.predict() on feature columns
    5. Assemble output (id_columns + predictions + run_date)
    6. Validate output schema
    7. Save results to project-specific DB table
    8. Log run status to platform.run_log

Phase 1: Steps 1-8 (basic pipeline).
Phase 2: Add monitoring step between 7 and 8.

Usage:
    config = ProjectConfig.from_yaml("project_registry/projects/fraud_detection.yaml")
    pipeline = InferencePipeline(config)
    pipeline.run("2026-03-05")
"""

from __future__ import annotations

import time

import pandas as pd
from mlflow.pyfunc import PyFuncModel

from src.core.data_connector import DataConnectorFactory
from src.core.model_loader import load_model
from src.core.project_config import ProjectConfig
from src.database.result_store import log_run, save
from src.database.schema_manager import ensure_result_table
from src.database.db_connection import get_engine
from src.inference.validators import validate_input, validate_output
from src.logger.logger import get_logger

logger = get_logger("pipeline")


class InferencePipeline:
    """Config-driven inference engine — one class for ALL projects.

    Args:
        config: ProjectConfig loaded from YAML. Contains everything
                the pipeline needs: model name, data source, schema,
                output table, monitoring thresholds.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.project_id = config.project_id

    def run(self, run_date: str) -> None:
        """Executes the full inference pipeline for one run date.

        Steps:
            1. Load model → 2. Fetch data → 3. Validate input →
            4. Predict → 5. Assemble output → 6. Validate output →
            7. Save results → 8. Log run

        On failure at any step, logs the error to run_log and re-raises.

        Args:
            run_date: Date string like "2026-03-05".
        """
        logger.info(f"[{self.project_id}] Starting inference for {run_date}")
        start_time = time.time()
        row_count = 0

        try:
            # Step 1: Load model from MLflow
            model = self._load_model()

            # Step 2: Fetch input data
            input_df = self._fetch_data(run_date)

            # Step 3: Validate input schema
            self._validate_input(input_df)

            # Step 4: Run predictions
            predictions = self._predict(model, input_df)

            # Step 5: Assemble output DataFrame
            result_df = self._assemble_output(input_df, predictions, run_date)

            # Step 6: Validate output
            self._validate_output(result_df)

            # Step 7: Save results to database
            row_count = self._save_results(result_df, run_date)

            # Step 8: Log success
            duration = time.time() - start_time
            log_run(self.config, run_date, "success", row_count, duration)
            logger.info(
                f"[{self.project_id}] Inference complete: "
                f"{row_count} rows in {duration:.1f}s"
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{self.project_id}] Pipeline failed: {e}")
            log_run(
                self.config, run_date, "failed", row_count, duration,
                error_message=str(e),
            )
            raise

    # ═══════════════════════════════════════════════════════════
    # PRIVATE STEP METHODS
    # ═══════════════════════════════════════════════════════════

    def _load_model(self) -> PyFuncModel:
        """Step 1: Load model from MLflow via pyfunc."""
        logger.info(
            f"[{self.project_id}] Loading model "
            f"'{self.config.model.mlflow_name}' (version={self.config.model.version})"
        )
        return load_model(
            self.config.model.mlflow_name,
            self.config.model.version,
        )

    def _fetch_data(self, run_date: str) -> pd.DataFrame:
        """Step 2: Fetch input data via the appropriate DataConnector."""
        connector = DataConnectorFactory.create(self.config.input)
        df = connector.fetch(run_date)
        logger.info(f"[{self.project_id}] Fetched {len(df)} rows for {run_date}")
        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Step 3: Validate input DataFrame against declared schema."""
        validate_input(df, self.config.schema)

    def _predict(self, model: PyFuncModel, input_df: pd.DataFrame) -> pd.DataFrame:
        """Step 4: Run model.predict() on feature columns only.

        The model receives ONLY the feature columns declared in YAML.
        ID columns are NOT passed to the model — they're carried through
        separately and joined back in the assemble step.
        """
        features = input_df[self.config.schema.feature_columns]
        logger.info(
            f"[{self.project_id}] Running predict on {len(features)} rows, "
            f"{len(self.config.schema.feature_columns)} features"
        )

        result = model.predict(features)

        # model.predict() can return ndarray, Series, or DataFrame.
        # Normalize to DataFrame with output_columns as column names.
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, pd.Series):
            return result.to_frame(name=self.config.schema.output_columns[0])
        else:
            # numpy array — could be 1D or 2D
            if result.ndim == 1:
                return pd.DataFrame(
                    {self.config.schema.output_columns[0]: result}
                )
            else:
                return pd.DataFrame(
                    result,
                    columns=self.config.schema.output_columns,
                )

    def _assemble_output(
        self,
        input_df: pd.DataFrame,
        predictions: pd.DataFrame,
        run_date: str,
    ) -> pd.DataFrame:
        """Step 5: Combine id_columns + predictions + run_date.

        The result table gets:
            - id_columns (from input data, for joining back to source)
            - output_columns (from model predictions)
            - run_date (for filtering and dedup)
        """
        # Reset index on predictions to align with input_df
        predictions = predictions.reset_index(drop=True)
        id_df = input_df[self.config.schema.id_columns].reset_index(drop=True)

        result = pd.concat([id_df, predictions], axis=1)
        result["run_date"] = run_date
        return result

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Step 6: Validate output DataFrame before saving."""
        validate_output(df, self.config.schema, self.config.monitoring)

    def _save_results(self, df: pd.DataFrame, run_date: str) -> int:
        """Step 7: Save predictions to the project's result table."""
        # Ensure result table exists (idempotent)
        engine = get_engine("platform")
        ensure_result_table(engine, self.config)

        return save(df, self.config, run_date)
