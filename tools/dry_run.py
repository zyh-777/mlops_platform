"""Dry run — test inference without saving to database.

Runs the full pipeline but prints results instead of saving.
Useful for debugging and verifying a new project config.

Usage:
    python -m tools.dry_run <project_id> --date 2025-12-15
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

import pandas as pd

from src.core.data_connector import DataConnectorFactory
from src.core.model_loader import load_model
from src.core.project_config import ProjectConfig
from src.inference.validators import validate_input, validate_output
from src.logger.logger import get_logger

logger = get_logger("dry_run")


def dry_run(config: ProjectConfig, run_date: str) -> pd.DataFrame | None:
    """Executes the inference pipeline without saving results.

    Same steps as InferencePipeline.run(), but instead of writing to DB,
    prints a preview and returns the DataFrame.

    Args:
        config: ProjectConfig loaded from YAML.
        run_date: Date string like "2025-12-15".

    Returns:
        The result DataFrame, or None if a step failed.
    """
    project_id = config.project_id
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  DRY RUN: {project_id} for {run_date}")
    print(f"{'='*60}\n")

    try:
        # Step 1: Load model
        print("[1/6] Loading model...", end=" ")
        model = load_model(config.model.mlflow_name, config.model.version)
        print("OK")

        # Step 2: Fetch data
        print("[2/6] Fetching data...", end=" ")
        connector = DataConnectorFactory.create(config.input)
        input_df = connector.fetch(run_date)
        print(f"OK ({len(input_df)} rows)")

        # Step 3: Validate input
        print("[3/6] Validating input...", end=" ")
        validate_input(input_df, config.schema)
        print("OK")

        # Step 4: Predict
        print("[4/6] Running predictions...", end=" ")
        features = input_df[config.schema.feature_columns]
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
        print(f"OK ({len(pred_df)} predictions)")

        # Step 5: Assemble output
        print("[5/6] Assembling output...", end=" ")
        pred_df = pred_df.reset_index(drop=True)
        id_df = input_df[config.schema.id_columns].reset_index(drop=True)
        result_df = pd.concat([id_df, pred_df], axis=1)
        result_df["run_date"] = run_date
        print("OK")

        # Step 6: Validate output
        print("[6/6] Validating output...", end=" ")
        validate_output(result_df, config.schema, config.monitoring)
        print("OK")

        duration = time.time() - start

        # Print preview
        print(f"\n--- Result Preview (first 10 rows) ---")
        print(result_df.head(10).to_string(index=False))
        print(f"\n--- Summary ---")
        print(f"  Rows:     {len(result_df)}")
        print(f"  Columns:  {list(result_df.columns)}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Target:   {config.output.target_table} (NOT saved — dry run)")

        # Output column stats
        print(f"\n--- Output Column Stats ---")
        for col in config.schema.output_columns:
            series = result_df[col]
            print(
                f"  {col}: min={series.min():.4f}, max={series.max():.4f}, "
                f"mean={series.mean():.4f}, null={series.isna().sum()}"
            )

        print(f"\n{'='*60}")
        print(f"  DRY RUN COMPLETE (results NOT saved)")
        print(f"{'='*60}\n")

        return result_df

    except Exception as e:
        duration = time.time() - start
        print(f"FAILED")
        print(f"\n  Error: {e}")
        print(f"  Duration: {duration:.1f}s")
        logger.error(f"[{project_id}] Dry run failed: {e}")
        return None


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference pipeline without saving results.",
    )
    parser.add_argument(
        "project_id",
        help="Project ID (matches the YAML filename)",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Run date in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    from src.core.project_config import PROJECT_ROOT
    yaml_path = PROJECT_ROOT / "project_registry" / "projects" / f"{args.project_id}.yaml"
    config = ProjectConfig.from_yaml(yaml_path)

    result = dry_run(config, args.date)
    sys.exit(0 if result is not None else 1)


if __name__ == "__main__":
    main()
