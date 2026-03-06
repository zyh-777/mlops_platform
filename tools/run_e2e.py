"""Step 5: Run the full pipeline end-to-end and verify results.

This is the final step — it does exactly what the platform does in production:
    1. Load model from MLflow
    2. Fetch data from MySQL
    3. Validate input schema
    4. Predict (model.predict on feature columns)
    5. Assemble output (id_columns + predictions + run_date)
    6. Validate output
    7. Save results to MySQL (results.pd_daily_us)
    8. Log run to platform.run_log

Prerequisites:
    1. Docker containers running:     docker compose -f docker-compose.dev.yaml up -d
    2. CSV loaded into MySQL:         python -m tools.load_csv_to_mysql --sample 50000
    3. Model registered in MLflow:    python -m tools.train_and_register

Usage:
    python -m tools.run_e2e                      # full run
    python -m tools.run_e2e --date 2026-03-05    # specific date
    python -m tools.run_e2e --dry-run            # prints results, doesn't save to DB
    python -m tools.run_e2e --step               # pause between each pipeline step
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

import pandas as pd
from sqlalchemy import text

from src.logger.logger import get_logger

logger = get_logger("run_e2e")


def check_prerequisites() -> bool:
    """Verifies that MySQL, MLflow, and data are ready."""
    all_ok = True

    # Check MySQL
    print("  [1/3] MySQL...", end=" ", flush=True)
    try:
        from src.database.db_connection import check_connection
        if check_connection("platform"):
            print("OK (localhost:3306)")
        else:
            print("FAILED — run: docker compose -f docker-compose.dev.yaml up -d")
            all_ok = False
    except Exception as e:
        print(f"FAILED ({e})")
        all_ok = False

    # Check MLflow
    print("  [2/3] MLflow...", end=" ", flush=True)
    try:
        import mlflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = mlflow.MlflowClient()
        client.search_experiments()
        print("OK (localhost:5000)")
    except Exception as e:
        print(f"FAILED ({e})")
        all_ok = False

    # Check features table in MySQL
    print("  [3/3] Features table...", end=" ", flush=True)
    try:
        from src.database.db_connection import get_engine
        engine = get_engine("features")
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM pd_input")).scalar()
        print(f"OK ({count:,} rows in features.pd_input)")
    except Exception as e:
        print(f"FAILED ({e})")
        print("         Run: python -m tools.load_csv_to_mysql --sample 50000")
        all_ok = False

    return all_ok


def run_e2e(run_date: str, dry_run: bool = False, step_mode: bool = False) -> None:
    """Runs the full end-to-end pipeline."""
    print(f"\n{'='*60}")
    print(f"  END-TO-END PIPELINE TEST")
    print(f"  Date:  {run_date}")
    print(f"  Mode:  {'DRY RUN (no DB save)' if dry_run else 'FULL (saves to MySQL)'}")
    print(f"  Steps: {'Interactive (press Enter)' if step_mode else 'Automatic'}")
    print(f"{'='*60}\n")

    # Step 0: Prerequisites
    print("--- Checking Prerequisites ---")
    if not check_prerequisites():
        print("\nSome prerequisites failed. Fix them and try again.")
        sys.exit(1)
    print()

    if step_mode:
        input("Press Enter to continue...")

    # Load config
    from src.core.project_config import PROJECT_ROOT, ProjectConfig
    yaml_path = PROJECT_ROOT / "project_registry" / "projects" / "_test_credit_risk_us.yaml"
    config = ProjectConfig.from_yaml(yaml_path)
    print(f"--- Config loaded: {config.project_id} ---")
    print(f"  Model:    {config.model.mlflow_name} ({config.model.version})")
    print(f"  Source:   {config.input.source} → {config.input.connection.get('database', '')}")
    print(f"  Features: {len(config.schema.feature_columns)} columns")
    print(f"  Output:   {config.output.target_table}")
    print()

    if step_mode:
        input("Press Enter to continue...")

    if dry_run:
        from tools.dry_run import dry_run as do_dry_run
        result = do_dry_run(config, run_date)
        sys.exit(0 if result is not None else 1)

    # Set up platform tables
    print("--- Step 0: Setting up database tables ---")
    from src.database.db_connection import get_engine
    from src.database.schema_manager import ensure_platform_tables, ensure_result_table
    engine = get_engine("platform")
    ensure_platform_tables(engine)
    ensure_result_table(engine, config)
    print("  Platform tables: OK (run_log, monitoring_log)")
    print(f"  Result table: OK ({config.output.target_table})")
    print()

    if step_mode:
        input("Press Enter to start the pipeline...")

    # Run the 8-step pipeline
    print("--- Running Pipeline (8 steps) ---\n")

    from src.core.data_connector import DataConnectorFactory
    from src.core.model_loader import load_model
    from src.inference.validators import validate_input, validate_output
    from src.database.result_store import save, log_run
    import numpy as np

    start_time = time.time()
    row_count = 0

    try:
        # Step 1: Load model
        print("  Step 1/8: Loading model from MLflow...", flush=True)
        model = load_model(config.model.mlflow_name, config.model.version)
        print(f"           OK — loaded {config.model.mlflow_name}/{config.model.version}")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 2: Fetch data
        print("\n  Step 2/8: Fetching data from MySQL...", flush=True)
        connector = DataConnectorFactory.create(config.input)
        input_df = connector.fetch(run_date)
        print(f"           OK — {len(input_df):,} rows, {input_df.shape[1]} columns")
        print(f"           Columns: {list(input_df.columns[:5])}... (+{input_df.shape[1]-5} more)")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 3: Validate input
        print("\n  Step 3/8: Validating input schema...", flush=True)
        validate_input(input_df, config.schema)
        null_pct = input_df[config.schema.feature_columns].isnull().mean().max() * 100
        print(f"           OK — all columns present, max null rate: {null_pct:.1f}%")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 4: Predict
        print("\n  Step 4/8: Running model.predict()...", flush=True)
        features = input_df[config.schema.feature_columns]
        raw_result = model.predict(features)

        # Normalize output (same logic as pipeline._predict)
        if isinstance(raw_result, pd.DataFrame):
            predictions = raw_result
        elif isinstance(raw_result, pd.Series):
            predictions = raw_result.to_frame(name=config.schema.output_columns[0])
        else:
            if raw_result.ndim == 1:
                predictions = pd.DataFrame({config.schema.output_columns[0]: raw_result})
            else:
                predictions = pd.DataFrame(raw_result, columns=config.schema.output_columns)

        print(f"           OK — {len(predictions):,} predictions")
        print(f"           PD stats: mean={predictions['pd_1y'].mean():.4f}, "
              f"min={predictions['pd_1y'].min():.4f}, max={predictions['pd_1y'].max():.4f}")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 5: Assemble output
        print("\n  Step 5/8: Assembling output...", flush=True)
        predictions = predictions.reset_index(drop=True)
        id_df = input_df[config.schema.id_columns].reset_index(drop=True)
        result_df = pd.concat([id_df, predictions], axis=1)
        result_df["run_date"] = run_date
        print(f"           OK — {result_df.shape[1]} columns: {list(result_df.columns)}")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 6: Validate output
        print("\n  Step 6/8: Validating output...", flush=True)
        validate_output(result_df, config.schema, config.monitoring)
        print(f"           OK — no NaN, all predictions in [0, 1]")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 7: Save to MySQL
        print("\n  Step 7/8: Saving results to MySQL...", flush=True)
        row_count = save(result_df, config, run_date)
        print(f"           OK — {row_count:,} rows saved to {config.output.target_table}")
        if step_mode:
            input("  Press Enter for next step...")

        # Step 8: Log run
        print("\n  Step 8/8: Logging run to platform.run_log...", flush=True)
        duration = time.time() - start_time
        log_run(config, run_date, "success", row_count, duration)
        print(f"           OK — status=success, duration={duration:.1f}s")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n  FAILED at step: {e}")
        log_run(config, run_date, "failed", row_count, duration, error_message=str(e))
        raise

    # --- Verify results in MySQL ---
    print(f"\n\n--- Verifying Results ---")

    # Check run_log
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM run_log WHERE project_id = :pid ORDER BY created_at DESC LIMIT 1"
        ), {"pid": config.project_id})
        row = result.fetchone()
        if row:
            print(f"  run_log: status={row.status}, rows={row.row_count}, "
                  f"duration={row.duration_sec}s")

    # Check result table
    result_engine = get_engine("results")
    with result_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM pd_daily_us WHERE run_date = :rd"
        ), {"rd": run_date})
        count = result.scalar()
        print(f"  results: {count:,} rows in results.pd_daily_us for {run_date}")

        # Show a sample
        sample = pd.read_sql(text(
            "SELECT * FROM pd_daily_us WHERE run_date = :rd LIMIT 5"
        ), conn, params={"rd": run_date})
        print(f"\n--- Sample predictions ---")
        print(sample.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  END-TO-END TEST COMPLETE")
    print(f"  {row_count:,} predictions saved in {duration:.1f}s")
    print(f"  MLflow UI: http://localhost:5000")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full end-to-end pipeline test.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Run date (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without saving to DB",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Pause between each pipeline step (interactive mode)",
    )
    args = parser.parse_args()
    run_e2e(args.date, dry_run=args.dry_run, step_mode=args.step)


if __name__ == "__main__":
    main()
