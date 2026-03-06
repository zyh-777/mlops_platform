"""Unified result storage with deduplication.

Saves prediction DataFrames to project-specific tables.
Handles dedup logic (don't insert duplicates on re-runs).
Logs run metadata to platform.run_log for audit trail.

Universal patterns:
    Write-with-Dedup: Check before writing so re-runs are safe.
        - append mode: skip if run_date data already exists.
        - replace_date mode: delete old rows for that date, then insert.
    Audit Log: Every run (success or failure) gets a row in run_log.
        This answers "did the model run yesterday?" without querying predictions.

Usage:
    row_count = save(predictions_df, config, "2026-03-05")
    log_run(config, "2026-03-05", "success", row_count, duration_sec=45.2)
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.database.db_connection import get_engine
from src.logger.logger import get_logger

logger = get_logger("result_store")


def save(df: pd.DataFrame, config, run_date: str) -> int:
    """Saves prediction results to the project's result table.

    Dedup logic based on config.output.write_mode:
        - "append": if rows for this run_date exist, skip (log warning).
        - "replace_date": delete existing rows for this run_date, then insert.

    Args:
        df: Prediction DataFrame (id_columns + output_columns).
        config: ProjectConfig with output and schema sections.
        run_date: Date string like "2026-03-05".

    Returns:
        Number of rows saved (0 if skipped due to dedup).
    """
    table_name = config.output.target_table
    write_mode = config.output.write_mode

    # Parse "results.fraud_scores" → schema="results", table="fraud_scores"
    if "." in table_name:
        schema_name, bare_table = table_name.split(".", 1)
    else:
        schema_name, bare_table = None, table_name

    engine = get_engine(schema_name or "")

    # --- Dedup check: does data for this run_date already exist? ---
    existing_count = _count_existing(engine, table_name, run_date)

    if existing_count > 0:
        if write_mode == "replace_date":
            # Delete old rows, then insert new ones
            _delete_by_date(engine, table_name, run_date)
            logger.info(
                f"[{config.project_id}] Replaced {existing_count} existing rows "
                f"for {run_date}"
            )
        else:
            # Default "append" with dedup: skip if data exists
            logger.warning(
                f"[{config.project_id}] Data for {run_date} already exists "
                f"({existing_count} rows). Skipping to prevent duplicates."
            )
            return 0

    # --- Add run_date column if not already present ---
    if "run_date" not in df.columns:
        df = df.copy()
        df["run_date"] = run_date

    # --- Write to database ---
    df.to_sql(
        name=bare_table,
        con=engine,
        schema=schema_name,
        if_exists="append",
        index=False,
    )

    logger.info(f"[{config.project_id}] Saved {len(df)} rows to {table_name}")
    return len(df)


def log_run(
    config,
    run_date: str,
    status: str,
    row_count: int = 0,
    duration_sec: float = 0.0,
    error_message: str | None = None,
) -> None:
    """Records a pipeline run in platform.run_log.

    Called after every run — success or failure. This is the audit trail
    that answers "when did this model last run? did it succeed?"

    Args:
        config: ProjectConfig (needs project_id).
        run_date: Date of the inference run.
        status: "success", "failed", or "skipped".
        row_count: Number of prediction rows produced.
        duration_sec: How long the run took in seconds.
        error_message: Error details if status is "failed".
    """
    engine = get_engine("platform")

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO run_log
                    (project_id, run_date, status, row_count, duration_sec, error_message)
                VALUES
                    (:project_id, :run_date, :status, :row_count, :duration_sec, :error_message)
            """),
            {
                "project_id": config.project_id,
                "run_date": run_date,
                "status": status,
                "row_count": row_count,
                "duration_sec": round(duration_sec, 2),
                "error_message": error_message,
            },
        )
        conn.commit()

    logger.info(
        f"[{config.project_id}] Logged run: status={status}, "
        f"rows={row_count}, duration={duration_sec:.1f}s"
    )


# ═══════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════


def _count_existing(engine, table_name: str, run_date: str) -> int:
    """Counts rows in the result table for a given run_date."""
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {table_name} WHERE run_date = :run_date"),
            {"run_date": run_date},
        )
        return result.scalar()


def _delete_by_date(engine, table_name: str, run_date: str) -> None:
    """Deletes all rows for a given run_date (used by replace_date mode)."""
    with engine.connect() as conn:
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE run_date = :run_date"),
            {"run_date": run_date},
        )
        conn.commit()
