"""Auto-creates database tables from project YAML config.

On first run of a project, creates the result table based on
output_columns declared in the YAML. Also ensures platform-level
tables (run_log, monitoring_log, project_registry) exist.

Universal pattern: Idempotent DDL
    - Every CREATE TABLE uses IF NOT EXISTS — safe to call repeatedly.
    - Running this once or 1000 times produces the same result.
    - The pipeline calls this at startup; it's a no-op after the first run.

Phase 1: Create result tables with TEXT columns (simple, works for anything).
Phase 2: Add dtype inference, indexes, partitioning.

Usage:
    engine = get_engine()
    ensure_platform_tables(engine)          # run_log, monitoring_log
    ensure_result_table(engine, config)     # project-specific prediction table
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.logger.logger import get_logger

logger = get_logger("schema_manager")


# ═══════════════════════════════════════════════════════════
# PLATFORM TABLES — fixed schema, shared across all projects
# ═══════════════════════════════════════════════════════════
# These track operational metadata: did the run succeed? how long?
# any alerts? They know nothing about what any model predicts.


def ensure_platform_tables(engine: Engine) -> None:
    """Creates platform-level tables if they don't exist.

    Three tables:
        - project_registry: which projects are registered
        - run_log: one row per pipeline run (success/failure, duration)
        - monitoring_log: one row per health check result

    Args:
        engine: SQLAlchemy engine connected to the platform database.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS project_registry (
                project_id       VARCHAR(100) PRIMARY KEY,
                display_name     VARCHAR(255),
                owner            VARCHAR(100),
                status           VARCHAR(20)  DEFAULT 'active',
                created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
                last_run_at      TIMESTAMP    NULL
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS run_log (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                project_id       VARCHAR(100) NOT NULL,
                run_date         DATE         NOT NULL,
                status           VARCHAR(20)  NOT NULL,
                row_count        INT          DEFAULT 0,
                duration_sec     FLOAT        DEFAULT 0,
                error_message    TEXT,
                created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_run_project_date (project_id, run_date)
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS monitoring_log (
                id               INT AUTO_INCREMENT PRIMARY KEY,
                project_id       VARCHAR(100) NOT NULL,
                run_date         DATE         NOT NULL,
                check_name       VARCHAR(100) NOT NULL,
                passed           BOOLEAN      NOT NULL,
                value            FLOAT,
                threshold        FLOAT,
                created_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_mon_project_date (project_id, run_date)
            )
        """))

        conn.commit()

    logger.info("Platform tables verified (project_registry, run_log, monitoring_log)")


# ═══════════════════════════════════════════════════════════
# RESULT TABLES — team-defined schema, one per project
# ═══════════════════════════════════════════════════════════
# Each project declares its own table name and columns in YAML.
# We auto-create the table on first run. The platform has no opinion
# on what the columns mean — it just saves whatever the model outputs.


def ensure_result_table(engine: Engine, config) -> None:
    """Creates a project's result table if it doesn't exist.

    The table has:
        - id_columns from schema (e.g., txn_id, comp_id)
        - run_date column (always added, used for dedup and partitioning)
        - output_columns from schema (e.g., fraud_probability, pd_1m)

    All columns default to TEXT type for simplicity in Phase 1.
    Phase 2 can infer types from the first batch of real data.

    Args:
        engine: SQLAlchemy engine for the results database.
        config: ProjectConfig with schema and output sections.
    """
    table_name = config.output.target_table

    # Build column list: id_columns + run_date + output_columns
    columns = []
    for col in config.schema.id_columns:
        columns.append(f"`{col}` VARCHAR(255)")

    # run_date is always present — used for filtering and dedup
    columns.append("`run_date` DATE NOT NULL")

    for col in config.schema.output_columns:
        columns.append(f"`{col}` TEXT")

    # created_at for audit trail
    columns.append("`created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    columns_sql = ",\n                ".join(columns)

    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
                {columns_sql}
        )
    """

    with engine.connect() as conn:
        # Ensure the schema/database exists (e.g., "results")
        if "." in table_name:
            schema_name = table_name.split(".")[0]
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{schema_name}`"))

        conn.execute(text(create_sql))
        conn.commit()

    logger.info(f"[{config.project_id}] Result table verified: {table_name}")
