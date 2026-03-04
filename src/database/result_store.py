"""Unified result storage with deduplication.

Saves prediction DataFrames to project-specific tables.
Handles dedup logic (don't insert if run_date already exists).
Logs run metadata to platform.run_log.

Phase 1: Basic save with dedup check.
Phase 2: Add write_mode support (append vs replace_date).
"""

from __future__ import annotations

# TODO: Implement save(df, config: ProjectConfig, run_date: str) -> int
#   - Check dedup: does data for this run_date + dedup_key already exist?
#   - If exists, log warning and skip (or replace if write_mode == "replace_date")
#   - If not, df.to_sql with if_exists="append"
#   - Return row count saved

# TODO: Implement log_run(config, run_date, status, row_count, duration, error) -> None
#   - Insert into platform.run_log
