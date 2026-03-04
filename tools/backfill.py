"""Backfill — re-run inference for a range of past dates.

Usage:
    python -m tools.backfill <project_id> --from 2025-12-01 --to 2025-12-15

Runs the inference pipeline for each date in the range.
Useful after fixing a data issue or onboarding a new project with historical data.
"""

from __future__ import annotations

# TODO (Phase 2): Implement backfill with date range iteration
