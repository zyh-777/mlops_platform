"""Backfill — re-run inference for a range of past dates.

Runs the inference pipeline for each date in the range sequentially.
Useful after fixing a data issue or onboarding a new project with
historical data.

Usage:
    python -m tools.backfill <project_id> --from 2025-12-01 --to 2025-12-15
    python -m tools.backfill <project_id> --from 2025-12-01 --to 2025-12-15 --replace
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

from src.core.project_config import PROJECT_ROOT, ProjectConfig
from src.inference.pipeline import InferencePipeline
from src.logger.logger import get_logger

logger = get_logger("backfill")


def backfill(
    config: ProjectConfig,
    start_date: date,
    end_date: date,
    replace: bool = False,
) -> dict[str, str]:
    """Runs the inference pipeline for each date in [start_date, end_date].

    Args:
        config: ProjectConfig loaded from YAML.
        start_date: First date to backfill (inclusive).
        end_date: Last date to backfill (inclusive).
        replace: If True, set write_mode to "replace_date" to overwrite existing data.

    Returns:
        Dict mapping date string to status ("success" or error message).
    """
    project_id = config.project_id
    results: dict[str, str] = {}

    # Override write_mode if --replace is set
    if replace:
        config.output.write_mode = "replace_date"

    # Generate date range
    dates: list[date] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    print(f"\n{'='*60}")
    print(f"  BACKFILL: {project_id}")
    print(f"  Range: {start_date} to {end_date} ({len(dates)} days)")
    print(f"  Mode: {'replace' if replace else 'append (skip existing)'}")
    print(f"{'='*60}\n")

    pipeline = InferencePipeline(config)

    for i, d in enumerate(dates, 1):
        date_str = d.isoformat()
        print(f"[{i}/{len(dates)}] {date_str}...", end=" ")

        try:
            pipeline.run(date_str)
            results[date_str] = "success"
            print("OK")
        except Exception as e:
            results[date_str] = str(e)
            print(f"FAILED ({e})")
            logger.error(f"[{project_id}] Backfill failed for {date_str}: {e}")

    # Summary
    success_count = sum(1 for v in results.values() if v == "success")
    fail_count = len(results) - success_count

    print(f"\n{'='*60}")
    print(f"  BACKFILL COMPLETE: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*60}\n")

    if fail_count > 0:
        print("Failed dates:")
        for date_str, status in results.items():
            if status != "success":
                print(f"  {date_str}: {status}")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Re-run inference for a range of past dates.",
    )
    parser.add_argument(
        "project_id",
        help="Project ID (matches the YAML filename)",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        required=True,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        required=True,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing data for each date instead of skipping",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    if start_date > end_date:
        print(f"Error: --from ({start_date}) is after --to ({end_date})")
        sys.exit(1)

    yaml_path = PROJECT_ROOT / "project_registry" / "projects" / f"{args.project_id}.yaml"
    config = ProjectConfig.from_yaml(yaml_path)

    results = backfill(config, start_date, end_date, replace=args.replace)
    failed = sum(1 for v in results.values() if v != "success")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
