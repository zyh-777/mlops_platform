"""Post-inference monitoring checks.

Runs after each inference to catch data pipeline failures
and model degradation. Results logged to platform.monitoring_log.

Three levels of checks:
    Level 1 — Business rules: row count change, NaN outputs, range violations.
    Level 2 — Feature drift (PSI) — delegated to drift_detector.py.
    Level 3 — Prediction distribution shift — combined with drift.

Usage:
    monitor = Monitor(config)
    alerts = monitor.check(result_df, run_date)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

from src.core.project_config import ProjectConfig
from src.database.db_connection import get_engine
from src.logger.logger import get_logger

logger = get_logger("monitor")


@dataclass
class CheckResult:
    """Result of a single monitoring check."""
    check_name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    message: str = ""


class Monitor:
    """Runs post-inference health checks for a project.

    All checks are driven by config.monitoring from YAML.
    Results are logged to platform.monitoring_log.

    Args:
        config: ProjectConfig with monitoring thresholds.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.project_id = config.project_id

    def check(self, result_df: pd.DataFrame, run_date: str) -> list[CheckResult]:
        """Runs all configured monitoring checks.

        Args:
            result_df: The prediction DataFrame that was just saved.
            run_date: Date of the current run.

        Returns:
            List of CheckResult objects (one per check).
        """
        results: list[CheckResult] = []

        # Level 1: Business rule checks
        results.append(self._check_row_count(result_df, run_date))
        results.append(self._check_null_outputs(result_df))
        results.extend(self._check_prediction_ranges(result_df))

        # Log all results to monitoring_log
        self._log_results(results, run_date)

        # Report
        failed = [r for r in results if not r.passed]
        if failed:
            for r in failed:
                logger.warning(
                    f"[{self.project_id}] ALERT: {r.check_name} — {r.message}"
                )
        else:
            logger.info(
                f"[{self.project_id}] All {len(results)} monitoring checks passed"
            )

        return results

    def _check_row_count(self, result_df: pd.DataFrame, run_date: str) -> CheckResult:
        """Compares today's row count against yesterday's run.

        Alerts if the change exceeds row_count_change_max_pct.
        """
        current_count = len(result_df)
        max_pct = self.config.monitoring.row_count_change_max_pct

        yesterday_count = self._get_yesterday_row_count(run_date)

        if yesterday_count is None or yesterday_count == 0:
            return CheckResult(
                check_name="row_count_change",
                passed=True,
                value=float(current_count),
                threshold=max_pct,
                message=f"No previous run to compare (count={current_count})",
            )

        change_pct = abs(current_count - yesterday_count) / yesterday_count * 100

        passed = change_pct <= max_pct
        return CheckResult(
            check_name="row_count_change",
            passed=passed,
            value=round(change_pct, 1),
            threshold=max_pct,
            message=(
                f"Row count changed {change_pct:.1f}% "
                f"({yesterday_count} -> {current_count}, "
                f"max allowed: {max_pct}%)"
            ),
        )

    def _check_null_outputs(self, result_df: pd.DataFrame) -> CheckResult:
        """Checks NaN rate across all output columns."""
        max_pct = self.config.monitoring.null_output_max_pct
        output_cols = self.config.schema.output_columns

        total_cells = len(result_df) * len(output_cols)
        if total_cells == 0:
            return CheckResult(
                check_name="null_outputs",
                passed=True,
                value=0.0,
                threshold=max_pct,
                message="No output cells to check",
            )

        null_count = result_df[output_cols].isna().sum().sum()
        null_pct = null_count / total_cells * 100

        passed = null_pct <= max_pct
        return CheckResult(
            check_name="null_outputs",
            passed=passed,
            value=round(null_pct, 2),
            threshold=max_pct,
            message=f"{null_pct:.2f}% NaN in output columns (max: {max_pct}%)",
        )

    def _check_prediction_ranges(self, result_df: pd.DataFrame) -> list[CheckResult]:
        """Checks each output column against its declared valid range."""
        results: list[CheckResult] = []

        for col, (lo, hi) in self.config.monitoring.prediction_range.items():
            if col not in result_df.columns:
                continue

            series = result_df[col].dropna()
            if series.empty:
                results.append(CheckResult(
                    check_name=f"range_{col}",
                    passed=True,
                    message=f"No non-null values in '{col}'",
                ))
                continue

            out_of_range = len(series[(series < lo) | (series > hi)])
            pct = out_of_range / len(series) * 100
            passed = out_of_range == 0

            results.append(CheckResult(
                check_name=f"range_{col}",
                passed=passed,
                value=round(pct, 2),
                threshold=0.0,
                message=(
                    f"{out_of_range}/{len(series)} values outside "
                    f"[{lo}, {hi}] ({pct:.2f}%)"
                ),
            ))

        return results

    def _get_yesterday_row_count(self, run_date: str) -> int | None:
        """Queries run_log for yesterday's row count."""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT row_count FROM run_log
                        WHERE project_id = :project_id
                          AND run_date < :run_date
                          AND status = 'success'
                        ORDER BY run_date DESC
                        LIMIT 1
                    """),
                    {"project_id": self.project_id, "run_date": run_date},
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(
                f"[{self.project_id}] Could not fetch previous row count: {e}"
            )
            return None

    def _log_results(self, results: list[CheckResult], run_date: str) -> None:
        """Writes check results to platform.monitoring_log."""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                for r in results:
                    conn.execute(
                        text("""
                            INSERT INTO monitoring_log
                                (project_id, run_date, check_name, passed, value, threshold)
                            VALUES
                                (:project_id, :run_date, :check_name, :passed, :value, :threshold)
                        """),
                        {
                            "project_id": self.project_id,
                            "run_date": run_date,
                            "check_name": r.check_name,
                            "passed": r.passed,
                            "value": r.value,
                            "threshold": r.threshold,
                        },
                    )
                conn.commit()
        except Exception as e:
            logger.warning(
                f"[{self.project_id}] Failed to log monitoring results: {e}"
            )
