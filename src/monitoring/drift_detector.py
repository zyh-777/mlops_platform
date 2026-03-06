"""PSI-based drift detection.

Population Stability Index (PSI) measures how much a distribution
has shifted from a reference. Used to detect:
    - Feature drift (input data changed)
    - Prediction drift (model behavior changed)

PSI interpretation:
    < 0.10  — No significant shift
    0.10-0.25 — Moderate shift (investigate)
    > 0.25  — Significant shift (alert)

Usage:
    detector = DriftDetector(config)
    results = detector.check(current_df, reference_df)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.project_config import ProjectConfig
from src.logger.logger import get_logger

logger = get_logger("drift_detector")

DEFAULT_BINS = 10
PSI_EPSILON = 1e-4  # Avoid log(0) in PSI formula


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = DEFAULT_BINS,
) -> float:
    """Computes the Population Stability Index between two distributions.

    PSI = SUM( (current_pct - reference_pct) * ln(current_pct / reference_pct) )

    Args:
        reference: Reference distribution (e.g., last 30 days).
        current: Current distribution (e.g., today's data).
        bins: Number of bins for the histogram.

    Returns:
        PSI value (0 = identical distributions).
    """
    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Create bins from the reference distribution
    bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Deduplicate edges to avoid empty bins
    bin_edges = np.unique(bin_edges)

    # Compute proportions for each bin
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    ref_pct = ref_counts / len(reference) + PSI_EPSILON
    cur_pct = cur_counts / len(current) + PSI_EPSILON

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


@dataclass
class DriftResult:
    """Result of a drift check for a single column."""
    column: str
    psi: float
    severity: str  # "none", "warn", "critical"
    message: str


class DriftDetector:
    """Detects feature and prediction drift using PSI.

    Compares current run data against a reference distribution.
    Thresholds come from config.monitoring.drift.

    Args:
        config: ProjectConfig with drift detection settings.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.project_id = config.project_id
        self.drift_config = config.monitoring.drift

        self.enabled = self.drift_config.get("enabled", False)
        self.psi_warn = float(self.drift_config.get("psi_warn", 0.10))
        self.psi_critical = float(self.drift_config.get("psi_critical", 0.25))

    def check(
        self,
        current_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> list[DriftResult]:
        """Runs PSI drift detection on specified columns.

        Args:
            current_df: Today's data (features or predictions).
            reference_df: Reference data (e.g., last 30 days).
            columns: Columns to check. Defaults to all feature columns.

        Returns:
            List of DriftResult objects (one per column).
        """
        if not self.enabled:
            logger.info(f"[{self.project_id}] Drift detection is disabled")
            return []

        if columns is None:
            columns = self.config.schema.feature_columns

        results: list[DriftResult] = []

        for col in columns:
            if col not in current_df.columns or col not in reference_df.columns:
                continue

            # Only compute PSI on numeric columns
            if not pd.api.types.is_numeric_dtype(current_df[col]):
                continue

            psi = compute_psi(
                reference_df[col].values,
                current_df[col].values,
            )

            severity = self._classify(psi)
            results.append(DriftResult(
                column=col,
                psi=round(psi, 4),
                severity=severity,
                message=f"PSI={psi:.4f} ({severity})",
            ))

            if severity != "none":
                logger.warning(
                    f"[{self.project_id}] Drift detected in '{col}': "
                    f"PSI={psi:.4f} ({severity})"
                )

        if results:
            drifted = sum(1 for r in results if r.severity != "none")
            logger.info(
                f"[{self.project_id}] Drift check: {len(results)} columns, "
                f"{drifted} with drift"
            )

        return results

    def _classify(self, psi: float) -> str:
        """Classifies PSI value into severity tier."""
        if psi >= self.psi_critical:
            return "critical"
        elif psi >= self.psi_warn:
            return "warn"
        return "none"
