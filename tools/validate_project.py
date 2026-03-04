"""Validate a project YAML config and its model end-to-end.

Usage:
    python -m tools.validate_project <project_id>
    python -m tools.validate_project <project_id> --candidate

Checks:
    1. YAML is valid and all required fields present
    2. Evaluation metrics are valid and registered in METRIC_REGISTRY
    3. Model loads from MLflow (Production or Candidate alias)
    4. Data source is reachable and query returns expected columns
    5. Test prediction succeeds on sample data
    6. Output matches declared schema
    7. (--candidate) Run evaluation metrics, check thresholds, compare with production
"""

from __future__ import annotations

# TODO (Phase 1): Implement as CLI tool
#   - argparse: project_id (required), --candidate flag (optional)
#   - Load config from project_registry/projects/{project_id}.yaml
#   - Run each validation step, print results
#   - Exit 0 if all pass, exit 1 if any fail

# TODO (Phase 1): Validate evaluation config
#   - Check all metric names exist in METRIC_REGISTRY
#   - Check threshold metric names are subset of declared metrics
#   - Warn if thresholds defined but no ground_truth source configured

# TODO (Phase 2): Candidate validation with evaluation
#   - Load candidate model from MLflow
#   - Run predictions on recent data
#   - If ground_truth configured: compute metrics, check thresholds
#   - Compare candidate metrics vs production metrics
#   - Print summary with pass/fail status
