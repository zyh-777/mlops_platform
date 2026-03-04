"""Inference pipeline — the core engine of the platform.

One class, one run() method, works for ALL projects.
The only thing that varies is the ProjectConfig passed in.

Pipeline steps:
    1. Load model from MLflow (via pyfunc)
    2. Fetch input data (via DataConnector)
    3. Validate input schema
    4. Run model.predict() on feature columns
    5. Assemble output (id_columns + predictions + run_date)
    6. Validate output schema
    7. Save results to project-specific DB table
    8. Run monitoring checks
    9. Log run status to platform.run_log

Phase 1: Steps 1-7 + 9 (basic pipeline, no monitoring).
Phase 2: Add step 8 (monitoring).
"""

from __future__ import annotations

# TODO: Implement InferencePipeline class
#   - __init__(self, config: ProjectConfig)
#   - run(self, run_date: str) -> None
#
# The run() method orchestrates all steps in sequence.
# Each step should be a separate private method for clarity.
# Wrap the entire run in try/except to log failures to run_log.
