"""Airflow DAG factory — auto-generates one DAG per active project.

Reads all YAML configs from project_registry/projects/,
creates an independent DAG for each active project with its own schedule.

Phase 2: Implement after the inference pipeline works manually.
"""

from __future__ import annotations

# TODO (Phase 2): Implement make_dag(config: ProjectConfig) -> DAG
#   - Create DAG with dag_id=f"inference_{config.project_id}"
#   - Set schedule_interval from config.schedule.cron
#   - Set default_args (retries, timeout) from config.schedule
#   - Single PythonOperator that runs InferencePipeline(config).run(run_date)

# TODO (Phase 2): Auto-discover loop
#   - for config in ProjectConfig.load_all_active():
#   -     globals()[f"dag_{config.project_id}"] = make_dag(config)
