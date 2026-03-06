"""Airflow DAG factory — auto-generates one DAG per active project.

Reads all YAML configs from project_registry/projects/,
creates an independent DAG for each active project with its own schedule.

How it works:
    1. On Airflow scheduler startup, this file is parsed.
    2. load_all_active() scans the registry for active YAMLs.
    3. For each config, make_dag() creates a DAG with one PythonOperator.
    4. The DAG is registered in globals() so Airflow discovers it.

Adding a new project = dropping a YAML file. No code changes here.

Usage:
    Place this file (or symlink it) in your Airflow dags_folder.
    Airflow will auto-discover all generated DAGs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.core.project_config import ProjectConfig
from src.inference.pipeline import InferencePipeline
from src.logger.logger import get_logger

logger = get_logger("dag_factory")


def _run_inference(project_id: str, **kwargs) -> None:
    """Airflow task callable — loads config and runs the pipeline.

    Loads the config fresh each run (not at DAG parse time) so that
    YAML changes take effect without restarting the scheduler.

    Args:
        project_id: Used to locate the YAML file.
        **kwargs: Airflow context (contains execution_date, etc.).
    """
    from src.core.project_config import PROJECT_ROOT

    yaml_path = PROJECT_ROOT / "project_registry" / "projects" / f"{project_id}.yaml"
    config = ProjectConfig.from_yaml(yaml_path)

    # Airflow provides logical_date (or execution_date in older versions)
    run_date = kwargs.get("ds")  # ds = YYYY-MM-DD string from Airflow
    if not run_date:
        run_date = datetime.now().strftime("%Y-%m-%d")

    pipeline = InferencePipeline(config)
    pipeline.run(run_date)


def make_dag(config: ProjectConfig) -> DAG:
    """Creates an Airflow DAG for a single project.

    Each project gets its own independent DAG with:
        - A unique dag_id based on project_id
        - Its own schedule from config.schedule.cron
        - Retry settings from config.schedule
        - A single task that runs the inference pipeline

    Args:
        config: ProjectConfig loaded from YAML.

    Returns:
        An Airflow DAG instance.
    """
    default_args = {
        "owner": config.owner or "mlops-platform",
        "retries": config.schedule.retries,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=config.schedule.timeout_minutes),
    }

    dag = DAG(
        dag_id=f"inference_{config.project_id}",
        default_args=default_args,
        description=f"Daily inference for {config.display_name or config.project_id}",
        schedule_interval=config.schedule.cron,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["inference", config.owner or "unowned"],
    )

    PythonOperator(
        task_id="run_inference",
        python_callable=_run_inference,
        op_kwargs={"project_id": config.project_id},
        dag=dag,
    )

    return dag


# ═══════════════════════════════════════════════════════════
# AUTO-DISCOVER: Generate one DAG per active project
# ═══════════════════════════════════════════════════════════
# Airflow scans this file's globals() for DAG objects.
# We load all active configs and create a DAG for each one.

try:
    for _config in ProjectConfig.load_all_active():
        _dag_var = f"dag_{_config.project_id}"
        globals()[_dag_var] = make_dag(_config)
        logger.info(f"Created DAG: inference_{_config.project_id}")
except Exception as e:
    # Don't let a bad YAML crash the entire scheduler
    logger.error(f"DAG factory failed: {e}")
