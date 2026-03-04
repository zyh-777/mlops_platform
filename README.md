# mlops-platform

A multi-project ML inference platform. Teams train models at their own sites, register them in MLflow, and this platform handles daily production inference, monitoring, and alerting.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your MySQL, MLflow, MinIO credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run inference for a project
python -m src.inference.pipeline <project_id> --date 2025-12-15
```

## Onboarding a New Model

See [`templates/onboarding_guide.md`](templates/onboarding_guide.md) for detailed instructions.

**TL;DR:** Write a pyfunc wrapper (~10-25 lines), register in MLflow, fill out a YAML config, run validation, drop YAML in `project_registry/projects/`. Done.

## Project Structure

```
mlops-platform/
├── project_registry/projects/   ← YAML configs (one per project)
├── src/
│   ├── core/                    ← Config loader, data connectors, model loader
│   ├── inference/               ← Pipeline engine + validators
│   ├── evaluation/              ← Config-driven metrics (single module)
│   ├── monitoring/              ← Drift detection, alerting
│   ├── database/                ← Connection, schema manager, result store
│   ├── configs/                 ← DB + MLflow config from .env
│   ├── logger/                  ← Structured logging
│   ├── dags/                    ← Airflow DAG factory
│   └── dashboard/               ← Streamlit monitoring UI
├── templates/                   ← Templates for onboarding teams
├── tools/                       ← validate_project, dry_run, backfill
└── tests/
```

## Airflow Deployment

```bash
docker compose up -d --build
# Access Airflow UI at http://localhost:8088
```

Each active project YAML auto-generates an independent Airflow DAG.

## Architecture

See [`CLAUDE.md`](CLAUDE.md) for detailed architecture documentation.
