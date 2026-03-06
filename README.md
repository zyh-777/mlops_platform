# mlops-platform

A multi-project ML inference platform. Teams train models at their own sites, register them in MLflow, and this platform handles daily production inference, monitoring, and alerting.

**The platform does NOT train models.** It loads any model via `mlflow.pyfunc`, runs predictions on scheduled data, validates results, saves to DB, and monitors for issues.

**Adding a new project = one YAML file + one pyfunc wrapper. Zero platform code changes.**

---

## Current Status

All platform code is implemented and tested. **49 tests passing.**

| Component | Status | Files |
|-----------|--------|-------|
| Config loader (YAML → dataclass) | Done | `src/core/project_config.py` |
| Data connector (MySQL) | Done | `src/core/data_connector.py` |
| Model loader (mlflow.pyfunc) | Done | `src/core/model_loader.py` |
| Inference pipeline (8-step engine) | Done | `src/inference/pipeline.py` |
| Input/output validators | Done | `src/inference/validators.py` |
| Evaluation metrics (config-driven) | Done | `src/evaluation/metrics.py` |
| Database (connection, schema, results) | Done | `src/database/` |
| Monitoring (row count, null, range) | Done | `src/monitoring/monitor.py` |
| Drift detection (PSI) | Done | `src/monitoring/drift_detector.py` |
| Alerting (Microsoft Teams) | Done | `src/monitoring/alerter.py` |
| Airflow DAG factory | Done | `src/dags/project_dag_factory.py` |
| Streamlit dashboard | Done | `src/dashboard/streamlit_app.py` |
| CLI tools (validate, dry_run, backfill) | Done | `tools/` |
| Tests | Done (49) | `tests/` |
| CI/CD | Not started | Future |

**Next step:** Connect to real infrastructure (MySQL + MLflow) and onboard the first real project.

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your MySQL and MLflow credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests (no database needed)
python -m pytest tests/ -v

# 4. Validate a project config
python -m tools.validate_project <project_id>

# 5. Dry run (runs pipeline, prints results, does NOT save to DB)
python -m tools.dry_run <project_id> --date 2026-03-05

# 6. Real run
python -m src.inference.pipeline <project_id> --date 2026-03-05
```

---

## How It Works

### The Daily Pipeline (8 steps)

```
For each active project (independent, parallel):

  1. Load model      → mlflow.pyfunc.load_model("models:/{name}/Production")
  2. Fetch data      → SQL query from YAML, with {run_date} substituted
  3. Validate input  → Check columns exist, no NULLs in features
  4. Predict         → model.predict(features_dataframe)
  5. Assemble output → id_columns + predictions + run_date
  6. Validate output → Check for NaN, values in declared range
  7. Save results    → Write to project-specific DB table (with dedup)
  8. Log run         → Record status, row count, duration to platform.run_log
```

All 8 steps run in `src/inference/pipeline.py` → `InferencePipeline.run()`.
If any step fails, the error is logged to `run_log` and the run stops.

### How Projects Onboard

```
Team trains model (any framework — LightGBM, XGBoost, PyTorch, etc.)
  → Team wraps model as mlflow.pyfunc.PythonModel (~10-25 lines)
  → Team registers model in shared MLflow
  → Team fills out project YAML config (~50 lines)
  → Admin runs: python -m tools.validate_project <project_id>
  → Admin runs: python -m tools.dry_run <project_id>
  → Drop YAML in project_registry/projects/
  → Airflow auto-discovers it → model runs daily. Done.
```

### How Model Updates Work

```
Team retrains → registers new version → tags as "Candidate"
  → Admin runs: python -m tools.validate_project <project_id> --candidate
  → If passes → team moves "Production" alias to new version (in MLflow UI)
  → Next daily run automatically loads new version
  → Rollback: move alias back to old version (instant, no code change)
```

---

## Project Structure

```
mlops-platform/
├── CLAUDE.md                          ← Detailed architecture doc (for AI + humans)
├── README.md                          ← You are here
├── .env.example                       ← Environment config template
├── requirements.txt                   ← Python dependencies
├── Dockerfile                         ← Container build
├── docker-compose.yaml                ← Airflow + Postgres
│
├── project_registry/                  ← WHERE TEAMS ONBOARD
│   └── projects/
│       └── _example_fraud_detection.yaml   ← Working example config
│
├── src/
│   ├── core/                          ← Platform backbone
│   │   ├── project_config.py          ← YAML → dataclass (foundation for everything)
│   │   ├── data_connector.py          ← ABC + MySQLConnector + Factory
│   │   └── model_loader.py            ← mlflow.pyfunc.load_model wrapper
│   │
│   ├── inference/                     ← The pipeline engine
│   │   ├── pipeline.py                ← InferencePipeline — the 8-step orchestrator
│   │   └── validators.py              ← Input/output schema + quality checks
│   │
│   ├── evaluation/                    ← Config-driven metrics
│   │   └── metrics.py                 ← METRIC_REGISTRY + Evaluator class
│   │
│   ├── monitoring/                    ← Post-inference health checks
│   │   ├── monitor.py                 ← Row count, null, range checks → monitoring_log
│   │   ├── drift_detector.py          ← PSI computation for feature/prediction drift
│   │   └── alerter.py                 ← Microsoft Teams webhook alerts
│   │
│   ├── database/                      ← Storage layer
│   │   ├── db_connection.py           ← SQLAlchemy engine singleton (connection pool)
│   │   ├── schema_manager.py          ← Auto-create platform + result tables
│   │   └── result_store.py            ← Save predictions with dedup + log runs
│   │
│   ├── configs/                       ← Infrastructure config from .env
│   │   ├── db_config.py               ← MySQL connection URL
│   │   └── mlflow_config.py           ← MLflow tracking URI
│   │
│   ├── logger/
│   │   └── logger.py                  ← Structured logging (always includes project_id)
│   │
│   ├── dags/
│   │   └── project_dag_factory.py     ← Auto-generates one Airflow DAG per active project
│   │
│   └── dashboard/
│       └── streamlit_app.py           ← Multi-project monitoring UI
│
├── templates/                         ← GIVE THESE TO ONBOARDING TEAMS
│   ├── project_config_template.yaml   ← Blank YAML for teams to fill out
│   ├── pyfunc_wrapper_template.py     ← Example pyfunc wrapper (3 patterns)
│   └── onboarding_guide.md            ← Step-by-step instructions
│
├── tools/                             ← Admin CLI utilities
│   ├── validate_project.py            ← Pre-flight check (YAML + model + data + predict)
│   ├── dry_run.py                     ← Test run without saving to DB
│   └── backfill.py                    ← Re-run inference for past date range
│
└── tests/                             ← 49 tests, all use mock data (no DB needed)
    ├── test_pipeline.py               ← ProjectConfig loading + pipeline orchestration
    ├── test_validators.py             ← Input/output validation logic
    ├── test_connectors.py             ← Connector factory + MySQL connector
    ├── test_evaluation.py             ← Metrics computation + thresholds
    └── test_monitoring.py             ← PSI drift + alerter
```

---

## Infrastructure Requirements

| Component | Required? | Purpose |
|-----------|-----------|---------|
| **MySQL** | Yes | Platform tables (run_log, monitoring_log) + result tables |
| **MLflow** | Yes | Model registry — loads models via `mlflow.pyfunc` |
| **MinIO / S3** | No (optional) | MLflow artifact storage. MLflow can use local filesystem instead. |
| **Airflow** | No (optional) | Automates daily scheduling. You can run the pipeline manually without it. |
| **Streamlit** | No (optional) | Dashboard UI. Everything works without it. |

### Minimal setup (for testing with real data):

```bash
# 1. MySQL — start locally or point to existing server
#    Edit .env with credentials

# 2. MLflow — start with local storage (no MinIO needed)
mlflow server --host 0.0.0.0 --port 5000

# 3. Run pipeline manually
python -m tools.dry_run <project_id> --date 2026-03-05
```

### Full production setup:

```bash
# Start Airflow (auto-discovers all project DAGs)
docker compose up -d --build
# Access Airflow UI at http://localhost:8088
```

---

## Key Concepts to Remember

### Everything is config-driven

- **One pipeline engine** (`InferencePipeline`) runs ALL projects. No per-project code.
- **Project-specific values** (model name, table, features, schedule) live in YAML files.
- **New project = new YAML file.** No Python changes needed.

### The platform never imports ML frameworks

```python
# The platform ONLY does this:
model = mlflow.pyfunc.load_model(uri)
predictions = model.predict(dataframe)

# It NEVER does this (no lightgbm, sklearn, torch, etc.):
import lightgbm
```

The team's pyfunc wrapper (bundled inside the MLflow artifact) handles all framework-specific logic.

### Each project is isolated

- Each project = separate Airflow DAG, separate DB table, separate monitoring.
- If one project fails, others continue unaffected.
- Each project's YAML declares its own schedule, thresholds, alert channel.

### Dedup protects against re-runs

- `write_mode: append` (default) — skips if data for that run_date already exists.
- `write_mode: replace_date` — deletes old rows, inserts new ones.
- Safe to re-run the same date without creating duplicates.

---

## CLI Tools Reference

```bash
# Validate a project config (6 checks: YAML, metrics, model, data, predict, output)
python -m tools.validate_project <project_id>
python -m tools.validate_project <project_id> --candidate  # test Candidate model

# Dry run — full pipeline, prints results, does NOT save
python -m tools.dry_run <project_id> --date 2026-03-05

# Backfill — re-run for a date range
python -m tools.backfill <project_id> --from 2026-01-01 --to 2026-01-31
python -m tools.backfill <project_id> --from 2026-01-01 --to 2026-01-31 --replace

# Run tests
python -m pytest tests/ -v
```

---

## Database Tables

### Platform tables (fixed schema, auto-created):

- `project_registry` — which projects are registered
- `run_log` — one row per pipeline run (status, row count, duration, error)
- `monitoring_log` — one row per health check (passed/failed, value, threshold)

### Result tables (per-project, auto-created from YAML):

Each project declares its own table name and columns in YAML.
The platform auto-creates it on first run.

```yaml
# Example from fraud_detection YAML:
output:
  target_table: results.fraud_scores
  columns: [txn_id, run_date, fraud_probability]
```

---

## Monitoring & Alerting

After each inference run, the platform checks:

1. **Row count change** — alerts if today's count differs >N% from yesterday
2. **Null outputs** — alerts if prediction columns have NaN values
3. **Prediction range** — alerts if values outside declared range (e.g., probability > 1.0)
4. **Feature drift (PSI)** — detects distribution shift vs reference period

Alerts go to **Microsoft Teams** via incoming webhook. Set `TEAMS_WEBHOOK_URL` in `.env`.
If not set, alerts are logged only (no external notifications).

---

## Troubleshooting

### "A project's predictions look wrong"
1. Check `run_log` — did the run succeed?
2. Check `monitoring_log` — any alerts?
3. Run `python -m tools.dry_run <project_id> --date <date>` to reproduce
4. Check model version in MLflow — is it the expected one?

### "Model load fails"
- Is MLflow running? Check `MLFLOW_TRACKING_URI` in `.env`
- Is the model registered with the correct name? (must match YAML `model.mlflow_name`)
- Does the "Production" alias exist on the model?

### "Data fetch returns 0 rows"
- Is the database reachable? Check connection config in YAML
- Does the query work for that date? Try running it manually in MySQL
- Is `{run_date}` being substituted correctly?

### "Rollback a bad model"
In MLflow UI: move the "Production" alias back to the previous version. Done.
Next run uses the old model. No code changes, no redeployment.

---

## Architecture Deep Dive

See [`CLAUDE.md`](CLAUDE.md) for:
- Detailed architecture rules
- Pyfunc wrapper patterns (A, B, C)
- Evaluation system design
- Monitoring strategy (3 levels, 3 tiers)
- Model lifecycle (candidate → production → rollback)
- Coding conventions
