# mlops-platform — Project Brief for Claude

> **Last updated:** 2026-03-04
> **Status:** REDESIGN IN PROGRESS. Migrating from single-project PD pipeline to multi-project production platform. This file describes the TARGET architecture.
> **Previous:** This project was originally a single PD model pipeline (LightGBM, China economy). We are redesigning it to support multiple projects/teams running in parallel.

This file is auto-read by Claude in VS Code (Claude Code extension). It provides persistent context about the project so you don't have to re-explain the architecture every time.

---

## What This Project Is

A **production inference platform** that runs multiple ML models daily for different teams. The platform does NOT train models — teams train at their own sites and register models in MLflow. The platform handles:

1. **Inference Pipeline** — loads registered model, fetches input data, validates, predicts, saves results. Same engine for every project, configured via YAML.
2. **Monitoring** — per-project drift detection, business rule checks, alerting. Catches data pipeline failures and model degradation.
3. **Evaluation** — config-driven metrics computation. Teams declare which metrics they want in YAML. Used for candidate validation and periodic quality monitoring.
4. **Model Lifecycle** — validates candidate models before promotion, supports instant rollback via MLflow aliases.

**Key design principle:** A new team goes from "we have a trained model" to "it runs daily in production" by providing a pyfunc wrapper (~10-25 lines) + a YAML config (~50 lines). Zero platform code changes.

## What This Project Is NOT

- **Not a training platform.** Teams own their training code, feature engineering, data pipelines, and model selection. We don't touch any of that.
- **Not model-specific.** The platform never imports LightGBM, sklearn, XGBoost, or any ML framework. It only calls `mlflow.pyfunc.load_model()` and `.predict()`.
- **Not a feature store.** Teams provide their own features via whatever data source they use. The platform just reads what the YAML config declares.

---

## Architecture Overview

### How Projects Onboard

```
Team trains model (their site, their code, any framework)
  → Team wraps model as mlflow.pyfunc.PythonModel (~10-25 lines)
  → Team registers model in shared MLflow (one command)
  → Team fills out project YAML config (~50 lines)
  → Platform validates: model loads? schema matches? predict works?
  → YAML dropped in project_registry/projects/ → Airflow auto-discovers
  → Model runs daily on schedule. Done.
```

### How Daily Inference Works

```
For EACH active project (parallel, independent DAGs):
  1. Load model         → mlflow.pyfunc.load_model("models:/{name}/Production")
  2. Fetch data         → DataConnector reads from source declared in YAML
  3. Validate input     → Check columns, dtypes, row count against schema
  4. Predict            → model.predict(features_dataframe)
  5. Validate output    → Check output columns, no NaN, values in range
  6. Save results       → Write to project-specific DB table
  7. Monitor            → Drift, business rules, compare with yesterday
  8. Log run            → Record status, row count, duration to platform.run_log
```

### How Model Updates Work

```
Team retrains → registers new version in MLflow → tags as "Candidate"
  → Platform validates candidate against real data (automated)
  → If evaluation.thresholds configured: metrics must pass
  → If passes: team promotes by moving "Production" alias (in MLflow UI)
  → Next daily run automatically loads new version
  → If problems: move alias back to old version (instant rollback)
```

---

## Project Structure

```
mlops-platform/
├── CLAUDE.md                          ← You are here
├── .env                               ← Environment config (DB creds, MLflow URI)
├── Dockerfile
├── docker-compose.yaml                ← Airflow + Postgres
├── requirements.txt
│
├── project_registry/                  ← WHERE TEAMS ONBOARD
│   └── projects/
│       ├── _example_fraud_detection.yaml
│       ├── credit_risk_cn.yaml        ← Each YAML = one production model
│       ├── fraud_detection.yaml       ← Could be any model objective
│       └── revenue_forecast.yaml      ← Teams define their own schema
│
├── src/
│   ├── dags/
│   │   └── project_dag_factory.py     ← Reads registry → creates Airflow DAGs
│   │
│   ├── core/                          ← Platform backbone
│   │   ├── project_config.py          ← YAML loader + validator (dataclass)
│   │   ├── data_connector.py          ← Pluggable data sources (MySQL, S3, API)
│   │   └── model_loader.py            ← mlflow.pyfunc.load_model wrapper
│   │
│   ├── inference/                     ← The pipeline engine
│   │   ├── pipeline.py                ← Orchestrator: fetch→validate→predict→save
│   │   └── validators.py             ← Input/output schema checks
│   │
│   ├── evaluation/                    ← Config-driven metric computation
│   │   └── metrics.py                 ← Metric registry + Evaluator class
│   │
│   ├── monitoring/                    ← Per-project health checks
│   │   ├── monitor.py                 ← Row count, null checks, drift
│   │   ├── drift_detector.py          ← PSI computation
│   │   └── alerter.py                 ← Slack/email notifications
│   │
│   ├── database/                      ← Storage layer
│   │   ├── db_connection.py           ← Connection pool (SQLAlchemy engine)
│   │   ├── schema_manager.py          ← Auto-create project tables from YAML
│   │   └── result_store.py            ← Unified save with dedup logic
│   │
│   ├── configs/
│   │   ├── db_config.py               ← MySQL connection config from .env
│   │   └── mlflow_config.py           ← MLflow URI + S3/MinIO setup
│   │
│   ├── logger/
│   │   └── logger.py                  ← Structured logging utility
│   │
│   └── dashboard/
│       └── streamlit_app.py           ← Multi-project monitoring UI
│
├── templates/                         ← GIVE THESE TO ONBOARDING TEAMS
│   ├── project_config_template.yaml   ← Blank YAML for teams to fill out
│   ├── pyfunc_wrapper_template.py     ← Example pyfunc wrapper (3 FE patterns)
│   └── onboarding_guide.md            ← Step-by-step instructions for teams
│
├── tools/                             ← Platform admin utilities
│   ├── validate_project.py            ← Validate a project YAML + model
│   ├── dry_run.py                     ← Test inference without saving to DB
│   └── backfill.py                    ← Re-run inference for past dates
│
└── tests/
    ├── test_pipeline.py
    ├── test_connectors.py
    └── test_validators.py
```

---

## Architecture Rules (MUST follow)

### Rule 1: The platform NEVER imports ML frameworks

The platform interacts with models exclusively through `mlflow.pyfunc`. No LightGBM, no sklearn, no XGBoost, no PyTorch imports anywhere in platform code. The pyfunc wrapper (written by the team, stored inside the MLflow artifact) handles all framework-specific logic.

```python
# ❌ NEVER do this anywhere in platform code
import lightgbm
from sklearn.ensemble import RandomForestClassifier
import torch

# ❌ NEVER do this — framework-specific model loading
model = mlflow.lightgbm.load_model(uri)
model = mlflow.sklearn.load_model(uri)

# ✅ ALWAYS do this — generic, works for ANY model
model = mlflow.pyfunc.load_model(uri)
predictions = model.predict(dataframe)
```

### Rule 2: All project-specific config lives in YAML, not in code

Never hardcode project names, model names, table names, feature lists, schedules, or thresholds in Python code. Everything project-specific comes from the YAML config.

```python
# ❌ WRONG — project-specific values in code
def run_inference():
    model = load_model("SomeModel_TeamA")
    data = query("SELECT ... FROM features WHERE team_id = 2")
    save_to_table("results.team_a_output")

# ✅ CORRECT — everything from config
def run_inference(config: ProjectConfig):
    model = load_model(config.model.mlflow_name)
    data = connector.fetch(config.input, run_date)
    save_to_table(config.output.target_table)
```

### Rule 3: One pipeline engine, many configs

The `InferencePipeline` class runs the exact same code path for every project. The only thing that varies is the `ProjectConfig` object passed to it. Do NOT create separate pipeline classes or files for different projects.

```python
# ❌ WRONG — project-specific pipelines
class TeamAPipeline: ...
class TeamBPipeline: ...

# ✅ CORRECT — one engine, config-driven
class InferencePipeline:
    def __init__(self, config: ProjectConfig): ...
    def run(self, run_date: str): ...

# Each project uses the same class:
for config in ProjectConfig.load_all_active():
    InferencePipeline(config).run(today)
```

### Rule 4: Project isolation — failures don't cascade

Each project runs as an independent Airflow DAG. If one project fails, others continue unaffected. Database tables are per-project. MLflow experiments are per-project.

### Rule 5: No bare except blocks

```python
# ❌ NEVER
try:
    something()
except:
    pass

# ✅ Catch specific exceptions, log with context, handle clearly
try:
    predictions = model.predict(features)
except Exception as e:
    logger.error(f"[{config.project_id}] Prediction failed: {e}")
    raise
```

### Rule 6: No hardcoded paths, IPs, or magic numbers

Everything environment-specific goes in `.env`. Project-specific values go in the project YAML.

### Rule 7: Database operations use context managers

```python
# ❌ WRONG
conn = engine.connect()
conn.execute(...)
conn.close()

# ✅ CORRECT
with engine.connect() as conn:
    conn.execute(...)
```

---

## How the Pyfunc Wrapper Works (important context)

The pyfunc wrapper is the "contract" between a team's model and this platform. Understanding it is essential for debugging issues.

**What it is:** A thin Python class with two methods:
- `load_context()` — called once when model is loaded. Sets up the model, loads weights, preprocessor, feature config.
- `predict()` — called each time the platform runs inference. Takes a DataFrame, returns a DataFrame.

**Where it lives:** Inside the MLflow artifact, NOT in platform code. When a team registers their model, the wrapper class gets serialized and stored alongside the model weights and pinned dependencies.

**Three FE patterns** — teams choose based on their setup:

- **Pattern A: Features arrive ready.** Data engineers compute features upstream. YAML query selects them directly. Wrapper just enforces column order and predicts (~10 lines). Simplest and most common.
- **Pattern B: Wrapper computes designed features.** Raw data arrives, wrapper applies stateless formulas (ratios, logs, indicators). No fitted state to save — just deterministic logic (~20 lines).
- **Pattern C: Wrapper handles fitted preprocessing.** Model needs fitted transforms (StandardScaler, LabelEncoder) that learned values from training data. These are bundled as artifact files (.pkl) alongside the model weights (~25 lines).

Patterns B and C can be combined (designed formulas + fitted scaler on results).

See `templates/pyfunc_wrapper_template.py` for concrete examples of all three patterns.

**Why it matters for consistency:** The wrapper bundles:
1. The exact library versions (e.g., `lightgbm==4.1.0`) — no version skew
2. The feature order used during training — no column order bugs
3. Any preprocessing logic (fitted or designed) — no training-serving skew

**When something goes wrong:** If a model produces bad predictions, check:
1. Is the model version correct? (`mlflow.pyfunc.load_model` → check version number)
2. Does the input schema match what the wrapper expects? (column names, dtypes)
3. Is the wrapper's preprocessor handling edge cases? (NaN, new categories, extreme values)

---

## Database Design

### Layer 1: Platform tables (fixed schema, always exist)

These track operational metadata. They know nothing about what any model does — only whether it ran, succeeded, and passed health checks.

```sql
platform.project_registry (
    project_id, display_name, owner, status, created_at, last_run_at
)
platform.run_log (
    project_id, run_date, status, row_count, duration_sec, error_message, created_at
)
platform.monitoring_log (
    project_id, run_date, check_name, passed, value, threshold, created_at
)
```

### Layer 2: Result tables (team-defined schema, auto-created from YAML)

Each team declares their own table name and columns in their project YAML. The platform auto-creates the table on first run and writes to it. The platform has NO opinion on what the columns mean — it just saves whatever the model outputs.

```yaml
# Example: a credit risk team
output:
  target_table: results.pd_daily_cn
  columns: [comp_id, run_date, pd_1m, poe_1m]

# Example: a fraud detection team
output:
  target_table: results.fraud_scores
  columns: [transaction_id, run_date, fraud_probability, risk_tier]

# Example: a revenue forecasting team
output:
  target_table: results.revenue_forecast
  columns: [business_unit, forecast_date, revenue_30d, confidence]
```

**Important:** The platform code that creates and writes to these tables is completely generic. It reads table name and column names from config. It never hardcodes or assumes any specific model objective, output column names, or domain semantics.

---

## Evaluation (Config-Driven)

Evaluation is handled by a single module (`src/evaluation/metrics.py`) with a metric registry pattern. No separate file per metric — one registry dict maps metric names to functions.

### How it works

Teams declare which metrics they want in their YAML:

```yaml
evaluation:
  metrics: [auc_roc, f1_macro, precision_macro, ks_statistic]
  thresholds:
    auc_roc: 0.80
    f1_macro: 0.55
```

The platform's `Evaluator` class reads this config and computes only the declared metrics:

```python
evaluator = Evaluator(config.evaluation)
results = evaluator.compute(y_true, y_pred, y_proba)
# → {"auc_roc": 0.94, "f1_macro": 0.82, "precision_macro": 0.88, "ks_statistic": 0.71}

checks = evaluator.check_thresholds(results)
# → {"auc_roc": {"value": 0.94, "threshold": 0.80, "passed": True}, ...}
```

### Available metrics

Classification: `auc_roc`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `accuracy`, `log_loss`, `ks_statistic`, `gini`
Regression: `mse`, `rmse`, `r2`, `mae`

### Adding new metrics

Add a function to `METRIC_REGISTRY` in `src/evaluation/metrics.py`. Signature: `def my_metric(y_true, y_pred, y_proba) -> float`. Teams can then use it in their YAML immediately.

### When evaluation runs

- **Candidate validation** (`tools/validate_project.py --candidate`): computes metrics on recent data, checks against thresholds. Used to decide whether a retrained model is good enough to promote.
- **Periodic quality monitoring** (Phase 3): when ground truth becomes available (often weeks after prediction), computes metrics and logs to `platform.monitoring_log`. Configured via `evaluation.ground_truth` in YAML.

---

## Monitoring Strategy

### Three levels of checks (run after each inference)

**Level 1 — Business rules (catch data pipeline failures):**
- Row count changed by >N% vs yesterday (threshold from YAML)
- Any predictions are NaN or outside declared valid range
- These trigger immediate alerts

**Level 2 — Feature drift using PSI:**
- Compute PSI per feature daily against reference distribution
- Log to `platform.monitoring_log`. Show on Streamlit dashboard
- PSI thresholds configurable per project in YAML

**Level 3 — Prediction distribution check:**
- Compare today's prediction distribution against recent history
- Only alert if BOTH feature drift AND prediction shift detected

### Alerting tiers

- **Tier 1:** Log only — all metrics written to DB, visible on dashboard
- **Tier 2:** Warning — sustained drift for 3+ of last 5 days → Slack
- **Tier 3:** Critical — any Level 1 violation or extreme PSI → immediate Slack

---

## Model Lifecycle

### MLflow aliases control what's in production

```
models:/{project_model_name}
  ├── Version 1  (archived)
  ├── Version 2  (archived)
  ├── Version 3  (alias: "Production")  ← currently serving
  └── Version 4  (alias: "Candidate")   ← under validation
```

YAML config says `version: Production` (an alias, not a number). Platform always loads whichever version has that alias.

### Promotion workflow

1. Team retrains and registers new version → MLflow auto-assigns next version number
2. Team tags it as "Candidate"
3. Platform runs `tools/validate_project.py --candidate` — tests against real data, checks evaluation thresholds
4. If validation passes → team moves "Production" alias to new version
5. Next daily run automatically uses the new model
6. Rollback: move "Production" alias back to old version (instant, no deployment)

### Validation checks for candidates

- Can the model load without errors?
- Does its output match the declared schema?
- Are predictions non-null and in valid range?
- If evaluation thresholds configured: do metrics pass? (e.g., AUC > 0.80)
- How different are predictions from current production model? (PSI)

---

## CI/CD (Deferred)

CI/CD is not needed for Phase 1 or Phase 2. The thing that changes most often is project configs and models, not platform code.

**What we have instead:** `tools/validate_project.py` acts as validation gate for new projects and model updates. This is more useful than code-level CI/CD at this stage.

**When to add CI/CD:**
- After Phase 2 (platform code stabilizes, tests exist)
- GitHub Actions: run pytest on push, lint with ruff
- Optional: Git hook on `project_registry/` — auto-run validation when a new YAML is pushed

**What we will NOT automate:** Model promotion to Production alias. This is intentionally a manual human decision for safety.

---

## Responsibility Boundaries

| Responsibility | Owner | NOT this platform |
|---|---|---|
| Upstream data pipelines (feature tables exist, are fresh) | Data Engineers | ✅ |
| Model training, feature selection, evaluation | Model Teams (Data Scientists) | ✅ |
| Pyfunc wrapper, MLflow registration | Model Teams | ✅ |
| YAML config for their project | Model Teams | ✅ |
| Inference orchestration (fetch → predict → save) | **This platform** | — |
| Schema validation, monitoring, alerting | **This platform** | — |
| Candidate validation before promotion | **This platform** | — |
| Evaluation metrics computation | **This platform** | — |
| Run logging, audit trail | **This platform** | — |
| Database infra (MySQL server, backups) | DevOps / Infra | ✅ |
| MLflow server, MinIO storage | DevOps / Infra | ✅ |

If a team's YAML query references a table that doesn't exist, that's a "talk to your data engineer" problem. The validation tool will catch it and tell the team.

---

## Build Phases

### Phase 1 — Core Pipeline (current focus)
- `core/project_config.py` — YAML loader + validation
- `core/data_connector.py` — MySQL connector (start with one source)
- `core/model_loader.py` — mlflow.pyfunc wrapper
- `inference/pipeline.py` — the 8-step engine
- `inference/validators.py` — schema validation
- `evaluation/metrics.py` — config-driven metric registry + evaluator
- `database/result_store.py` — save results
- Goal: one project runs end-to-end via `python -m inference.pipeline {project_id}`

### Phase 2 — Automation
- `dags/project_dag_factory.py` — auto-generate Airflow DAGs from registry
- `monitoring/monitor.py` — Level 1 business rules
- `database/schema_manager.py` — auto-create tables
- `tools/validate_project.py` — validate YAML + model + evaluation config
- `tools/dry_run.py` — test without saving
- Goal: projects run daily unattended, failures are caught

### Phase 3 — Lifecycle & Observability
- `monitoring/drift_detector.py` — PSI computation
- `monitoring/alerter.py` — Slack integration
- `dashboard/streamlit_app.py` — multi-project UI
- `tools/backfill.py` — re-run past dates
- Candidate validation with evaluation thresholds
- Periodic quality monitoring (when ground truth available)
- Goal: teams self-serve end-to-end, full observability

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Model serving | MLflow (pyfunc) | Framework-agnostic model loading and prediction |
| Model registry | MLflow | Version management, aliases (Production/Candidate) |
| Artifact storage | MinIO (S3-compatible) | Stores model artifacts (weights, wrapper, dependencies) |
| Orchestration | Airflow | Schedules independent DAG per project |
| Platform DB | MySQL | Run logs, monitoring logs, project registry |
| Results DB | MySQL | Per-project prediction tables |
| Data sources | Pluggable | MySQL, S3, API — configured per project in YAML |
| Evaluation | sklearn.metrics (wrapped) | Config-driven: teams declare metrics in YAML, platform computes |
| Drift monitoring | PSI (custom) | Feature and prediction distribution drift detection |
| Alerting | Slack webhook | Tiered alerts: critical, warning, info |
| Dashboard | Streamlit | Multi-project monitoring and predictions |
| Containerization | Docker Compose | Runs Airflow + Postgres |

---

## Coding Conventions

### Python style
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- f-strings for string formatting
- `pathlib.Path` for file paths
- `from __future__ import annotations` at top of files

### Naming
- Files: snake_case (`project_config.py`)
- Classes: PascalCase (`InferencePipeline`, `ProjectConfig`)
- Functions/methods: snake_case (`run_inference`, `validate_schema`)
- Constants: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT_SEC`)

### Imports
- Standard library → third-party → local
- No wildcard imports except in `__init__.py`
- Absolute imports preferred

### Logging
- Use the project's `get_logger()` from `src/logger/`
- Always include `project_id` in log messages for multi-project context
- Levels: INFO for normal flow, WARNING for recoverable issues, ERROR for failures

```python
logger.info(f"[{config.project_id}] Loaded {len(data)} rows")
logger.error(f"[{config.project_id}] Schema validation failed: missing {missing_cols}")
```

---

## Common Tasks and How to Approach Them

### "Onboard a new project"
1. Team wraps model as mlflow.pyfunc and registers in MLflow
2. Team fills out YAML from `templates/project_config_template.yaml`
3. Run `python -m tools.validate_project new_project` to verify
4. Drop YAML in `project_registry/projects/`
5. Done. Airflow auto-discovers it.

### "A project's predictions look wrong"
1. Check `platform.run_log` — did the run succeed?
2. Check `platform.monitoring_log` — any alerts?
3. Run `python -m tools.dry_run project_id --date 2025-01-15` to reproduce
4. Check model version — is it the expected one? Did someone promote a bad candidate?
5. If model issue → team investigates. If data issue → check data source.

### "Add a new data source type (e.g., S3)"
1. Create new class in `core/data_connector.py` implementing `DataConnector`
2. Register it in `DataConnectorFactory._registry`
3. Done. Projects can now use `source: s3` in their YAML.

### "Add a new evaluation metric"
1. Write a function in `src/evaluation/metrics.py`: `def _my_metric(y_true, y_pred, y_proba) -> float`
2. Add it to `METRIC_REGISTRY`: `"my_metric": _my_metric`
3. Done. Teams can now list `my_metric` in their YAML `evaluation.metrics`.

### "Team wants to retrain and update their model"
1. Team retrains and registers new version in MLflow
2. Team tags new version as "Candidate"
3. Run `python -m tools.validate_project project_id --candidate` to validate
4. If passes → team moves "Production" alias to new version in MLflow UI
5. Next daily run uses the new model automatically

### "Roll back a model"
1. In MLflow UI: move "Production" alias back to previous version
2. Done. Next run uses the old model. No code changes, no deployment.

---

## What NOT to Do

- **Don't import ML frameworks in platform code** — no lightgbm, sklearn, torch, etc. Only mlflow.pyfunc.
- **Don't create separate files per evaluation metric** — use the config-driven registry pattern in `src/evaluation/metrics.py`. One file, one registry dict, teams declare metrics in YAML.
- **Don't hardcode project-specific values in Python** — model names, table names, feature lists all come from YAML.
- **Don't create per-project pipeline classes** — one InferencePipeline, many configs.
- **Don't put training code in this repo** — training is the team's responsibility.
- **Don't put feature engineering in this repo** — teams bundle it inside their pyfunc wrapper or have it done upstream.
- **Don't add Kubernetes, feature stores, or complex infra** — keep it simple for now.
- **Don't set Tier 2 drift alert thresholds yet** — need 1-2 months of observed data for baseline.
- **Don't write SQL strings scattered in Python files** — keep queries in dedicated modules or templated in YAML.
- **Don't create "dev" vs "prod" duplicate files** — use config/env vars for environment differences.
- **Don't add CI/CD yet** — deferred until Phase 2 is stable and tests exist. See "CI/CD" section.

---

## Migration Notes (from old single-project codebase)

### Removed (team responsibility now)
- `feature_pipeline/` — teams handle feature engineering inside their pyfunc wrapper or upstream
- `steps_training/` — training happens at team sites
- `pipelines/training_pipeline.py` — not part of production platform
- `models/lgbm_model.py` — model-specific code stays with teams
- `configs/training_config.py` — training config stays with teams

### Kept and adapted
- `abstractions/model_contract.py` → conceptually replaced by mlflow.pyfunc interface (same idea, standard implementation)
- `abstractions/evaluation_abc.py` → replaced by config-driven `src/evaluation/metrics.py` (no more strategy pattern, single registry)
- `evaluation/` → collapsed from 12 files (one per metric) to 1 file with a metric registry dict. Teams declare metrics in YAML instead of code.
- `monitoring/` → enhanced with per-project config from YAML
- `database/` → extended with `schema_manager.py` and `result_store.py`
- `configs/db_config.py` → kept as shared infra config
- `configs/mlflow_config.py` → simplified (no per-model experiment routing needed)
- `logger/` → kept as shared logging infra

### Fundamentally redesigned
- `pipelines/inference_pipeline.py` → `inference/pipeline.py` (config-driven, project-agnostic)
- `pipelines/model_loader.py` → `core/model_loader.py` (generic pyfunc loading, no model registry map)
- `steps_inference/` → absorbed into `inference/pipeline.py` steps
- `dags/prediction_dag.py` → `dags/project_dag_factory.py` (dynamic DAG generation)
- `streamlit_app.py` → `dashboard/streamlit_app.py` (multi-project selector)
