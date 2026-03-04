# Onboarding Guide: How to Deploy Your Model on mlops-platform

## Overview

You have a trained model. You want it to run daily in production, score new data, and save results to a database. Here's how to get there in ~1 hour.

## What You Need to Provide

1. **A pyfunc wrapper** (~10-25 lines of Python) — standardizes your model
2. **Model registered in MLflow** (one command)
3. **A YAML config** (~50 lines) — tells the platform everything it needs

## Step 1: Write the Pyfunc Wrapper

Copy `templates/pyfunc_wrapper_template.py` and implement two methods:

- `load_context()` — load your model weights, preprocessor, feature config
- `predict()` — take a DataFrame, return a DataFrame

**Choose your pattern based on how your features work:**

| Pattern | When to use | Effort |
|---------|------------|--------|
| **A: Features arrive ready** | Data engineers compute features upstream in DB | ~10 lines |
| **B: Designed FE in wrapper** | You compute features via formulas (ratios, logs, indicators) — no fitted state | ~20 lines |
| **C: Fitted preprocessing** | You have fitted transforms (StandardScaler, LabelEncoder, etc.) | ~25 lines |

Patterns B and C can be combined. See the template for examples of all three.

**Key rules:**
- Enforce column order inside `predict()` from your feature config
- If you have fitted preprocessing (Pattern C), bundle the `.pkl` file as an artifact
- Pin exact dependency versions when registering
- Return a pandas DataFrame, not a numpy array

## Step 2: Register in MLflow

```python
import mlflow
mlflow.set_tracking_uri("http://mlflow-server:5000")

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=YourWrapper(),
    artifacts={"model_file": "path/to/model", ...},
    pip_requirements=["your-framework==x.y.z", "pandas==2.1.0"],
    registered_model_name="your_project_id",
)

# Tag as Production:
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("your_project_id", "Production", 1)
```

## Step 3: Fill Out the YAML Config

Copy `templates/project_config_template.yaml` to `project_registry/projects/your_project_id.yaml` and fill in:

- `model.mlflow_name` — must match `registered_model_name` from Step 2
- `input.query` — SQL query to fetch your input data (use `{run_date}` placeholder)
- `schema.feature_columns` — must match what your `predict()` method expects
- `schema.output_columns` — must match what your `predict()` method returns
- `output.target_table` — where results are saved
- `evaluation.metrics` — which metrics to compute when validating new model versions
- `evaluation.thresholds` — minimum acceptable values for candidate promotion

**Available evaluation metrics:**
- Classification: `auc_roc`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro`, `accuracy`, `log_loss`, `ks_statistic`, `gini`
- Regression: `mse`, `rmse`, `r2`, `mae`

## Step 4: Validate

```bash
python -m tools.validate_project your_project_id
```

This checks: YAML is valid, model loads, query works, prediction succeeds, output matches schema, evaluation metrics are recognized.

## Step 5: Deploy

Drop the YAML file in `project_registry/projects/`. Airflow auto-discovers it. Done.

## Updating Your Model

1. Retrain at your site, wrap in same pyfunc class
2. Register as new version: `mlflow.pyfunc.log_model(..., registered_model_name="your_project_id")`
3. Tag as Candidate: `client.set_registered_model_alias("your_project_id", "Candidate", 2)`
4. Validate: `python -m tools.validate_project your_project_id --candidate`
5. Promote: `client.set_registered_model_alias("your_project_id", "Production", 2)`
6. Rollback (if needed): move alias back to old version

## FAQ

**Q: What ML framework can I use?**
A: Anything. The pyfunc wrapper hides the framework from the platform.

**Q: Does the platform handle feature engineering?**
A: No. Either compute features upstream (Pattern A) or inside your pyfunc wrapper (Pattern B/C).

**Q: My features are designed formulas (ratios, logs) — do I need a scaler?**
A: No. Only Pattern C (fitted preprocessing) needs a saved scaler. If your FE is pure math, use Pattern A or B — no .pkl files needed.

**Q: What if my model needs a new feature column after retraining?**
A: Update your YAML (query + schema), re-run validation, then promote the new model.

**Q: Who maintains the input data tables?**
A: Your data engineers. The platform reads from whatever source your YAML declares.

**Q: How do I add a custom evaluation metric?**
A: Ask the platform team to add it to `METRIC_REGISTRY` in `src/evaluation/metrics.py`. Then use it in your YAML.
