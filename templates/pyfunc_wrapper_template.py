"""
Pyfunc Wrapper Template
========================
Copy this file and adapt it for your model.

This wrapper standardizes your model so the platform can load and run it
without knowing anything about your framework (LightGBM, XGBoost, sklearn, etc.).

You need to implement two methods:
  - load_context(): load your model, preprocessor, feature config, etc.
  - predict(): take a DataFrame of features, return a DataFrame of predictions.

THREE PATTERNS depending on your feature engineering approach:

  Pattern A — "Features arrive ready"
    Data engineer computes features upstream in DB/data pipeline.
    YAML query selects ready-to-use features.
    Wrapper just enforces column order and predicts. (~10 lines)

  Pattern B — "Wrapper computes designed features from raw data"
    Raw data arrives (e.g., net_income, total_assets).
    Wrapper applies deterministic formulas (ratios, logs, indicators).
    No fitted state — just pure functions. (~20 lines)

  Pattern C — "Wrapper handles fitted preprocessing"
    Model needs fitted transforms (StandardScaler, LabelEncoder, etc.)
    that learned values from training data. Bundle them as artifacts. (~25 lines)

After implementing, register your model with:
  mlflow.pyfunc.log_model(
      artifact_path="model",
      python_model=YourModelWrapper(),
      artifacts={...},
      pip_requirements=[...],
      registered_model_name="your_project_id",
  )
"""

from __future__ import annotations

import json

import mlflow.pyfunc
import pandas as pd


# ═══════════════════════════════════════════════════════════
# PATTERN A — Features arrive ready (simplest)
# ═══════════════════════════════════════════════════════════
#
# When to use:
#   - Data engineers compute features upstream (in SQL, Spark, etc.)
#   - Your YAML query already returns model-ready features
#   - No transforms needed at prediction time
#
# Artifacts needed:
#   - model_file: trained model weights
#   - feature_config: feature names + order (JSON)
#
# Example YAML input.query:
#   SELECT comp_id, yyyy, mm,
#          dtd_level, dtd_trend, ni2ta_level, sigma
#   FROM features.daily_features
#   WHERE run_date = '{run_date}'

class ReadyFeaturesWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for models that receive pre-computed features."""

    def load_context(self, context):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=context.artifacts["model_file"])

        with open(context.artifacts["feature_config"]) as f:
            self.config = json.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        # Enforce column order (must match training)
        features = model_input[self.config["feature_order"]]

        # Predict
        raw = self.model.predict(features)

        return pd.DataFrame({
            "pd_1m": raw[:, 1],
            "poe_1m": raw[:, 2],
        })

# feature_config.json:
# {
#   "feature_order": ["dtd_level", "dtd_trend", "ni2ta_level", "sigma"],
#   "training_date": "2025-06-15",
#   "training_rows": 125000
# }


# ═══════════════════════════════════════════════════════════
# PATTERN B — Wrapper computes designed features from raw data
# ═══════════════════════════════════════════════════════════
#
# When to use:
#   - Team wants to own FE logic (not depend on data engineers)
#   - FE is stateless: ratios, log transforms, indicators, clipping
#   - No fitted state needed — given same input, always same output
#
# Artifacts needed:
#   - model_file: trained model weights
#   - feature_config: raw column names + feature formulas reference (JSON)
#
# Example YAML input.query:
#   SELECT comp_id, yyyy, mm,
#          net_income, total_assets, market_cap, total_debt, ownership_type
#   FROM raw_data.company_financials
#   WHERE run_date = '{run_date}'

class DesignedFEWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper that computes features via designed formulas."""

    def load_context(self, context):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=context.artifacts["model_file"])

        with open(context.artifacts["feature_config"]) as f:
            self.config = json.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        df = model_input.copy()

        # ── Designed FE: pure functions, no fitted state ──
        # Replace with your actual formulas
        df["ni2ta"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
        df["log_size"] = np.log(df["market_cap"].clip(lower=1))
        df["debt_ratio"] = df["total_debt"] / df["total_assets"].replace(0, np.nan)
        df["is_soe"] = (df["ownership_type"] == "state").astype(int)

        # Select model features in correct order
        features = df[self.config["feature_order"]]
        raw = self.model.predict(features)

        return pd.DataFrame({
            "pd_1m": raw[:, 1],
            "poe_1m": raw[:, 2],
        })

# feature_config.json:
# {
#   "feature_order": ["ni2ta", "log_size", "debt_ratio", "is_soe"],
#   "raw_columns": ["net_income", "total_assets", "market_cap",
#                    "total_debt", "ownership_type"],
#   "training_date": "2025-06-15"
# }


# ═══════════════════════════════════════════════════════════
# PATTERN C — Wrapper handles fitted preprocessing
# ═══════════════════════════════════════════════════════════
#
# When to use:
#   - Model needs fitted transforms (StandardScaler, MinMaxScaler)
#   - Model needs fitted encoders (LabelEncoder, OneHotEncoder)
#   - Model needs fitted imputers (fill NaN with training median)
#   - These objects LEARNED values from training data — must be reused exactly
#
# Artifacts needed:
#   - model_file: trained model weights
#   - feature_config: feature names + which columns to transform (JSON)
#   - preprocessor: fitted sklearn pipeline or scaler (.pkl file)
#
# Note: Pattern C can be combined with Pattern B. You might have
# BOTH designed formulas AND a fitted scaler on the results.

class FittedPreprocessingWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper for models that need fitted preprocessing."""

    def load_context(self, context):
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(context.artifacts["model_file"])

        import joblib
        self.preprocessor = joblib.load(context.artifacts["preprocessor"])

        with open(context.artifacts["feature_config"]) as f:
            self.config = json.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import xgboost as xgb

        # ① Enforce column order
        features = model_input[self.config["feature_order"]].copy()

        # ② Apply SAME fitted preprocessing as training
        scale_cols = self.config["scale_columns"]
        features[scale_cols] = self.preprocessor.transform(features[scale_cols])

        # ③ Predict
        dmatrix = xgb.DMatrix(features)
        scores = self.model.predict(dmatrix)

        return pd.DataFrame({"fraud_probability": scores})

# feature_config.json:
# {
#   "feature_order": ["amount", "merchant_risk", "hour_of_day",
#                      "is_international", "velocity_24h"],
#   "scale_columns": ["amount", "velocity_24h"],
#   "training_date": "2025-12-10"
# }


# ═══════════════════════════════════════════════════════════
# REGISTRATION — run this after training
# ═══════════════════════════════════════════════════════════
#
# import mlflow
# mlflow.set_tracking_uri("http://your-mlflow-server:5000")
#
# # Choose the right pattern:
#
# mlflow.pyfunc.log_model(
#     artifact_path="model",
#     python_model=ReadyFeaturesWrapper(),     # or DesignedFEWrapper()
#     artifacts={                               # or FittedPreprocessingWrapper()
#         "model_file": "/path/to/model",
#         "feature_config": "/path/to/feature_config.json",
#         # "preprocessor": "/path/to/scaler.pkl",  # only for Pattern C
#     },
#     pip_requirements=[
#         # Pin EXACT versions used during training:
#         "lightgbm==4.1.0",    # or "xgboost==2.0.3", etc.
#         "pandas==2.1.0",
#         "numpy==1.26.0",
#         # "scikit-learn==1.3.0",  # only if using fitted preprocessor
#     ],
#     registered_model_name="your_project_id",
# )
#
# # Tag first version as Production:
# from mlflow import MlflowClient
# client = MlflowClient()
# client.set_registered_model_alias("your_project_id", "Production", 1)
