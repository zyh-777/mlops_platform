"""Generic model loader via mlflow.pyfunc.

This is deliberately simple — the platform loads ANY model the same way.
No framework-specific imports. No model registry mapping.

Usage:
    model = load_model("fraud_detection_alpha", "Production")
    predictions = model.predict(features_df)
"""

from __future__ import annotations

# TODO: Implement load_model(mlflow_name: str, version: str) -> mlflow.pyfunc.PyFuncModel
#   - Construct URI: f"models:/{mlflow_name}/{version}"
#   - Call mlflow.pyfunc.load_model(uri)
#   - Log success with model name and version
#   - Handle MLflow connection errors with clear messages
