"""Generic model loader via mlflow.pyfunc.

This is deliberately simple — the platform loads ANY model the same way.
No framework-specific imports. No model registry mapping.

Universal pattern: Thin Wrapper + Pyfunc Contract
    - mlflow.pyfunc is a standard interface: load_model() → .predict(df).
    - The platform never knows what's inside (LightGBM, XGBoost, PyTorch...).
    - The team's pyfunc wrapper handles all framework-specific logic.
    - Nothing connects to MLflow until load_model() is actually called.

Usage:
    model = load_model("fraud_detection_alpha", "Production")
    predictions = model.predict(features_df)
"""

from __future__ import annotations

import mlflow.pyfunc
from mlflow.pyfunc import PyFuncModel

from src.configs.mlflow_config import MlflowConfig
from src.logger.logger import get_logger

logger = get_logger("model_loader")


def load_model(mlflow_name: str, version: str = "Production") -> PyFuncModel:
    """Loads a model from MLflow by name and alias/version.

    Constructs a URI like "models:/fraud_detection_alpha/Production"
    and calls mlflow.pyfunc.load_model(). This works for ANY model
    framework — the pyfunc wrapper (written by the team) handles
    framework-specific loading internally.

    Args:
        mlflow_name: The registered model name in MLflow
                     (e.g., "fraud_detection_alpha").
        version: MLflow alias ("Production", "Candidate") or version number
                 ("1", "2"). Defaults to "Production".

    Returns:
        A PyFuncModel with a .predict(df) method.

    Raises:
        mlflow.exceptions.MlflowException: If the model is not found or
            MLflow server is unreachable.
    """
    # Ensure MLflow tracking URI is configured before loading
    MlflowConfig.setup()

    # MLflow v3: aliases use @ (e.g., models:/name@Production)
    # Numeric versions still use / (e.g., models:/name/1)
    if version.isdigit():
        uri = f"models:/{mlflow_name}/{version}"
    else:
        uri = f"models:/{mlflow_name}@{version}"
    logger.info(f"Loading model from {uri}")

    try:
        model = mlflow.pyfunc.load_model(uri)
        logger.info(f"Loaded model '{mlflow_name}' (version={version})")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{mlflow_name}' (version={version}): {e}")
        raise
