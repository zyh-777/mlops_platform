"""Registers a dummy PD model in MLflow for testing.

This creates a pyfunc model that returns random PD scores.
Once registered, the real pipeline can load it via mlflow.pyfunc.load_model().

Usage:
    # 1. Start MLflow first:
    #    docker compose -f docker-compose.dev.yaml up -d
    #
    # 2. Register the dummy model:
    #    python -m tools.register_dummy_model
    #
    # 3. Check MLflow UI:
    #    open http://localhost:5000

What this does:
    1. Creates a PythonModel class (the pyfunc wrapper)
    2. Logs it to MLflow as an experiment run
    3. Registers it as "credit_risk_us" in the model registry
    4. Sets the "Production" alias so the pipeline can find it
"""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "credit_risk_us"


class DummyPDModel(mlflow.pyfunc.PythonModel):
    """A fake PD model that returns random probabilities.

    This is a placeholder to test the full pipeline end-to-end.
    Replace this with the real model wrapper when the model team provides it.

    The pyfunc contract:
        - load_context(): called once when model loads (setup)
        - predict(): called each time with a DataFrame, returns predictions
    """

    def load_context(self, context) -> None:
        """Called once when mlflow.pyfunc.load_model() is called.

        A real model would load weights and preprocessors here.
        Our dummy model has nothing to load.
        """
        pass

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Returns random PD scores between 0 and 1.

        A real model would:
            1. Preprocess features (if needed)
            2. Run model.predict(features)
            3. Return predictions as a DataFrame

        Args:
            context: MLflow context (unused here).
            model_input: DataFrame with feature columns.

        Returns:
            DataFrame with one column: pd_1y (probability of default, 1 year).
        """
        n = len(model_input)
        # Use a simple formula instead of pure random — more realistic
        # Higher sigma and lower DTD → higher PD (makes financial sense)
        np.random.seed(42)
        pd_scores = np.random.beta(a=2, b=5, size=n)  # Skewed toward low PD
        return pd.DataFrame({"pd_1y": pd_scores})


def main() -> None:
    """Registers the dummy model in MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
    print(f"Registering model: {MODEL_NAME}\n")

    # Create or get experiment
    experiment_name = "credit_risk_us_experiment"
    mlflow.set_experiment(experiment_name)

    # Log the model in a run
    with mlflow.start_run(run_name="dummy_model_v1") as run:
        print(f"Run ID: {run.info.run_id}")

        # Log some fake metrics (so MLflow UI has something to show)
        mlflow.log_metric("auc_roc", 0.85)
        mlflow.log_metric("ks_statistic", 0.45)
        mlflow.log_metric("gini", 0.70)
        mlflow.log_param("model_type", "dummy_beta_distribution")
        mlflow.log_param("description", "Placeholder model for pipeline testing")

        # Log the pyfunc model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DummyPDModel(),
            registered_model_name=MODEL_NAME,
        )
        print(f"Model URI: {model_info.model_uri}")

    # Set the "Production" alias on version 1
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "Production", version="1")

    print(f"\n{'='*50}")
    print(f"  Model registered: {MODEL_NAME}")
    print(f"  Version: 1")
    print(f"  Alias: Production")
    print(f"  MLflow UI: {MLFLOW_TRACKING_URI}")
    print(f"{'='*50}")
    print(f"\nYou can now run the pipeline:")
    print(f"  python -m tools.run_e2e")


if __name__ == "__main__":
    main()
