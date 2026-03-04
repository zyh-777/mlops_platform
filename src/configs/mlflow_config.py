"""MLflow configuration from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class MlflowConfig:
    """MLflow connection settings from .env."""

    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")

    @classmethod
    def setup(cls) -> None:
        """Sets MLflow environment variables for S3/MinIO access."""
        import mlflow

        mlflow.set_tracking_uri(cls.TRACKING_URI)
        if cls.S3_ENDPOINT_URL:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = cls.S3_ENDPOINT_URL
