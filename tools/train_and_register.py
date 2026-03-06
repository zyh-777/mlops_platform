"""Step 3: Train a LightGBM model and register in MLflow as pyfunc.

This simulates what a data science team does:
    1. Read feature data
    2. Generate synthetic default labels (since we don't have real labels)
    3. Train a LightGBM binary classifier
    4. Wrap it as mlflow.pyfunc.PythonModel (the "contract" with the platform)
    5. Register in MLflow and set the "Production" alias

The pyfunc wrapper is Pattern A (features arrive ready):
    - No feature engineering inside the wrapper
    - Just receives the 17 features and calls model.predict()
    - LightGBM handles NaN values natively

Prerequisites:
    Docker containers running: docker compose -f docker-compose.dev.yaml up -d

Usage:
    python -m tools.train_and_register
    python -m tools.train_and_register --train-size 100000  # use 100k rows for training
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "StkIndx", "STInt", "dtdlevel", "dtdtrend",
    "liqnonfinlevel", "liqnonfintrend", "ni2talevel", "ni2tatrend",
    "sizelevel", "sizetrend", "m2b", "sigma",
    "liqfinlevel", "lqfintrend", "DTDmedianFin", "DTDmedianNonFin",
    "dummyfin/SOE",
]

MODEL_NAME = "credit_risk_us"
CSV_PATH = "data/pd_input.csv"


# ═══════════════════════════════════════════════════════════
# Pyfunc wrapper — this is what the platform loads via
# mlflow.pyfunc.load_model(). It defines the "contract".
# ═══════════════════════════════════════════════════════════

class CreditRiskPDModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper for the LightGBM PD model.

    Pattern A: Features arrive ready.
    The wrapper just calls model.predict() on the input DataFrame.
    LightGBM handles NaN values natively (no imputation needed).

    This class gets serialized and stored inside the MLflow artifact.
    When the platform calls mlflow.pyfunc.load_model(), this class
    is deserialized and its predict() method is called.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Called once when the model is loaded. Loads the LightGBM booster."""
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=context.artifacts["lgb_model"])
        self.feature_names = self.booster.feature_name()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        """Called each time the platform runs inference.

        Args:
            model_input: DataFrame with the 17 feature columns.

        Returns:
            1D numpy array of PD (probability of default) scores in [0, 1].
        """
        return self.booster.predict(model_input[self.feature_names])


# ═══════════════════════════════════════════════════════════
# Synthetic label generation
# ═══════════════════════════════════════════════════════════

def generate_synthetic_labels(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Creates synthetic binary default labels based on features.

    Logic (mimics real PD relationships):
        - Lower DTD (distance to default) → higher default probability
        - Higher sigma (volatility) → higher default probability
        - Smaller size → higher default probability
        - Financial firms (dummyfin/SOE=1) → different risk profile

    We use a logistic function on a weighted combination of features,
    then add noise to make it realistic (not perfectly separable).

    Args:
        df: DataFrame with feature columns.
        seed: Random seed for reproducibility.

    Returns:
        Binary Series: 1 = default, 0 = no default. ~5% default rate.
    """
    rng = np.random.RandomState(seed)

    # Normalize key features to [0, 1] range for combining
    def norm(series: pd.Series) -> pd.Series:
        s = series.fillna(series.median())
        smin, smax = s.min(), s.max()
        if smax == smin:
            return pd.Series(0.5, index=series.index)
        return (s - smin) / (smax - smin)

    # Weighted risk score: higher = riskier
    risk_score = (
        -0.3 * norm(df["dtdlevel"])          # lower DTD = higher risk
        + 0.2 * norm(df["sigma"])             # higher volatility = higher risk
        - 0.15 * norm(df["sizelevel"])        # smaller size = higher risk
        - 0.1 * norm(df["ni2talevel"])        # lower profitability = higher risk
        - 0.1 * norm(df["liqnonfinlevel"])    # lower liquidity = higher risk
        + 0.05 * norm(df["STInt"])            # higher interest rates = higher risk
    )

    # Add noise
    risk_score += rng.normal(0, 0.15, size=len(df))

    # Logistic transform → probability
    prob = 1 / (1 + np.exp(-10 * (risk_score - risk_score.quantile(0.95))))

    # Sample binary labels from probability (target ~5% default rate)
    labels = rng.binomial(1, prob.clip(0, 1))

    default_rate = labels.mean()
    print(f"  Synthetic default rate: {default_rate:.2%} ({labels.sum():,} defaults out of {len(labels):,})")
    return pd.Series(labels, name="default_flag")


# ═══════════════════════════════════════════════════════════
# Main training + registration flow
# ═══════════════════════════════════════════════════════════

def train_and_register(train_size: int = 100_000) -> None:
    """Trains LightGBM and registers in MLflow as pyfunc."""
    print(f"\n{'='*60}")
    print(f"  TRAIN & REGISTER: {MODEL_NAME}")
    print(f"  Training rows: {train_size:,}")
    print(f"{'='*60}\n")

    # --- Step 1: Read data ---
    print("[1/5] Reading training data...", end=" ", flush=True)
    start = time.time()
    df = pd.read_csv(CSV_PATH)
    if train_size < len(df):
        df = df.sample(n=train_size, random_state=42)
    print(f"OK ({len(df):,} rows, {time.time()-start:.1f}s)")

    # --- Step 2: Generate synthetic labels ---
    print("[2/5] Generating synthetic default labels...")
    labels = generate_synthetic_labels(df)

    # --- Step 3: Train LightGBM ---
    print("[3/5] Training LightGBM...", end=" ", flush=True)
    start = time.time()

    X = df[FEATURE_COLUMNS]
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    elapsed = time.time() - start
    print(f"Done ({elapsed:.1f}s)")

    # Evaluate on validation set
    from sklearn.metrics import roc_auc_score
    y_pred_val = booster.predict(X_val)
    auc = roc_auc_score(y_val, y_pred_val)
    print(f"  Validation AUC: {auc:.4f}")

    # Feature importance
    importance = booster.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])
    print(f"\n  Top 5 features by importance:")
    for name, imp in feat_imp[:5]:
        print(f"    {name:25s} {imp:.0f}")

    # --- Step 4: Save model file + register in MLflow ---
    print(f"\n[4/5] Registering in MLflow...", flush=True)

    # Save booster to a temp file (required for pyfunc artifact)
    import tempfile
    import os
    model_dir = tempfile.mkdtemp()
    model_path = os.path.join(model_dir, "model.lgb")
    booster.save_model(model_path)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("credit_risk_us_training")

    with mlflow.start_run(run_name="lgb_pd_model_v1") as run:
        # Log training parameters
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("num_features", len(FEATURE_COLUMNS))

        # Log metrics
        mlflow.log_metric("val_auc", auc)
        mlflow.log_metric("default_rate", labels.mean())

        # Log the pyfunc model with the LightGBM booster as artifact
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CreditRiskPDModel(),
            artifacts={"lgb_model": model_path},
            pip_requirements=["lightgbm>=4.0", "pandas", "numpy"],
        )

        run_id = run.info.run_id
        print(f"  MLflow run ID: {run_id}")
        print(f"  Experiment: credit_risk_us_training")

    # --- Step 5: Register model and set Production alias ---
    print(f"[5/5] Setting 'Production' alias...", end=" ", flush=True)

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"Registered as version {version}")

    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "Production", version)
    print(f"  Alias 'Production' → version {version}")

    # --- Done ---
    print(f"\n{'='*60}")
    print(f"  MODEL REGISTERED SUCCESSFULLY")
    print(f"  Name:       {MODEL_NAME}")
    print(f"  Version:    {version}")
    print(f"  Alias:      Production")
    print(f"  AUC:        {auc:.4f}")
    print(f"  MLflow UI:  http://localhost:5000")
    print(f"{'='*60}\n")

    # Clean up temp file
    os.remove(model_path)
    os.rmdir(model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM and register in MLflow.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=100_000,
        help="Number of rows for training (default: 100000)",
    )
    args = parser.parse_args()
    train_and_register(args.train_size)


if __name__ == "__main__":
    main()
