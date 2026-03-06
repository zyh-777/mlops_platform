"""Step 2: Load pd_input.csv into MySQL to mimic production data.

In production, the DE team maintains feature tables in a database.
This script simulates that by loading the CSV into MySQL.

Prerequisites:
    Docker containers running: docker compose -f docker-compose.dev.yaml up -d

Usage:
    python -m tools.load_csv_to_mysql                    # load all 2.3M rows
    python -m tools.load_csv_to_mysql --sample 50000     # load 50k rows (faster for testing)
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from sqlalchemy import text

from src.database.db_connection import get_engine
from src.logger.logger import get_logger

logger = get_logger("load_csv_to_mysql")

CSV_PATH = "data/pd_input.csv"
TARGET_DB = "features"
TARGET_TABLE = "pd_input"


def load_csv_to_mysql(sample_n: int | None = None) -> None:
    """Loads pd_input.csv into MySQL features.pd_input table."""

    # --- Step 1: Read CSV ---
    print(f"[1/4] Reading {CSV_PATH}...", end=" ", flush=True)
    start = time.time()
    df = pd.read_csv(CSV_PATH)
    print(f"OK ({len(df):,} rows, {df.shape[1]} columns, {time.time()-start:.1f}s)")

    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42)
        print(f"       Sampled down to {len(df):,} rows")

    # --- Step 2: Create features database ---
    print(f"[2/4] Creating database '{TARGET_DB}'...", end=" ", flush=True)
    engine = get_engine()  # connect without specifying database
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{TARGET_DB}`"))
        conn.commit()
    print("OK")

    # --- Step 3: Write to MySQL ---
    print(f"[3/4] Writing {len(df):,} rows to {TARGET_DB}.{TARGET_TABLE}...", flush=True)
    start = time.time()
    features_engine = get_engine(TARGET_DB)

    # Drop existing table and recreate (clean start for testing)
    with features_engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{TARGET_TABLE}`"))
        conn.commit()

    # Write in chunks for large datasets
    chunk_size = 10_000
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[chunk_start:chunk_start + chunk_size]
        chunk.to_sql(
            name=TARGET_TABLE,
            con=features_engine,
            if_exists="append",
            index=False,
        )
        pct = (i + 1) / total_chunks * 100
        print(f"       Chunk {i+1}/{total_chunks} ({pct:.0f}%)", end="\r", flush=True)

    elapsed = time.time() - start
    print(f"       Done — {len(df):,} rows written in {elapsed:.1f}s          ")

    # --- Step 4: Verify ---
    print(f"[4/4] Verifying...", end=" ", flush=True)
    with features_engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM `{TARGET_TABLE}`")).scalar()
        sample = pd.read_sql(text(f"SELECT * FROM `{TARGET_TABLE}` LIMIT 3"), conn)
    print(f"OK ({count:,} rows in {TARGET_DB}.{TARGET_TABLE})")

    print(f"\n--- Sample rows ---")
    print(sample.to_string(index=False))

    print(f"\n--- Column types ---")
    for col in df.columns[:5]:
        print(f"  {col}: {df[col].dtype}")
    print(f"  ... and {len(df.columns) - 5} more columns")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load pd_input.csv into MySQL.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only load N rows (faster for testing). Default: load all.",
    )
    args = parser.parse_args()
    load_csv_to_mysql(args.sample)


if __name__ == "__main__":
    main()
