"""Pluggable data source abstraction.

Each data source type (MySQL, S3, API) has a concrete implementation.
The factory creates the right connector based on input.source in the YAML.

Universal pattern: ABC + Factory (Strategy Pattern)
    - ABC defines the contract: fetch(run_date) → DataFrame.
    - Concrete classes implement the contract for each source type.
    - Factory picks the right class based on a string key from YAML.
    - Adding a new source = one new class + one registry entry. No other changes.

Phase 1: MySQLConnector + FileConnector.
Phase 2: Add SnowflakeConnector, S3Connector, APIConnector as needed.

Usage:
    connector = DataConnectorFactory.create(config.input)
    df = connector.fetch("2026-03-05")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.core.project_config import InputConfig
from src.database.db_connection import get_engine
from src.logger.logger import get_logger

logger = get_logger("data_connector")


# ═══════════════════════════════════════════════════════════
# ABSTRACT BASE CLASS — the contract all connectors follow
# ═══════════════════════════════════════════════════════════


class DataConnector(ABC):
    """Base class for all data connectors.

    Every connector must implement fetch(), which takes a run_date string
    and returns a pandas DataFrame. That's the entire contract.
    """

    @abstractmethod
    def fetch(self, run_date: str) -> pd.DataFrame:
        """Fetches input data for the given run date.

        Args:
            run_date: Date string like "2026-03-05". Used to filter data.

        Returns:
            DataFrame with the input data for model prediction.
        """


# ═══════════════════════════════════════════════════════════
# CONCRETE IMPLEMENTATIONS — one per source type
# ═══════════════════════════════════════════════════════════


class MySQLConnector(DataConnector):
    """Reads data from a MySQL database using a SQL query from YAML.

    The query can contain {run_date} which gets substituted at runtime.
    Example YAML:
        query: |
            SELECT txn_id, amount, merchant_risk
            FROM transactions
            WHERE txn_date = '{run_date}'
    """

    def __init__(self, input_config: InputConfig) -> None:
        self.input_config = input_config
        self.database = input_config.connection.get("database", "")

    def fetch(self, run_date: str) -> pd.DataFrame:
        """Executes the SQL query with {run_date} substituted.

        Args:
            run_date: Date string to substitute into the query template.

        Returns:
            DataFrame with query results.

        Raises:
            ValueError: If no query is configured.
            sqlalchemy.exc.OperationalError: If the database is unreachable.
        """
        if not self.input_config.query:
            raise ValueError("MySQL connector requires a 'query' in input config")

        # Substitute {run_date} in the query template
        query = self.input_config.query.format(run_date=run_date)

        engine = get_engine(self.database)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        logger.info(f"Fetched {len(df)} rows from MySQL (database={self.database})")
        return df


class FileConnector(DataConnector):
    """Reads data from a local CSV or Parquet file.

    The file path can contain {run_date} which gets substituted at runtime.
    Useful for local testing (no DB needed) and backfills from data exports.

    Example YAML:
        input:
          source: file
          path: "data/fraud_features_{run_date}.parquet"

    If the path has no {run_date} placeholder, the same file is used for all dates
    (typical when DE gives you a single export with a date column inside).
    """

    def __init__(self, input_config: InputConfig) -> None:
        self.input_config = input_config
        # "path" can live in connection dict or as a top-level key
        self.path_template = (
            input_config.connection.get("path", "")
            or input_config.query  # reuse query field as path if connection.path not set
        )

    def fetch(self, run_date: str) -> pd.DataFrame:
        """Reads a CSV or Parquet file, substituting {run_date} in the path.

        Args:
            run_date: Date string to substitute into the path template.

        Returns:
            DataFrame with file contents.

        Raises:
            ValueError: If no path is configured.
            FileNotFoundError: If the resolved file doesn't exist.
        """
        if not self.path_template:
            raise ValueError(
                "File connector requires a 'path' in input.connection "
                "or a 'query' field used as file path"
            )

        resolved = self.path_template.format(run_date=run_date)
        path = Path(resolved)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in (".csv", ".tsv"):
            sep = "\t" if path.suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            raise ValueError(
                f"Unsupported file format: '{path.suffix}'. "
                f"Use .csv, .tsv, or .parquet"
            )

        logger.info(f"Fetched {len(df)} rows from file: {path}")
        return df


# ═══════════════════════════════════════════════════════════
# FACTORY — picks the right connector from a string key
# ═══════════════════════════════════════════════════════════
# Why a factory? Because the pipeline code doesn't care which
# connector it uses. It just calls:
#     connector = DataConnectorFactory.create(config.input)
#     df = connector.fetch(run_date)
# The YAML decides which connector gets created.


class DataConnectorFactory:
    """Creates the appropriate DataConnector based on input.source.

    Registry pattern: maps source type strings to connector classes.
    To add a new source type:
        1. Create a class that extends DataConnector
        2. Add it to _registry below
        3. Teams can now use source: your_type in their YAML
    """

    _registry: dict[str, type[DataConnector]] = {
        "mysql": MySQLConnector,
        "file": FileConnector,
        # "snowflake": SnowflakeConnector,  # Phase 2
        # "s3": S3Connector,               # Phase 2
    }

    @classmethod
    def create(cls, input_config: InputConfig) -> DataConnector:
        """Creates a connector instance for the given input config.

        Args:
            input_config: The input section from ProjectConfig.

        Returns:
            A DataConnector instance ready to call .fetch(run_date).

        Raises:
            ValueError: If the source type is not in the registry.
        """
        source = input_config.source.lower()
        connector_cls = cls._registry.get(source)
        if connector_cls is None:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Unknown data source: '{source}'. Available: {available}"
            )
        return connector_cls(input_config)
