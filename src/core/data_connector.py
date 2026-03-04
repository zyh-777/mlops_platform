"""Pluggable data source abstraction.

Each data source type (MySQL, S3, API) has a concrete implementation.
The factory creates the right connector based on input.source in the YAML.

Phase 1: Implement MySQLConnector only.
Phase 2: Add S3Connector, APIConnector as needed.
"""

from __future__ import annotations

# TODO: Implement DataConnector ABC with:
#   - fetch(run_date: str) -> pd.DataFrame

# TODO: Implement MySQLConnector(DataConnector)
#   - Reads connection config from InputConfig
#   - Substitutes {run_date} into query template
#   - Returns DataFrame from pd.read_sql

# TODO: Implement DataConnectorFactory
#   - _registry: dict mapping source type strings to connector classes
#   - create(input_config) -> DataConnector
