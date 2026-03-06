"""Tests for src/core/data_connector.py"""

from __future__ import annotations

import pytest

from src.core.project_config import InputConfig
from src.core.data_connector import (
    DataConnector,
    DataConnectorFactory,
    FileConnector,
    MySQLConnector,
)


class TestDataConnectorFactory:
    def test_creates_mysql_connector(self):
        input_config = InputConfig(
            source="mysql",
            connection={"host": "localhost", "port": 3306, "database": "test"},
            query="SELECT 1",
        )
        connector = DataConnectorFactory.create(input_config)
        assert isinstance(connector, MySQLConnector)

    def test_case_insensitive_source(self):
        input_config = InputConfig(
            source="MySQL",
            connection={"host": "localhost", "database": "test"},
            query="SELECT 1",
        )
        connector = DataConnectorFactory.create(input_config)
        assert isinstance(connector, MySQLConnector)

    def test_creates_file_connector(self):
        input_config = InputConfig(
            source="file",
            connection={"path": "data/test.csv"},
        )
        connector = DataConnectorFactory.create(input_config)
        assert isinstance(connector, FileConnector)

    def test_raises_on_unknown_source(self):
        input_config = InputConfig(source="mongodb")
        with pytest.raises(ValueError, match="Unknown data source"):
            DataConnectorFactory.create(input_config)


class TestMySQLConnector:
    def test_init_reads_database_from_connection(self):
        input_config = InputConfig(
            source="mysql",
            connection={"database": "warehouse"},
            query="SELECT * FROM t WHERE date = '{run_date}'",
        )
        connector = MySQLConnector(input_config)
        assert connector.database == "warehouse"

    def test_raises_on_empty_query(self):
        input_config = InputConfig(
            source="mysql",
            connection={"database": "warehouse"},
            query="",
        )
        connector = MySQLConnector(input_config)
        with pytest.raises(ValueError, match="query"):
            connector.fetch("2026-03-05")


class TestFileConnector:
    def test_reads_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,amount,risk\nA,100,0.3\nB,200,0.8\n")

        input_config = InputConfig(
            source="file",
            connection={"path": str(csv_file)},
        )
        connector = FileConnector(input_config)
        df = connector.fetch("2026-03-05")

        assert len(df) == 2
        assert list(df.columns) == ["id", "amount", "risk"]

    def test_reads_parquet(self, tmp_path):
        import pandas as pd
        parquet_file = tmp_path / "data.parquet"
        pd.DataFrame({"id": ["A", "B"], "value": [1.0, 2.0]}).to_parquet(parquet_file)

        input_config = InputConfig(
            source="file",
            connection={"path": str(parquet_file)},
        )
        connector = FileConnector(input_config)
        df = connector.fetch("2026-03-05")

        assert len(df) == 2
        assert "value" in df.columns

    def test_substitutes_run_date_in_path(self, tmp_path):
        csv_file = tmp_path / "data_2026-03-05.csv"
        csv_file.write_text("id,val\nX,1\n")

        template = str(tmp_path / "data_{run_date}.csv")
        input_config = InputConfig(
            source="file",
            connection={"path": template},
        )
        connector = FileConnector(input_config)
        df = connector.fetch("2026-03-05")

        assert len(df) == 1

    def test_raises_on_missing_file(self):
        input_config = InputConfig(
            source="file",
            connection={"path": "/nonexistent/file.csv"},
        )
        connector = FileConnector(input_config)
        with pytest.raises(FileNotFoundError):
            connector.fetch("2026-03-05")

    def test_raises_on_no_path(self):
        input_config = InputConfig(source="file")
        connector = FileConnector(input_config)
        with pytest.raises(ValueError, match="path"):
            connector.fetch("2026-03-05")

    def test_raises_on_unsupported_format(self, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text('{"a": 1}')

        input_config = InputConfig(
            source="file",
            connection={"path": str(json_file)},
        )
        connector = FileConnector(input_config)
        with pytest.raises(ValueError, match="Unsupported file format"):
            connector.fetch("2026-03-05")
