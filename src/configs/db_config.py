"""Database configuration from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class DbConfig:
    """MySQL connection parameters from .env."""

    HOST: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    USER: str = os.getenv("MYSQL_USER", "")
    PASSWORD: str = os.getenv("MYSQL_PASS", "")

    @classmethod
    def get_url(cls, database: str = "") -> str:
        """Returns SQLAlchemy connection URL."""
        db_part = f"/{database}" if database else ""
        return f"mysql+pymysql://{cls.USER}:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}{db_part}"
