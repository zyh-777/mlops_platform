"""Database connection management.

Creates and manages SQLAlchemy engine for MySQL connections.
Used by both result_store (writing predictions) and run logging.

All database operations should use context managers (with engine.connect()).

Universal pattern: Singleton Engine
    - create_engine() does NOT open a connection — it prepares a pool.
    - Actual connections happen lazily when you call engine.connect().
    - So this module is safe to import/call without a running database.
    - We cache the engine so the entire platform shares one connection pool.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.configs.db_config import DbConfig
from src.logger.logger import get_logger

logger = get_logger("db_connection")

# Module-level cache: one engine per database name.
# Most apps only need one, but we support multiple (e.g., "platform", "results").
_engines: dict[str, Engine] = {}


def get_engine(database: str = "") -> Engine:
    """Returns a SQLAlchemy engine, creating it on first call.

    The engine manages a connection pool internally. It does NOT open a
    connection until you actually call engine.connect(), so this function
    is safe to call even if the database is not reachable yet.

    Args:
        database: MySQL database/schema name. Empty string for no default DB.

    Returns:
        A SQLAlchemy Engine instance (cached per database name).
    """
    if database not in _engines:
        url = DbConfig.get_url(database)
        _engines[database] = create_engine(
            url,
            pool_size=5,          # Keep 5 connections ready in the pool
            max_overflow=10,      # Allow up to 10 extra connections under load
            pool_recycle=3600,    # Recycle connections after 1 hour (MySQL default timeout is 8h)
            pool_pre_ping=True,   # Test connection health before using it
        )
        logger.info(f"Created engine for database='{database}' at {DbConfig.HOST}:{DbConfig.PORT}")
    return _engines[database]


def check_connection(database: str = "") -> bool:
    """Tests whether the database is reachable.

    Useful for health checks and startup validation.
    Returns True if the connection works, False otherwise.

    Args:
        database: MySQL database/schema name to test.

    Returns:
        True if connection succeeds, False otherwise.
    """
    try:
        engine = get_engine(database)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connection check passed for database='{database}'")
        return True
    except Exception as e:
        logger.error(f"Connection check failed for database='{database}': {e}")
        return False
