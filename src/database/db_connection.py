"""Database connection management.

Creates and manages SQLAlchemy engine for MySQL connections.
Used by both result_store (writing predictions) and run logging.

All database operations should use context managers (with engine.connect()).
"""

from __future__ import annotations

# TODO: Implement get_engine() -> sqlalchemy.Engine
#   - Read connection params from DbConfig
#   - Create engine with connection pooling
#   - Singleton pattern (reuse engine across calls)
