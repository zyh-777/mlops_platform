"""Structured logging for the platform.

All log messages should include project_id for multi-project context.
Usage:
    logger = get_logger("pipeline")
    logger.info(f"[{config.project_id}] Loaded 15000 rows")
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Creates a configured logger.

    Args:
        name: Logger name (typically module or component name).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(f"mlops.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
