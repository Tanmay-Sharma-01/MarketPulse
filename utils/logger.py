"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(name: str = "ai_stock_assistant", level: int = logging.INFO) -> logging.Logger:
    """Configure and return an application logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a child logger under the app namespace."""
    base = setup_logger()
    if name:
        return base.getChild(name)
    return base
