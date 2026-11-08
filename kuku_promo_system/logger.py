"""Structured logging module for the Kuku Promo Extraction System.

Provides consistent console and file logging while preventing accidental
leakage of API keys or credentials.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional
from kuku_promo_system.config import mask_secret, LLM_API_KEY


class SecretMaskingFormatter(logging.Formatter):
    """Custom formatter that automatically scrubs sensitive credentials from log output."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        self._secret = LLM_API_KEY if (LLM_API_KEY and len(LLM_API_KEY) > 5) else None

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._secret and self._secret in msg:
            msg = msg.replace(self._secret, mask_secret(self._secret))
        return msg


def setup_logger(name: str = "kuku_promo", level: int = logging.INFO) -> logging.Logger:
    """Initialize and configure a standard logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on multiple calls
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = SecretMaskingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Global default logger instance
logger = setup_logger()
