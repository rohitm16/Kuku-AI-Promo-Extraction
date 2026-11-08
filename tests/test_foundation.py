"""Unit tests for Step 1 Foundation: Config, Logger, and Security."""

import os
from pathlib import Path
import pytest

from kuku_promo_system.config import (
    MIN_PROMO_DURATION,
    MAX_PROMO_DURATION,
    BEAT_MERGE_GAP_SEC,
    validate_duration,
    mask_secret,
    PROJECT_ROOT,
    DEFAULT_DATA_DIR,
)
from kuku_promo_system.logger import setup_logger, SecretMaskingFormatter


def test_duration_constraints():
    """Verify that duration bounds strictly match the assignment brief (20s to 60s)."""
    assert MIN_PROMO_DURATION == 20.0
    assert MAX_PROMO_DURATION == 60.0
    assert BEAT_MERGE_GAP_SEC == 0.8

    # Boundary tests
    assert validate_duration(20.0) is True
    assert validate_duration(60.0) is True
    assert validate_duration(35.5) is True

    # Violations
    assert validate_duration(19.99) is False
    assert validate_duration(60.01) is False
    assert validate_duration(12.6) is False
    assert validate_duration(75.0) is False


def test_secret_masking():
    """Verify that API keys are never exposed in plaintext."""
    assert mask_secret(None) == "<NOT_SET>"
    assert mask_secret("") == "<NOT_SET>"
    assert mask_secret("short") == "***"
    assert mask_secret("sk-or-v1-abcdef1234567890") == "sk-o...7890"


def test_gitignore_protects_secrets():
    """Verify that .gitignore properly excludes .env and sensitive caches."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"
    content = gitignore_path.read_text(encoding="utf-8")
    assert ".env" in content
    assert "*.env" in content
    assert "eval_checkpoint.jsonl" in content


@pytest.mark.skipif(not DEFAULT_DATA_DIR.exists(), reason="DEFAULT_DATA_DIR does not exist")
def test_data_directory_exists():
    """Verify that the official screenplay directory is accessible."""
    assert DEFAULT_DATA_DIR.exists(), f"Data dir {DEFAULT_DATA_DIR} must exist"
    files = list(DEFAULT_DATA_DIR.glob("*.txt"))
    assert len(files) == 50, f"Expected 50 episodes, found {len(files)}"
