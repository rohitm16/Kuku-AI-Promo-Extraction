"""Configuration module for the Kuku Promo Extraction System.

Enforces pipeline constants, path definitions, duration bounds, and official
OpenAI API runtime settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load local .env if present (strictly for local development)
load_dotenv()

# --- Filesystem Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "I Married a Broken Soldier. He was a Duke"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "raw_outputs"
RAW_OUTPUTS_DIR = DEFAULT_OUTPUT_DIR
HISTORICAL_BASELINE_DIR = DEFAULT_OUTPUT_DIR / "simulated_baseline"
OPENAI_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "openai_run"

# Standard Deliverable Artifact Paths
FINAL_RANKED_PROMOS_JSON = DEFAULT_OUTPUT_DIR / "ranked_promos.json"
FINAL_RANKED_PROMOS_CSV = DEFAULT_OUTPUT_DIR / "ranked_promos.csv"
UNRESTRICTED_RANKED_PROMOS_JSON = DEFAULT_OUTPUT_DIR / "ranked_promos_unrestricted.json"
UNRESTRICTED_RANKED_PROMOS_CSV = DEFAULT_OUTPUT_DIR / "ranked_promos_unrestricted.csv"
FINAL_RAW_EVALUATIONS_JSON = DEFAULT_OUTPUT_DIR / "raw_evaluations.json"
DEFAULT_CHECKPOINT_FILE = DEFAULT_OUTPUT_DIR / "eval_checkpoint.jsonl"
FINAL_TELEMETRY_FILE = DEFAULT_OUTPUT_DIR / "phase10_final_telemetry.json"

# --- Hard Duration Bounds (Seconds) ---
# Strictly enforced by the assignment brief: 20.0s <= promo duration <= 60.0s
MIN_PROMO_DURATION: float = 20.0
MAX_PROMO_DURATION: float = 60.0

# --- Timeline & Clustering Parameters ---
# Maximum temporal gap between consecutive/overlapping track events to merge into one Atomic Beat
BEAT_MERGE_GAP_SEC: float = 0.8

# --- Candidate Generation & Pruning Parameters ---
MIN_CANDIDATES_PER_EPISODE: int = 8
TARGET_CANDIDATES_PER_EPISODE: int = int(os.getenv("MAX_CANDIDATES_PER_EPISODE", "10"))
MAX_CANDIDATES_PER_EPISODE: int = 12

# Temporal Non-Maximum Suppression (NMS) Intersection-over-Union threshold
TEMPORAL_NMS_IOU_THRESHOLD: float = 0.60

# Candidate Lane Allocations (within target count)
LANE_QUOTAS = {
    "boundary": 4,     # High-salience hooks and sharp cliffhangers
    "development": 4,  # Pacing acceleration and dialogue turn velocity
    "anomaly": 3,      # Subtle revelations and high-stakes thematic events
}

# --- Official OpenAI Provider Configuration ---
# Active runtime uses the official OpenAI API exclusively
OPENAI_DEFAULT_MODEL: str = "gpt-4o-mini"
OPENAI_VERIFIED_PRESETS: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
    "gpt-4-turbo",
]

# Estimated pricing per 1M tokens for cost projections (USD)
OPENAI_MODEL_PRICING = {
    "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00},
    "o3-mini": {"input_per_m": 1.10, "output_per_m": 4.40},
    "gpt-4-turbo": {"input_per_m": 10.00, "output_per_m": 30.00},
    "meta-llama/llama-3.3-70b-instruct": {"input_per_m": 0.12, "output_per_m": 0.30},
}

LLM_PROVIDER: str = "openai"
LLM_MODEL: str = os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", OPENAI_DEFAULT_MODEL))
LLM_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL", None)
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", None))
LLM_API_KEY: Optional[str] = OPENAI_API_KEY

# Evaluation Execution Controls & Token Bounds
MAX_COMPLETION_TOKENS: int = int(os.getenv("MAX_COMPLETION_TOKENS") or "1500")
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE") or "3")
EVALUATION_TIMEOUT_SEC: float = float(os.getenv("EVALUATION_TIMEOUT_SEC") or "60.0")
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES") or "3")
RETRY_BASE_DELAY_SEC: float = 2.0


def validate_duration(duration_sec: float) -> bool:
    """Assert that a duration strictly satisfies the 20.0s to 60.0s requirement."""
    return MIN_PROMO_DURATION <= round(duration_sec, 2) <= MAX_PROMO_DURATION


def mask_secret(secret: Optional[str]) -> str:
    """Safely mask API keys for logs and UI display."""
    if not secret:
        return "<NOT_SET>"
    secret_str = str(secret).strip()
    if len(secret_str) <= 8:
        return "***"
    return f"{secret_str[:4]}...{secret_str[-4:]}"
