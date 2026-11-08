"""Phase 12 Controlled Pilot Evaluation on 3 Representative Candidates.

Executes live OpenAI inference on 3 benchmark candidates:
1. Strong: EP01_CAND_01_09 (Episode 1, high dramatic tension, altar confrontation)
2. Borderline: EP06_CAND_02_07 (Episode 6, mid-tier narrative dialogue)
3. Lore-Risk / Weak Standalone: EP01_CAND_03_06 (Episode 1, heavy prior lore dependency)

Saves outputs in raw_outputs/openai_run/phase12_pilot/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import List, Dict, Any, Optional, Tuple

from kuku_promo_system.config import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_OUTPUT_DIR,
    RAW_OUTPUTS_DIR,
    OPENAI_MODEL_PRICING,
)
from kuku_promo_system.logger import logger
from kuku_promo_system.llm_client import LLMClientAdapter
from kuku_promo_system.evaluator import (
    PromoEvaluator,
    EvaluatedCandidate,
    SinglePromoEvaluation,
)
from kuku_promo_system.pruner import CandidateWindow
from kuku_promo_system.ranker import (
    rank_and_diversify_promos,
    rank_promos_unrestricted,
    export_promos_to_files,
)

PILOT_OUTPUT_DIR = OPENAI_OUTPUT_DIR / "phase12_pilot"
TARGET_CANDIDATE_IDS = [
    "EP01_CAND_01_09",  # Strong
    "EP06_CAND_02_07",  # Borderline
    "EP01_CAND_03_06",  # Lore-Risk
]


def load_target_candidates() -> List[CandidateWindow]:
    """Extract the 3 target candidates from the canonical raw evaluations dataset."""
    raw_evals_path = RAW_OUTPUTS_DIR / "raw_evaluations.json"
    if not raw_evals_path.exists():
        raise FileNotFoundError(f"Raw evaluations not found at {raw_evals_path}")

    with open(raw_evals_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cands_by_id = {}
    for item in data:
        cand_dict = item.get("candidate", {})
        cid = cand_dict.get("candidate_id")
        if cid in TARGET_CANDIDATE_IDS:
            cands_by_id[cid] = CandidateWindow.model_validate(cand_dict)

    ordered = [cands_by_id[cid] for cid in TARGET_CANDIDATE_IDS if cid in cands_by_id]
    if len(ordered) != len(TARGET_CANDIDATE_IDS):
        raise ValueError(f"Could not locate all target candidates. Found {len(ordered)}/{len(TARGET_CANDIDATE_IDS)}")

    return ordered


def run_pilot(
    api_key: Optional[str] = None,
    model_id: str = OPENAI_DEFAULT_MODEL,
    output_dir: Path = PILOT_OUTPUT_DIR,
) -> Tuple[List[EvaluatedCandidate], Dict[str, Any]]:
    """Execute the controlled pilot evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    candidates = load_target_candidates()
    logger.info(f"Loaded {len(candidates)} target candidates for Phase 12 Pilot.")

    # Initialize OpenAI adapter
    adapter = LLMClientAdapter(
        api_key=api_key,
        model=model_id,
    )

    # Preflight connection check
    success, msg, lat = adapter.verify_connection(model_id)
    logger.info(f"Connection Preflight: success={success}, msg='{msg}', latency={lat}s")
    if not success:
        print(f"Connection probe failed: {msg}")
        sys.exit(1)

    evaluator = PromoEvaluator(
        llm_client=adapter,
        checkpoint_dir=output_dir,
        checkpoint_filename="eval_checkpoint.jsonl",
        batch_size=3,
    )

    logger.info(f"Dispatching batch evaluation on model '{model_id}'...")
    evaluated_records = evaluator.evaluate_candidates(candidates)

    # Rank & Export pilot deliverables
    diversity_promos = rank_and_diversify_promos(evaluated_records)
    unrestricted_promos = rank_promos_unrestricted(evaluated_records)

    export_promos_to_files(
        ranked_promos=diversity_promos,
        raw_candidates=evaluated_records,
        output_dir=output_dir,
        unrestricted_promos=unrestricted_promos,
    )

    elapsed = round(time.time() - t0, 2)
    telemetry = {
        "model_id": model_id,
        "provider": "openai",
        "is_simulated": False,
        "candidate_count": len(candidates),
        "evaluated_success": sum(1 for r in evaluated_records if r.status == "evaluated"),
        "evaluated_failed": sum(1 for r in evaluated_records if r.status == "evaluation_failed"),
        "elapsed_seconds": elapsed,
        "output_directory": str(output_dir),
    }

    telemetry_path = output_dir / "pilot_telemetry.json"
    with open(telemetry_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    logger.info(f"Pilot completed in {elapsed}s. Outputs saved to {output_dir}")
    return evaluated_records, telemetry


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
    run_pilot(api_key=key, model_id=model)
