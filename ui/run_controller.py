"""Pipeline execution controller with real-time Streamlit progress indicators and cost estimation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

from kuku_promo_system.config import (
    OPENAI_OUTPUT_DIR,
    OPENAI_MODEL_PRICING,
    TARGET_CANDIDATES_PER_EPISODE,
    DEFAULT_DATA_DIR,
)
from kuku_promo_system.parser import parse_screenplay_file, ParsedEpisode
from kuku_promo_system.timeline import build_episode_timeline
from kuku_promo_system.pruner import (
    CandidateWindow,
    generate_all_valid_windows,
    prune_candidates_3_lane,
)
from kuku_promo_system.llm_client import LLMClientAdapter
from kuku_promo_system.evaluator import PromoEvaluator, EvaluatedCandidate
from kuku_promo_system.ranker import (
    rank_and_diversify_promos,
    rank_promos_unrestricted,
    export_promos_to_files,
    FinalRankedPromo,
)


def estimate_pipeline_cost(num_episodes: int, target_cands_per_ep: int, model_id: str) -> Dict[str, Any]:
    """Calculate estimated token usage and financial cost before dispatching an OpenAI run."""
    est_candidates = num_episodes * target_cands_per_ep
    est_batches = (est_candidates + 2) // 3
    # Average tokens from production benchmarks: ~2,300 prompt tokens + ~650 completion tokens per batch of 3
    est_prompt_tokens = est_batches * 2300
    est_completion_tokens = est_batches * 650
    est_total_tokens = est_prompt_tokens + est_completion_tokens

    pricing = OPENAI_MODEL_PRICING.get(model_id, {"input_per_m": 0.15, "output_per_m": 0.60})
    est_cost = (est_prompt_tokens * pricing["input_per_m"] / 1_000_000.0) + (
        est_completion_tokens * pricing["output_per_m"] / 1_000_000.0
    )

    return {
        "candidate_count": est_candidates,
        "batch_count": est_batches,
        "prompt_tokens": est_prompt_tokens,
        "completion_tokens": est_completion_tokens,
        "total_tokens": est_total_tokens,
        "estimated_cost_usd": est_cost,
        "pricing_basis": pricing,
    }


def execute_pipeline_ui(
    screenplay_files: List[Path],
    api_key: Optional[str],
    model_id: str,
    max_tokens: int,
    target_cands_per_ep: int = TARGET_CANDIDATES_PER_EPISODE,
    output_dir: Path = OPENAI_OUTPUT_DIR,
    base_url: Optional[str] = None,
) -> Tuple[List[FinalRankedPromo], List[FinalRankedPromo], List[EvaluatedCandidate], Dict[str, Any]]:
    """Execute the end-to-end promotional extraction pipeline with Streamlit visual progress."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Progress containers in Streamlit UI
    progress_bar = st.progress(0, text="Initializing Pipeline...")
    status_text = st.empty()
    batch_log_area = st.empty()

    # Step 1: Parsing & Timeline Construction (0% to 25%)
    status_text.markdown("**Phase 1/4: Parsing Multi-Track Screenplays & Assembling Timeline...**")
    all_shortlisted: List[CandidateWindow] = []
    total_events = 0
    total_beats = 0
    total_valid_windows = 0

    num_files = len(screenplay_files)
    for idx, fpath in enumerate(screenplay_files, 1):
        progress_val = int(25 * (idx / num_files))
        progress_bar.progress(progress_val, text=f"Parsing screenplay {idx}/{num_files}: {fpath.name}")

        ep = parse_screenplay_file(fpath)
        if len(ep.all_events) == 0:
            raise ValueError(f"Parsed file '{fpath.name}' has zero events.")
        total_events += len(ep.all_events)

        beats = build_episode_timeline(ep)
        total_beats += len(beats)

        windows = generate_all_valid_windows(beats, episode_num=ep.episode_num)
        total_valid_windows += len(windows)

        shortlisted = prune_candidates_3_lane(windows, target_count=target_cands_per_ep)
        all_shortlisted.extend(shortlisted)

    progress_bar.progress(30, text=f"Pruned {len(all_shortlisted)} candidates from {total_valid_windows} valid windows.")

    # Step 2: LLM Evaluation (30% to 85%)
    status_text.markdown(f"**Phase 2/4: Cold-Viewer LLM Evaluation ({model_id})...**")

    adapter = LLMClientAdapter(
        api_key=api_key,
        model=model_id,
        base_url=base_url or None,
        max_tokens=max_tokens,
    )

    evaluator = PromoEvaluator(
        llm_client=adapter,
        checkpoint_dir=output_dir,
        checkpoint_filename="eval_checkpoint.jsonl",
        batch_size=3,
    )

    # Callback to update Streamlit UI on each evaluated batch
    def on_batch_progress(batch_num: int, total_batches: int, batch_cands: List[EvaluatedCandidate], msg: str):
        eval_progress = 30 + int(55 * (batch_num / total_batches))
        progress_bar.progress(eval_progress, text=f"Evaluating Batch {batch_num}/{total_batches} ({msg})")
        evaluated_so_far = sum(1 for c in batch_cands if c.status == "evaluated")
        failed_so_far = sum(1 for c in batch_cands if c.status == "evaluation_failed")
        batch_log_area.caption(
            f"Batch {batch_num}/{total_batches} &bull; Candidates in batch: {len(batch_cands)} &bull; "
            f"Evaluated: {evaluated_so_far} &bull; Failed: {failed_so_far}"
        )

    evaluated_records = evaluator.evaluate_candidates(
        candidates=all_shortlisted,
        progress_callback=on_batch_progress,
    )

    evaluated_count = sum(1 for r in evaluated_records if r.status == "evaluated")
    if evaluated_count == 0:
        first_err = next((r.error_message.strip() for r in evaluated_records if r.error_message and r.error_message.strip()), "")
        msg = "No candidate evaluated successfully. Nothing was ranked or exported."
        if first_err:
            msg = f"{msg} {first_err}"
        raise RuntimeError(msg)

    # Step 3: Global Ranking & Diversity Guardrails (85% to 95%)
    progress_bar.progress(88, text="Phase 3/4: Deduplicating and Computing Diversity-Adjusted Rankings...")
    status_text.markdown("**Phase 3/4: Deduplicating and Computing Diversity-Adjusted Rankings...**")

    diversity_promos = rank_and_diversify_promos(evaluated_records)
    unrestricted_promos = rank_promos_unrestricted(evaluated_records)

    # Step 4: Deliverable Export (95% to 100%)
    progress_bar.progress(95, text="Phase 4/4: Exporting JSON and CSV Deliverables...")
    status_text.markdown("**Phase 4/4: Exporting JSON and CSV Deliverables...**")

    export_promos_to_files(
        ranked_promos=diversity_promos,
        raw_candidates=evaluated_records,
        output_dir=output_dir,
        unrestricted_promos=unrestricted_promos,
    )

    elapsed = round(time.time() - t0, 2)
    progress_bar.progress(100, text=f"Pipeline Execution Complete in {elapsed}s!")
    status_text.success(
        f"Completed: {len(screenplay_files)} episodes &bull; {len(all_shortlisted)} candidates &bull; "
        f"{len(diversity_promos)} ranked promos &bull; Elapsed: {elapsed}s"
    )

    telemetry = {
        "model_id": model_id,
        "provider": adapter.provider,
        "is_simulated": False,
        "episodes_count": len(screenplay_files),
        "total_events": total_events,
        "total_beats": total_beats,
        "valid_windows": total_valid_windows,
        "shortlisted_candidates": len(all_shortlisted),
        "evaluated_success": sum(1 for r in evaluated_records if r.status == "evaluated"),
        "evaluated_failed": sum(1 for r in evaluated_records if r.status == "evaluation_failed"),
        "ranked_promos_count": len(diversity_promos),
        "elapsed_seconds": elapsed,
    }

    return diversity_promos, unrestricted_promos, evaluated_records, telemetry
