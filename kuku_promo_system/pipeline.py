"""Unified CLI execution pipeline for Kuku Promo Extraction System.

Coordinates end-to-end ingestion, parsing, timeline clustering, 3-lane pruning,
OpenAI LLM evaluation, global ranking, and export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from pydantic import BaseModel

from kuku_promo_system.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    OPENAI_OUTPUT_DIR,
    TARGET_CANDIDATES_PER_EPISODE,
    MIN_PROMO_DURATION,
    MAX_PROMO_DURATION,
    OPENAI_DEFAULT_MODEL,
    MAX_COMPLETION_TOKENS,
)
from kuku_promo_system.logger import logger
from kuku_promo_system.parser import parse_screenplay_file, ParsedEpisode
from kuku_promo_system.timeline import build_episode_timeline, AtomicBeat
from kuku_promo_system.pruner import (
    CandidateWindow,
    generate_all_valid_windows,
    prune_candidates_3_lane,
)
from kuku_promo_system.llm_client import LLMClientAdapter, MissingAPIKeyError
from kuku_promo_system.evaluator import PromoEvaluator, EvaluatedCandidate
from kuku_promo_system.ranker import (
    rank_and_diversify_promos,
    rank_promos_unrestricted,
    export_promos_to_files,
    FinalRankedPromo,
)
from kuku_promo_system.audit import run_benchmark_audit


class PipelineSummary(BaseModel):
    total_episodes_processed: int = 0
    total_events_extracted: int = 0
    total_beats_generated: int = 0
    total_valid_windows_generated: int = 0
    total_shortlisted_candidates: int = 0
    total_evaluated_success: int = 0
    total_evaluated_failed: int = 0
    total_ranked_promos: int = 0
    elapsed_seconds: float = 0.0


def run_promo_pipeline(
    episode_numbers: Optional[List[int]] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = OPENAI_OUTPUT_DIR,
    target_candidates_per_ep: int = TARGET_CANDIDATES_PER_EPISODE,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    checkpoint_filename: str = "eval_checkpoint.jsonl",
    progress_callback: Optional[Callable[[int, int, List[EvaluatedCandidate], str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[List[FinalRankedPromo], PipelineSummary]:
    """Execute the complete promo extraction and ranking pipeline."""
    start_time = time.time()
    summary = PipelineSummary()

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Screenplay directory not found: {data_path}")

    # Discover episode files
    all_files = sorted(list(data_path.glob("episode-*.txt")))
    if not all_files:
        raise FileNotFoundError(f"No episode files found in {data_path}")

    # Filter by requested episode numbers if provided
    selected_files = []
    for f in all_files:
        parts = f.name.split("-")
        if len(parts) >= 2:
            try:
                ep_n = int(parts[1])
                if episode_numbers is None or ep_n in episode_numbers:
                    selected_files.append(f)
            except ValueError:
                continue

    logger.info(f"Starting pipeline on {len(selected_files)} episode file(s)...")

    # Initialize OpenAI LLM evaluator
    llm_adapter = LLMClientAdapter(
        api_key=api_key,
        model=model or OPENAI_DEFAULT_MODEL,
    )
    evaluator = PromoEvaluator(
        llm_client=llm_adapter,
        checkpoint_dir=output_dir,
        checkpoint_filename=checkpoint_filename,
    )

    all_shortlisted_candidates: List[CandidateWindow] = []
    failed_or_empty_files: List[str] = []

    # Phase 1 & 2: Parsing, Timeline, and 3-Lane Pruning
    for fpath in selected_files:
        if is_cancelled and is_cancelled():
            logger.info("Pipeline cancelled during parsing phase.")
            break

        try:
            ep = parse_screenplay_file(fpath)
            if len(ep.all_events) == 0:
                logger.error(f"Episode file {fpath.name} has zero events.")
                failed_or_empty_files.append(fpath.name)
                continue

            summary.total_episodes_processed += 1
            summary.total_events_extracted += len(ep.all_events)

            beats = build_episode_timeline(ep)
            summary.total_beats_generated += len(beats)

            valid_windows = generate_all_valid_windows(beats, episode_num=ep.episode_num)
            summary.total_valid_windows_generated += len(valid_windows)

            shortlisted = prune_candidates_3_lane(valid_windows, target_count=target_candidates_per_ep)
            summary.total_shortlisted_candidates += len(shortlisted)

            all_shortlisted_candidates.extend(shortlisted)
            logger.info(
                f"Episode {ep.episode_num:02d}: {len(ep.all_events)} events -> "
                f"{len(beats)} beats -> {len(valid_windows)} valid windows -> "
                f"{len(shortlisted)} shortlisted"
            )

        except Exception as e:
            logger.error(f"Failed processing episode file {fpath.name}: {e}")
            failed_or_empty_files.append(fpath.name)

    if failed_or_empty_files:
        raise RuntimeError(
            f"Parsing failed or yielded zero events for file(s): {', '.join(failed_or_empty_files)}"
        )

    # Phase 3: Batch LLM Evaluation
    evaluated_records = evaluator.evaluate_candidates(
        all_shortlisted_candidates,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )
    summary.total_evaluated_success = sum(1 for r in evaluated_records if r.status == "evaluated")
    summary.total_evaluated_failed = sum(1 for r in evaluated_records if r.status == "evaluation_failed")

    if summary.total_evaluated_success == 0:
        raise RuntimeError("No candidate evaluated successfully. Halting before ranking or export.")

    # Phase 4: Global Ranking, Diversity Capping, and Export
    ranked_promos = rank_and_diversify_promos(evaluated_records)
    unrestricted_promos = rank_promos_unrestricted(evaluated_records)
    summary.total_ranked_promos = len(ranked_promos)

    # Export deliverables to disk
    export_promos_to_files(
        ranked_promos,
        evaluated_records,
        output_dir=output_dir,
        unrestricted_promos=unrestricted_promos,
    )

    summary.elapsed_seconds = round(time.time() - start_time, 2)

    logger.info("==================================================")
    logger.info("PIPELINE EXECUTION SUMMARY:")
    logger.info(f"  Episodes Processed:      {summary.total_episodes_processed}")
    logger.info(f"  Events Extracted:        {summary.total_events_extracted}")
    logger.info(f"  Atomic Beats Formed:     {summary.total_beats_generated}")
    logger.info(f"  Valid 20-60s Windows:    {summary.total_valid_windows_generated}")
    logger.info(f"  Shortlisted for LLM:     {summary.total_shortlisted_candidates}")
    logger.info(f"  Successfully Evaluated:  {summary.total_evaluated_success}")
    logger.info(f"  Failed Evaluations:      {summary.total_evaluated_failed}")
    logger.info(f"  Final Promos Ranked:     {summary.total_ranked_promos}")
    logger.info(f"  Total Runtime:           {summary.elapsed_seconds:.2f}s")
    logger.info("==================================================")

    return ranked_promos, summary


def main():
    """CLI Argument Parser."""
    parser = argparse.ArgumentParser(description="Kuku AI Generalist - Find the Promos Pipeline (OpenAI Powered)")
    parser.add_argument("--episode", type=int, default=None, help="Process a single episode (1-50)")
    parser.add_argument("--start-ep", type=int, default=None, help="Start of episode range")
    parser.add_argument("--end-ep", type=int, default=None, help="End of episode range")
    parser.add_argument("--all", action="store_true", help="Process all 50 episodes in the dataset")
    parser.add_argument("--audit", action="store_true", help="Run the automated 12-case benchmark audit")
    parser.add_argument("--model", type=str, default=OPENAI_DEFAULT_MODEL, help="OpenAI Model ID (preset or custom)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key override")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to screenplay files")
    parser.add_argument("--output-dir", type=str, default=str(OPENAI_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--checkpoint-file", type=str, default="eval_checkpoint.jsonl", help="Checkpoint filename")
    parser.add_argument("--target-per-ep", type=int, default=TARGET_CANDIDATES_PER_EPISODE, help="Target candidates per episode")

    args = parser.parse_args()

    try:
        # 1. Audit Mode
        if args.audit:
            llm_adapter = LLMClientAdapter(api_key=args.api_key, model=args.model)
            evaluator = PromoEvaluator(llm_client=llm_adapter, checkpoint_dir=Path(args.output_dir))
            report = run_benchmark_audit(evaluator)
            print(json.dumps(report, indent=2))
            return

        # 2. Episode Selection
        episode_numbers = None
        if args.episode is not None:
            episode_numbers = [args.episode]
        elif args.start_ep is not None and args.end_ep is not None:
            episode_numbers = list(range(args.start_ep, args.end_ep + 1))
        elif not args.all:
            logger.info("No episode range specified. Defaulting to Episode 1. Use --all for full 50-episode run.")
            episode_numbers = [1]

        run_promo_pipeline(
            episode_numbers=episode_numbers,
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            target_candidates_per_ep=args.target_per_ep,
            model=args.model,
            api_key=args.api_key,
            checkpoint_filename=args.checkpoint_file,
        )
    except MissingAPIKeyError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
