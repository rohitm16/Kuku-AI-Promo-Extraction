"""Global ranking, diversity-aware deduplication, and export module.

Ranks evaluated candidate promos across episodes, applies release diversification
guardrails, and exports structured JSON and CSV outputs conforming to the assignment brief.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from kuku_promo_system.config import DEFAULT_OUTPUT_DIR
from kuku_promo_system.logger import logger
from kuku_promo_system.pruner import compute_temporal_iou
from kuku_promo_system.evaluator import EvaluatedCandidate


class FinalRankedPromo(BaseModel):
    """Deliverable schema for an officially ranked promotional clip."""
    rank: int
    promo_id: str
    episode_num: int
    start_time: str
    end_time: str
    duration_sec: float
    composite_score: float
    hook: Dict[str, Any]
    cliffhanger: Dict[str, Any]
    standalone: Dict[str, Any]
    editorial_rationale: str
    segments: List[Dict[str, Any]]
    telemetry: Dict[str, Any]


def deduplicate_intra_episode(candidates: List[EvaluatedCandidate], iou_thresh: float = 0.50) -> List[EvaluatedCandidate]:
    """Suppress overlapping candidates within the same episode."""
    # Group by episode
    by_ep: Dict[int, List[EvaluatedCandidate]] = {}
    for c in candidates:
        by_ep.setdefault(c.candidate.episode_num, []).append(c)

    deduped: List[EvaluatedCandidate] = []
    for ep_num, ep_cands in by_ep.items():
        # Sort descending by composite_score
        sorted_ep = sorted(ep_cands, key=lambda c: c.composite_score, reverse=True)
        kept_ep: List[EvaluatedCandidate] = []
        for cand in sorted_ep:
            is_dup = False
            for kept in kept_ep:
                if compute_temporal_iou(cand.candidate, kept.candidate) > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept_ep.append(cand)
        deduped.extend(kept_ep)

    return deduped


def rank_promos_unrestricted(
    candidates: List[EvaluatedCandidate],
) -> List[FinalRankedPromo]:
    """Rank promos globally strictly by composite score without diversity caps."""
    valid_candidates = [c for c in candidates if c.status == "evaluated" and c.evaluation is not None]
    if not valid_candidates:
        return []

    # 1. Deduplicate temporally within each episode
    deduped = deduplicate_intra_episode(valid_candidates, iou_thresh=0.50)

    # 2. Sort descending by composite score, tie-breaking by hook_score then cliffhanger_score
    sorted_all = sorted(
        deduped,
        key=lambda c: (
            c.composite_score,
            c.evaluation.hook_score if c.evaluation else 0,
            c.evaluation.cliffhanger_score if c.evaluation else 0,
        ),
        reverse=True,
    )

    ranked_promos: List[FinalRankedPromo] = []
    for idx, c in enumerate(sorted_all, 1):
        promo = _build_ranked_promo_record(c, idx)
        ranked_promos.append(promo)

    return ranked_promos


def rank_and_diversify_promos(
    candidates: List[EvaluatedCandidate],
    max_top_10_per_ep: int = 1,
    max_top_25_per_ep: int = 2,
) -> List[FinalRankedPromo]:
    """Rank promos globally while enforcing diversity guardrails across episodes."""
    valid_candidates = [c for c in candidates if c.status == "evaluated" and c.evaluation is not None]
    if not valid_candidates:
        logger.warning("No evaluated candidates available for ranking.")
        return []

    # 1. Deduplicate temporally within each episode
    deduped = deduplicate_intra_episode(valid_candidates, iou_thresh=0.50)

    # 2. Sort descending by composite score, tie-breaking by hook_score then cliffhanger_score
    sorted_all = sorted(
        deduped,
        key=lambda c: (
            c.composite_score,
            c.evaluation.hook_score if c.evaluation else 0,
            c.evaluation.cliffhanger_score if c.evaluation else 0,
        ),
        reverse=True,
    )

    ranked_promos: List[FinalRankedPromo] = []
    ep_counts: Dict[int, int] = {}
    deferred: List[EvaluatedCandidate] = []

    # Phase 1: Build Top 10 with strict per-episode cap (max 1)
    for c in sorted_all:
        ep = c.candidate.episode_num
        curr_count = ep_counts.get(ep, 0)

        if len(ranked_promos) < 10:
            if curr_count < max_top_10_per_ep:
                rank = len(ranked_promos) + 1
                promo = _build_ranked_promo_record(c, rank)
                ranked_promos.append(promo)
                ep_counts[ep] = curr_count + 1
            else:
                deferred.append(c)
        elif len(ranked_promos) < 25:
            # Phase 2: Top 11 to 25 with relaxed cap (max 2)
            if curr_count < max_top_25_per_ep:
                rank = len(ranked_promos) + 1
                promo = _build_ranked_promo_record(c, rank)
                ranked_promos.append(promo)
                ep_counts[ep] = curr_count + 1
            else:
                deferred.append(c)
        else:
            # Phase 3: Remaining slots filled in order
            rank = len(ranked_promos) + 1
            promo = _build_ranked_promo_record(c, rank)
            ranked_promos.append(promo)
            ep_counts[ep] = curr_count + 1

    # Append deferred candidates that were capped earlier
    for c in deferred:
        rank = len(ranked_promos) + 1
        promo = _build_ranked_promo_record(c, rank)
        ranked_promos.append(promo)

    return ranked_promos


def _build_ranked_promo_record(rec: EvaluatedCandidate, rank: int) -> FinalRankedPromo:
    """Helper to assemble a FinalRankedPromo object."""
    cand = rec.candidate
    ev = rec.evaluation

    return FinalRankedPromo(
        rank=rank,
        promo_id=f"PROMO_{rank:03d}_{cand.candidate_id}",
        episode_num=cand.episode_num,
        start_time=cand.start_str,
        end_time=cand.end_str,
        duration_sec=cand.duration_sec,
        composite_score=rec.composite_score,
        hook={
            "score": ev.hook_score if ev else 0.0,
            "type": ev.hook_type if ev else "unknown",
            "description": ev.hook_description if ev else "",
        },
        cliffhanger={
            "score": ev.cliffhanger_score if ev else 0.0,
            "type": ev.cliffhanger_type if ev else "unknown",
            "unresolved_question": ev.cliffhanger_question if ev else "",
            "is_immediately_resolved": ev.is_immediately_resolved if ev else False,
        },
        standalone={
            "score": ev.standalone_clarity_score if ev else 0.0,
            "conflict": ev.identified_conflict if ev else "",
            "requires_prior_lore": ev.requires_prior_lore if ev else False,
        },
        editorial_rationale=ev.editorial_rationale if ev else "",
        segments=[
            {
                "segment_index": 1,
                "episode": cand.episode_num,
                "start_time": cand.start_str,
                "end_time": cand.end_str,
                "duration_sec": cand.duration_sec,
            }
        ],
        telemetry={
            "pruning_lane": cand.lane,
            "boundary_score": cand.boundary_score,
            "development_score": cand.development_score,
            "anomaly_score": cand.anomaly_score,
            "is_simulated": ev.is_simulated if ev else False,
        },
    )


def export_promos_to_files(
    ranked_promos: List[FinalRankedPromo],
    raw_candidates: List[EvaluatedCandidate],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    unrestricted_promos: Optional[List[FinalRankedPromo]] = None,
) -> Tuple[Path, Path, Path]:
    """Export ranked promos to structured JSON, flattened CSV, and raw evaluations JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "ranked_promos.json"
    csv_path = out_path / "ranked_promos.csv"
    raw_path = out_path / "raw_evaluations.json"

    # 1. Export Clean JSON (Diversity-adjusted)
    promos_dict = [p.model_dump() for p in ranked_promos]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(promos_dict, f, indent=2)

    # 2. Export Flattened CSV (Diversity-adjusted)
    _write_csv(csv_path, ranked_promos)

    # 3. Export Raw Evaluations (Preserving raw provenance)
    raw_dict = [c.model_dump() for c in raw_candidates]
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_dict, f, indent=2)

    # 4. Export Unrestricted Rankings if provided
    if unrestricted_promos:
        unres_json_path = out_path / "ranked_promos_unrestricted.json"
        unres_csv_path = out_path / "ranked_promos_unrestricted.csv"
        with open(unres_json_path, "w", encoding="utf-8") as f:
            json.dump([p.model_dump() for p in unrestricted_promos], f, indent=2)
        _write_csv(unres_csv_path, unrestricted_promos)
        logger.info(f"  Unrestricted JSON: {unres_json_path}")
        logger.info(f"  Unrestricted CSV:  {unres_csv_path}")

    logger.info(f"Successfully exported {len(ranked_promos)} ranked promos to:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV:  {csv_path}")
    logger.info(f"  RAW:  {raw_path}")

    return json_path, csv_path, raw_path


def _write_csv(csv_path: Path, promos: List[FinalRankedPromo]):
    """Helper to write promo list to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank",
            "Promo ID",
            "Episode",
            "Start Time",
            "End Time",
            "Duration (s)",
            "Composite Score",
            "Hook Score",
            "Hook Type",
            "Hook Description",
            "Cliffhanger Score",
            "Cliffhanger Type",
            "Cliffhanger Question",
            "Immediately Resolved",
            "Clarity Score",
            "Identified Conflict",
            "Requires Lore",
            "Editorial Rationale",
            "Lane",
            "Simulated",
        ])

        for p in promos:
            writer.writerow([
                p.rank,
                p.promo_id,
                p.episode_num,
                p.start_time,
                p.end_time,
                p.duration_sec,
                p.composite_score,
                p.hook["score"],
                p.hook["type"],
                p.hook["description"],
                p.cliffhanger["score"],
                p.cliffhanger["type"],
                p.cliffhanger["unresolved_question"],
                p.cliffhanger["is_immediately_resolved"],
                p.standalone["score"],
                p.standalone["conflict"],
                p.standalone["requires_prior_lore"],
                p.editorial_rationale,
                p.telemetry["pruning_lane"],
                p.telemetry["is_simulated"],
            ])
