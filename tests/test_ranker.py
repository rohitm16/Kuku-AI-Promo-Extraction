"""Unit tests for Global Ranking, Diversity Deduplication, and File Export."""

import json
from pathlib import Path
import pytest

from kuku_promo_system.pruner import CandidateWindow
from kuku_promo_system.evaluator import SinglePromoEvaluation, EvaluatedCandidate
from kuku_promo_system.ranker import (
    deduplicate_intra_episode,
    rank_and_diversify_promos,
    export_promos_to_files,
    FinalRankedPromo,
)


def make_eval_candidate(
    cid: str, ep: int, start: float, end: float, score: float, hook: float = 8.0, cliff: float = 8.0
) -> EvaluatedCandidate:
    cand = CandidateWindow(
        candidate_id=cid,
        episode_num=ep,
        scene_start=1,
        scene_end=1,
        start_beat_id=1,
        end_beat_id=3,
        start_sec=start,
        end_sec=end,
        start_str=f"00:{int(start):02d}.00",
        end_str=f"00:{int(end):02d}.00",
        duration_sec=end - start,
        lane="boundary",
        window_text="sample",
    )
    ev = SinglePromoEvaluation(
        candidate_id=cid,
        hook_score=hook,
        hook_type="audiovisual_sync",
        hook_description="hook",
        cliffhanger_score=cliff,
        cliffhanger_type="intra_scene",
        cliffhanger_question="question",
        is_immediately_resolved=False,
        standalone_clarity_score=9.0,
        identified_conflict="conflict",
        requires_prior_lore=False,
        editorial_rationale="rationale",
    )
    rec = EvaluatedCandidate(candidate=cand, evaluation=ev, status="evaluated", composite_score=score)
    return rec


def test_intra_episode_deduplication():
    """Verify that overlapping candidates in the same episode are deduplicated."""
    # Two candidates in Ep 1 with high overlap: [10, 40] and [12, 42]
    c1 = make_eval_candidate("C1", ep=1, start=10.0, end=40.0, score=9.0)
    c2 = make_eval_candidate("C2", ep=1, start=12.0, end=42.0, score=7.5)
    # Non-overlapping candidate in Ep 1: [50, 80]
    c3 = make_eval_candidate("C3", ep=1, start=50.0, end=80.0, score=8.0)

    deduped = deduplicate_intra_episode([c1, c2, c3], iou_thresh=0.50)
    assert len(deduped) == 2
    # C1 should be kept (higher score), C2 suppressed
    kept_ids = [c.candidate.candidate_id for c in deduped]
    assert "C1" in kept_ids
    assert "C2" not in kept_ids
    assert "C3" in kept_ids


def test_diversity_guardrail_top_10():
    """Verify that Top 10 limits any single episode to max 1 slot."""
    candidates = []
    # Create 5 strong candidates for Episode 1 (scores 9.5 down to 9.1)
    for i in range(5):
        candidates.append(make_eval_candidate(f"EP01_{i}", ep=1, start=i * 50, end=i * 50 + 30, score=9.5 - (i * 0.1)))

    # Create candidates for Episodes 2 through 10 (scores 8.5)
    for ep in range(2, 11):
        candidates.append(make_eval_candidate(f"EP{ep:02d}_01", ep=ep, start=10.0, end=40.0, score=8.5))

    ranked = rank_and_diversify_promos(candidates, max_top_10_per_ep=1)
    top_10 = ranked[:10]

    # Exactly 1 promo from Episode 1 in Top 10
    ep1_in_top_10 = [p for p in top_10 if p.episode_num == 1]
    assert len(ep1_in_top_10) == 1, f"Expected 1 Ep 1 promo in Top 10, got {len(ep1_in_top_10)}"

    # Top 10 should contain diverse episodes
    episodes_in_top_10 = set(p.episode_num for p in top_10)
    assert len(episodes_in_top_10) == 10, "Top 10 should have 10 distinct episodes"


def test_export_promos_to_files(tmp_path):
    """Verify JSON, CSV, and Raw export files."""
    c1 = make_eval_candidate("C1", ep=1, start=10.0, end=40.0, score=9.0)
    ranked = rank_and_diversify_promos([c1])
    assert len(ranked) == 1

    json_p, csv_p, raw_p = export_promos_to_files(ranked, [c1], output_dir=tmp_path)
    assert json_p.exists()
    assert csv_p.exists()
    assert raw_p.exists()

    # Check JSON content
    with open(json_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["rank"] == 1
    assert data[0]["promo_id"] == "PROMO_001_C1"
    assert data[0]["duration_sec"] == 30.0
    assert len(data[0]["segments"]) == 1

    # Check CSV content
    csv_text = csv_p.read_text(encoding="utf-8")
    assert "PROMO_001_C1" in csv_text
    assert "Rank,Promo ID,Episode" in csv_text
