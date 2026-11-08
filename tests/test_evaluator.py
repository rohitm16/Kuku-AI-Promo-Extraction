"""Unit tests for the LLM Client Adapter and Hermetic Cold-Viewer Evaluator."""

import json
from pathlib import Path
import pytest

from kuku_promo_system.config import DEFAULT_OUTPUT_DIR
from kuku_promo_system.pruner import CandidateWindow
from kuku_promo_system.llm_client import LLMClientAdapter
from kuku_promo_system.evaluator import (
    SinglePromoEvaluation,
    BatchPromoEvaluation,
    EvaluatedCandidate,
    PromoEvaluator,
    build_batch_prompt,
)


def create_sample_candidate(candidate_id: str = "EP01_CAND_01") -> CandidateWindow:
    return CandidateWindow(
        candidate_id=candidate_id,
        episode_num=1,
        scene_start=1,
        scene_end=1,
        start_beat_id=1,
        end_beat_id=4,
        start_sec=11.0,
        end_sec=36.0,
        start_str="00:11.00",
        end_str="00:36.00",
        duration_sec=25.0,
        lane="boundary",
        window_text="MATRIARCH: You should turn around.\nJULIANA: Why?\nMATRIARCH: Gerald is not waiting for you.",
        lookahead_text="GERALD: Because I am marrying Clarissa.",
    )


def test_single_promo_evaluation_schema():
    """Verify validation of SinglePromoEvaluation model."""
    data = {
        "candidate_id": "EP01_CAND_01",
        "hook_score": 8.5,
        "hook_type": "audiovisual_sync",
        "hook_description": "Matriarch confrontation at cathedral doors",
        "cliffhanger_score": 9.0,
        "cliffhanger_type": "intra_scene",
        "cliffhanger_question": "Why is Gerald not waiting for Juliana?",
        "is_immediately_resolved": False,
        "standalone_clarity_score": 9.5,
        "identified_conflict": "Altar betrayal where bride is rejected for her sister",
        "requires_prior_lore": False,
        "editorial_rationale": "High-impact visceral betrayal that immediately engages scrollers.",
        "is_simulated": False,
    }
    ev = SinglePromoEvaluation.model_validate(data)
    assert ev.hook_score == 8.5
    assert ev.cliffhanger_type == "intra_scene"
    assert ev.is_immediately_resolved is False


def test_composite_scoring_and_penalties():
    """Verify composite scoring formula and penalty multipliers."""
    cand = create_sample_candidate()
    ev = SinglePromoEvaluation(
        candidate_id=cand.candidate_id,
        hook_score=10.0,
        hook_type="visual",
        hook_description="slap",
        cliffhanger_score=10.0,
        cliffhanger_type="scene_terminal",
        cliffhanger_question="question",
        is_immediately_resolved=False,
        standalone_clarity_score=10.0,
        identified_conflict="conflict",
        requires_prior_lore=False,
        editorial_rationale="rationale",
    )
    # Perfect score: 0.35*10 + 0.35*10 + 0.30*10 = 10.0
    rec = EvaluatedCandidate(candidate=cand, evaluation=ev, status="evaluated")
    assert rec.compute_composite_score() == 10.0

    # Test Lookahead Penalty: 30% reduction when immediately resolved
    ev_deflated = ev.model_copy(update={"is_immediately_resolved": True})
    rec_deflated = EvaluatedCandidate(candidate=cand, evaluation=ev_deflated, status="evaluated")
    assert rec_deflated.compute_composite_score() == 7.0

    # Test Lore Penalty: 50% reduction when requiring prior lore
    ev_lore = ev.model_copy(update={"requires_prior_lore": True})
    rec_lore = EvaluatedCandidate(candidate=cand, evaluation=ev_lore, status="evaluated")
    assert rec_lore.compute_composite_score() == 5.0

    # Test Combined Penalties: 10.0 * 0.70 * 0.50 = 3.50
    ev_both = ev.model_copy(update={"is_immediately_resolved": True, "requires_prior_lore": True})
    rec_both = EvaluatedCandidate(candidate=cand, evaluation=ev_both, status="evaluated")
    assert rec_both.compute_composite_score() == 3.50


def test_build_batch_prompt_isolation():
    """Verify that batch prompt properly isolates candidate text and includes lookahead."""
    c1 = create_sample_candidate("CAND_01")
    c2 = create_sample_candidate("CAND_02")
    prompt = build_batch_prompt([c1, c2])

    assert "CAND_01" in prompt
    assert "CAND_02" in prompt
    assert "NEXT-BEAT LOOKAHEAD CONTEXT" in prompt
    assert "MATRIARCH: You should turn around" in prompt


def test_mock_evaluator_and_checkpointing(tmp_path):
    """Verify evaluation and checkpoint serialization with mocked adapter."""
    from tests.conftest import build_mock_adapter
    adapter = build_mock_adapter()
    evaluator = PromoEvaluator(llm_client=adapter, checkpoint_dir=tmp_path, batch_size=2)

    c1 = create_sample_candidate("EP01_CAND_01")
    c2 = create_sample_candidate("EP01_CAND_02")

    results = evaluator.evaluate_candidates([c1, c2])
    assert len(results) == 2
    for r in results:
        assert r.status == "evaluated"
        assert r.evaluation is not None
        assert r.evaluation.is_simulated is False
        assert r.composite_score > 0.0

    # Verify checkpoint file exists on disk
    checkpoint_file = tmp_path / "eval_checkpoint.jsonl"
    assert checkpoint_file.exists()

    # Second run should load directly from cache without re-evaluating
    evaluator_cached = PromoEvaluator(llm_client=adapter, checkpoint_dir=tmp_path, batch_size=2)
    cached_results = evaluator_cached.evaluate_candidates([c1, c2])
    assert len(cached_results) == 2
    assert cached_results[0].candidate.candidate_id == "EP01_CAND_01"


def test_failure_handling_preserves_error():
    """Verify that failed candidates are recorded as evaluation_failed and score is 0.0."""
    cand = create_sample_candidate("FAIL_CAND")
    record = EvaluatedCandidate(
        candidate=cand,
        evaluation=None,
        status="evaluation_failed",
        error_message="HTTP 500: Server Error",
    )
    score = record.compute_composite_score()
    assert score == 0.0
    assert record.status == "evaluation_failed"
    assert "HTTP 500" in record.error_message


def test_checkpoint_not_reused_if_window_text_differs(tmp_path):
    """Verify that a checkpoint record with the same candidate ID but different window_text is not reused and client is called."""
    from tests.conftest import build_mock_adapter
    adapter1 = build_mock_adapter()
    evaluator1 = PromoEvaluator(llm_client=adapter1, checkpoint_dir=tmp_path, batch_size=2)

    c1 = create_sample_candidate("EP01_CAND_01")
    evaluator1.evaluate_candidates([c1])
    assert adapter1._client.chat.completions.create.call_count == 1

    # Incoming candidate has same candidate_id but different window_text
    c1_diff_text = c1.model_copy(update={"window_text": "DIFFERENT SHOW DIALOGUE TEXT"})

    adapter2 = build_mock_adapter()
    evaluator2 = PromoEvaluator(llm_client=adapter2, checkpoint_dir=tmp_path, batch_size=2)
    results = evaluator2.evaluate_candidates([c1_diff_text])

    assert len(results) == 1
    assert adapter2._client.chat.completions.create.call_count == 1
    assert results[0].candidate.window_text == "DIFFERENT SHOW DIALOGUE TEXT"
