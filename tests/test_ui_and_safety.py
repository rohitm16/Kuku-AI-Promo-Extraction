"""Unit and safety tests for UI service layer, Zip-Slip prevention, and cost estimation."""

import io
import zipfile
from pathlib import Path
import pytest

from ui.file_manager import ScreenplayFileManager, MAX_ZIP_EXTRACTED_BYTES
from ui.run_controller import estimate_pipeline_cost
from ui.results_view import get_available_datasets
from kuku_promo_system.config import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_MODEL_PRICING,
    TARGET_CANDIDATES_PER_EPISODE,
    DEFAULT_DATA_DIR,
)
from kuku_promo_system.evaluator import (
    PromoEvaluator,
    EvaluatedCandidate,
    SinglePromoEvaluation,
)
from kuku_promo_system.pruner import CandidateWindow
from kuku_promo_system.llm_client import LLMClientAdapter


def test_builtin_dataset_discovery():
    """Verify built-in dataset discovery locates canonical screenplays."""
    files = ScreenplayFileManager.get_builtin_dataset_files()
    if DEFAULT_DATA_DIR.exists():
        assert len(files) == 50
        assert all(f.name.startswith("episode-") for f in files)
    else:
        assert isinstance(files, list)


def test_screenplay_text_validation():
    """Verify validation of multi-track screenplay format."""
    # Empty content
    valid, err = ScreenplayFileManager.validate_screenplay_text("", "test.txt")
    assert not valid
    assert "empty" in err.lower()

    # Plain text without timestamps or dialogue markers
    valid, err = ScreenplayFileManager.validate_screenplay_text("Just some random text.", "test.txt")
    assert not valid
    assert "markers" in err.lower()

    # Valid multi-track screenplay sample
    sample = """
    Start: 00:00.00
    End: 00:30.00
    Scene 1 - Great Hall
    [CAMERA] Symmetrical framing on the door.
    JULIANA (Angry): "You lied to me!"
    """
    valid, err = ScreenplayFileManager.validate_screenplay_text(sample, "sample.txt")
    assert valid
    assert err is None


def test_zip_extraction_safe_files(tmp_path):
    """Verify safe extraction of valid screenplay text files from ZIP archive."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("episode-01.txt", "Start: 00:00.00\nEnd: 00:25.00\nScene 1\nJULIANA: Hello")
        zf.writestr("episode-02.txt", "Start: 00:00.00\nEnd: 00:25.00\nScene 1\nGERALD: Goodbye")
        zf.writestr("ignored.pdf", "%PDF-1.4 ignored")  # Non-txt should be skipped

    dest_dir = tmp_path / "extracted"
    dest_dir.mkdir()

    extracted, errors = ScreenplayFileManager.extract_zip_safely(zip_buffer.getvalue(), dest_dir)
    assert len(extracted) == 2
    assert (dest_dir / "episode-01.txt").exists()
    assert (dest_dir / "episode-02.txt").exists()
    assert not (dest_dir / "ignored.pdf").exists()
    assert len(errors) == 0


def test_zip_slip_path_traversal_prevention(tmp_path):
    """Verify malicious ZIP archive containing path traversal ('..') is blocked."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Malicious entry attempting to escape extraction directory
        zf.writestr("../../evil_escape.txt", "MALICIOUS PAYLOAD")
        zf.writestr("safe_file.txt", "Start: 00:00.00\nEnd: 00:25.00\nScene 1\nSafe content")

    dest_dir = tmp_path / "sandbox"
    dest_dir.mkdir()

    extracted, errors = ScreenplayFileManager.extract_zip_safely(zip_buffer.getvalue(), dest_dir)

    # Malicious file must NOT be extracted to parent directory
    escaped_file = tmp_path / "evil_escape.txt"
    assert not escaped_file.exists()
    assert any("Security Alert" in err for err in errors)

    # Safe file must be extracted
    assert len(extracted) == 1
    assert (dest_dir / "safe_file.txt").exists()


def test_cost_estimation_logic():
    """Verify cost calculation across different OpenAI models and candidate volumes."""
    # 50 episodes with 9 candidates/ep = 450 candidates -> 150 batches of 3
    est_mini = estimate_pipeline_cost(
        num_episodes=50,
        target_cands_per_ep=9,
        model_id="gpt-4o-mini",
    )
    assert est_mini["candidate_count"] == 450
    assert est_mini["batch_count"] == 150
    assert est_mini["prompt_tokens"] == 150 * 2300
    assert est_mini["completion_tokens"] == 150 * 650
    assert est_mini["total_tokens"] == (150 * 2300) + (150 * 650)
    # Cost with gpt-4o-mini ($0.15 / 1M prompt, $0.60 / 1M completion)
    expected_cost = (est_mini["prompt_tokens"] * 0.15 / 1e6) + (est_mini["completion_tokens"] * 0.60 / 1e6)
    assert pytest.approx(est_mini["estimated_cost_usd"], 0.0001) == expected_cost
    assert est_mini["estimated_cost_usd"] < 0.20  # Under 20 cents

    # Cost with gpt-4o ($2.50 / 1M prompt, $10.00 / 1M completion)
    est_4o = estimate_pipeline_cost(
        num_episodes=50,
        target_cands_per_ep=9,
        model_id="gpt-4o",
    )
    assert est_4o["estimated_cost_usd"] > est_mini["estimated_cost_usd"]
    # Ratio of cost should roughly match ~16x higher pricing
    assert pytest.approx(est_4o["estimated_cost_usd"] / est_mini["estimated_cost_usd"], 0.1) == 16.66


def test_model_aware_checkpoint_separation(tmp_path):
    """Verify evaluator checkpoint loader separates records by model_id."""
    cand = CandidateWindow(
        candidate_id="EP01_CAND_01_03",
        episode_num=1,
        scene_start=1,
        scene_end=1,
        start_beat_id=1,
        end_beat_id=3,
        start_sec=0.0,
        end_sec=25.0,
        start_str="00:00.00",
        end_str="00:25.00",
        duration_sec=25.0,
        lane="development",
        boundary_score=4.0,
        development_score=10.0,
        anomaly_score=5.0,
        prune_rank_score=10.0,
        window_text="Dialogue test",
        lookahead_text="Lookahead test",
        beats=[],
    )

    eval_gpt4o = EvaluatedCandidate(
        candidate=cand,
        evaluation=SinglePromoEvaluation(
            candidate_id="EP01_CAND_01_03",
            hook_score=8.5,
            hook_type="dialogue_punchline",
            hook_description="Sharp dialogue hook",
            cliffhanger_score=8.0,
            cliffhanger_type="intra_scene",
            cliffhanger_question="Will she leave?",
            is_immediately_resolved=False,
            standalone_clarity_score=8.5,
            identified_conflict="Conflict over marriage",
            requires_prior_lore=False,
            editorial_rationale="High quality promo",
            is_simulated=False,
            evaluation_source="openai",
            model_id="gpt-4o-mini",
        ),
        composite_score=8.35,
        status="evaluated",
    )

    # Save to checkpoint
    evaluator = PromoEvaluator(
        llm_client=LLMClientAdapter(api_key="sk-dummy", model="gpt-4o-mini"),
        checkpoint_dir=tmp_path,
        checkpoint_filename="test_checkpoint.jsonl",
    )
    evaluator.append_checkpoint(eval_gpt4o)

    # Loader with matching model_id "gpt-4o-mini" should find it
    matching_evaluator = PromoEvaluator(
        llm_client=LLMClientAdapter(api_key="sk-dummy", model="gpt-4o-mini"),
        checkpoint_dir=tmp_path,
        checkpoint_filename="test_checkpoint.jsonl",
    )
    loaded = matching_evaluator.load_checkpoints()
    assert "EP01_CAND_01_03" in loaded

    # Loader with DIFFERENT model_id "gpt-4o" should IGNORE the cached record
    different_evaluator = PromoEvaluator(
        llm_client=LLMClientAdapter(api_key="sk-dummy", model="gpt-4o"),
        checkpoint_dir=tmp_path,
        checkpoint_filename="test_checkpoint.jsonl",
    )
    loaded_diff = different_evaluator.load_checkpoints()
    assert "EP01_CAND_01_03" not in loaded_diff


def test_available_datasets_discovery():
    """Verify dataset discovery function finds outputs without exceptions."""
    datasets = get_available_datasets()
    assert isinstance(datasets, dict)
    for name, meta in datasets.items():
        assert "diversity_json" in meta
        assert "unrestricted_json" in meta
        assert "raw_evals" in meta
        assert "provider" in meta
