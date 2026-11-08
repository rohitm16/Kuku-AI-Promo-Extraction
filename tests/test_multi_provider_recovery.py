"""Unit tests for Official OpenAI Client Integration, Error Handling, and Checkpoint Recovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from kuku_promo_system.config import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_VERIFIED_PRESETS,
    MAX_COMPLETION_TOKENS,
    FINAL_RANKED_PROMOS_JSON,
)
from kuku_promo_system.llm_client import LLMClientAdapter, MissingAPIKeyError, is_reasoning_model, infer_provider
from kuku_promo_system.evaluator import (
    SinglePromoEvaluation,
    BatchPromoEvaluation,
    EvaluatedCandidate,
    PromoEvaluator,
)
from kuku_promo_system.pruner import CandidateWindow
from openai import (
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)


class DummyModel(BaseModel):
    message: str


def test_openai_configuration_resolution():
    """Verify OpenAI defaults automatically resolve without manual override."""
    adapter = LLMClientAdapter(api_key="test-key-mock")
    assert adapter.provider == "openai"
    assert adapter.model == OPENAI_DEFAULT_MODEL
    assert adapter.max_tokens == MAX_COMPLETION_TOKENS

    # Test custom model ID
    custom_adapter = LLMClientAdapter(model="gpt-4o-custom", api_key="test-key-mock")
    assert custom_adapter.model == "gpt-4o-custom"


def test_missing_api_keys_raises_missing_api_key_error():
    """Verify that empty or placeholder credentials raise MissingAPIKeyError."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "", "LLM_API_KEY": ""}, clear=True):
        with pytest.raises(MissingAPIKeyError):
            LLMClientAdapter(api_key="")


def test_output_token_limit_configured():
    """Verify max_tokens defaults to 1500 to prevent context reservation runaway."""
    adapter = LLMClientAdapter(api_key="test-key")
    assert adapter.max_tokens == 1500


def test_http_401_auth_circuit_breaker():
    """Verify that AuthenticationError triggers immediate circuit breaker with zero retries."""
    adapter = LLMClientAdapter(api_key="invalid-test-key")
    
    mock_client = MagicMock()
    ae = AuthenticationError(
        message="Incorrect API key provided",
        response=MagicMock(),
        body={"error": {"code": "invalid_api_key"}},
    )
    mock_client.chat.completions.create.side_effect = ae
    adapter._client = mock_client

    messages = [{"role": "user", "content": "test"}]
    res, is_sim, err, usage, lat = adapter.generate_structured_completion(
        messages=messages,
        response_model=DummyModel,
    )

    assert res is None
    assert is_sim is False
    assert adapter._circuit_broken is True
    assert "CIRCUIT BREAKER TRIPPED" in err
    assert mock_client.chat.completions.create.call_count == 1

    # Subsequent call should immediately fail via circuit breaker without hitting API
    res2, is_sim2, err2, _, _ = adapter.generate_structured_completion(messages, DummyModel)
    assert res2 is None
    assert "CIRCUIT BREAKER ACTIVE" in err2
    assert mock_client.chat.completions.create.call_count == 1


def test_insufficient_quota_circuit_breaker():
    """Verify that insufficient_quota rate limit error trips circuit breaker immediately."""
    adapter = LLMClientAdapter(api_key="real-test-key")
    mock_client = MagicMock()

    rle = RateLimitError(
        message="You exceeded your current quota, please check your plan and billing details.",
        response=MagicMock(),
        body={"error": {"code": "insufficient_quota"}},
    )
    mock_client.chat.completions.create.side_effect = rle
    adapter._client = mock_client

    messages = [{"role": "user", "content": "test"}]
    res, is_sim, err, usage, lat = adapter.generate_structured_completion(messages, DummyModel)

    assert res is None
    assert adapter._circuit_broken is True
    assert "CIRCUIT BREAKER TRIPPED" in err
    assert mock_client.chat.completions.create.call_count == 1


def test_retry_limits_on_server_error():
    """Verify server error retries are bounded and do not exceed attempt limit."""
    adapter = LLMClientAdapter(api_key="real-test-key")
    mock_client = MagicMock()
    
    ae = APIError(
        message="Internal server error",
        request=MagicMock(),
        body={"error": {"code": 500}},
    )
    ae.status_code = 500
    mock_client.chat.completions.create.side_effect = ae
    adapter._client = mock_client

    with patch("time.sleep", return_value=None):
        res, is_sim, err, usage, lat = adapter.generate_structured_completion(
            messages=[{"role": "user", "content": "test"}],
            response_model=DummyModel,
        )

    assert res is None
    assert mock_client.chat.completions.create.call_count == 3  # Initial + 2 retries = 3 attempts total


def test_model_aware_checkpoint_compatibility(tmp_path):
    """Verify that load_checkpoints() strictly respects model compatibility."""
    ckpt_file = tmp_path / "test_eval_ckpt.jsonl"
    
    cand1 = CandidateWindow(
        candidate_id="CAND_GPT4O_MINI", episode_num=1, scene_start=1, scene_end=1,
        start_beat_id=1, end_beat_id=2, start_sec=0.0, end_sec=25.0, start_str="00:00.00",
        end_str="00:25.00", duration_sec=25.0, lane="boundary", window_text="ok"
    )
    ev1 = SinglePromoEvaluation(
        candidate_id="CAND_GPT4O_MINI", hook_score=8.0, hook_type="visual", hook_description="h",
        cliffhanger_score=8.0, cliffhanger_type="scene_terminal", cliffhanger_question="q",
        is_immediately_resolved=False, standalone_clarity_score=8.0, identified_conflict="c",
        requires_prior_lore=False, editorial_rationale="e", is_simulated=False,
        evaluation_source="openai", model_id="gpt-4o-mini"
    )
    rec1 = EvaluatedCandidate(candidate=cand1, evaluation=ev1, status="evaluated", composite_score=8.0)

    # Different model (gpt-4o)
    cand2 = CandidateWindow(
        candidate_id="CAND_DIFF_MODEL", episode_num=1, scene_start=1, scene_end=1,
        start_beat_id=3, end_beat_id=4, start_sec=25.0, end_sec=50.0, start_str="00:25.00",
        end_str="00:50.00", duration_sec=25.0, lane="boundary", window_text="err"
    )
    ev2 = SinglePromoEvaluation(
        candidate_id="CAND_DIFF_MODEL", hook_score=9.0, hook_type="visual", hook_description="h",
        cliffhanger_score=9.0, cliffhanger_type="scene_terminal", cliffhanger_question="q",
        is_immediately_resolved=False, standalone_clarity_score=9.0, identified_conflict="c",
        requires_prior_lore=False, editorial_rationale="e", is_simulated=False,
        evaluation_source="openai", model_id="gpt-4o"
    )
    rec2 = EvaluatedCandidate(candidate=cand2, evaluation=ev2, status="evaluated", composite_score=9.0)

    with open(ckpt_file, "w", encoding="utf-8") as f:
        f.write(rec1.model_dump_json() + "\n")
        f.write(rec2.model_dump_json() + "\n")

    # Load with active model = gpt-4o-mini
    adapter = LLMClientAdapter(api_key="real-test-key", model="gpt-4o-mini")
    evaluator = PromoEvaluator(llm_client=adapter, checkpoint_dir=tmp_path, checkpoint_filename="test_eval_ckpt.jsonl")
    
    cached = evaluator.load_checkpoints()
    assert len(cached) == 1
    assert "CAND_GPT4O_MINI" in cached
    # CAND_DIFF_MODEL must be ignored because model_id does not match!
    assert "CAND_DIFF_MODEL" not in cached


@pytest.mark.skipif(not Path("raw_outputs/eval_checkpoint.jsonl").exists(), reason="raw_outputs/eval_checkpoint.jsonl absent")
def test_preservation_of_historical_live_evaluations():
    """Verify production checkpoint holds all 463 genuine live evaluations and backup preserves historical 432/31 state."""
    # 1. Verify backup preserves the 432 / 31 state from Phase 6
    backup_ckpt = Path("raw_outputs/pre_phase9_backup/eval_checkpoint.jsonl")
    if backup_ckpt.exists():
        backup_records = [json.loads(line) for line in open(backup_ckpt, "r", encoding="utf-8") if line.strip()]
        b_eval = [r for r in backup_records if r.get("status") == "evaluated"]
        b_fail = [r for r in backup_records if r.get("status") == "evaluation_failed"]
        assert len(b_eval) == 432
        assert len(b_fail) == 31

    # 2. Verify production checkpoint holds all 463 completed live evaluations
    prod_ckpt = Path("raw_outputs/eval_checkpoint.jsonl")
    assert prod_ckpt.exists(), "Production checkpoint must exist"

    records = []
    with open(prod_ckpt, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 463, f"Expected 463 records, found {len(records)}"
    
    evaluated = [r for r in records if r.get("status") == "evaluated"]
    failed = [r for r in records if r.get("status") == "evaluation_failed"]
    
    assert len(evaluated) == 463, f"Expected 463 evaluated, got {len(evaluated)}"
    assert len(failed) == 0, f"Expected 0 failed, got {len(failed)}"

    # Check that all 463 are genuine live evaluations preserving original OpenRouter provenance
    for r in evaluated:
        ev = r["evaluation"]
        assert ev["is_simulated"] is False
        assert ev["evaluation_source"] == "openrouter"
        assert ev["model_id"] == "meta-llama/llama-3.3-70b-instruct"
        assert 20.0 <= r["candidate"]["duration_sec"] <= 60.0


def test_openai_request_parameter_compatibility_reasoning_vs_standard():
    """Verify request parameters: reasoning models omit temperature and require max_completion_tokens."""
    from kuku_promo_system.llm_client import is_reasoning_model, sanitize_error_text

    # 1. Reasoning model detection
    assert is_reasoning_model("o3-mini") is True
    assert is_reasoning_model("o1-mini") is True
    assert is_reasoning_model("o1") is True
    assert is_reasoning_model("gpt-4o-mini") is False
    assert is_reasoning_model("gpt-4o") is False
    assert is_reasoning_model("gpt-4-turbo") is False

    adapter = LLMClientAdapter(api_key="test-key", model="gpt-4o-mini")
    messages = [{"role": "user", "content": "ping"}]

    # 2. Standard model request kwargs (gpt-4o-mini)
    std_kwargs = adapter._build_request_kwargs("gpt-4o-mini", messages, max_tokens=1500, temperature=0.0)
    assert std_kwargs["max_completion_tokens"] == 1500
    assert std_kwargs["temperature"] == 0.0
    assert std_kwargs["response_format"] == {"type": "json_object"}

    # 3. Reasoning model request kwargs (o3-mini)
    reasoning_kwargs = adapter._build_request_kwargs("o3-mini", messages, max_tokens=1500, temperature=0.0)
    assert reasoning_kwargs["max_completion_tokens"] == 1500
    # Must omit temperature because o1/o3 reject temperature != default
    assert "temperature" not in reasoning_kwargs
    assert reasoning_kwargs["response_format"] == {"type": "json_object"}

    # 4. Probe kwargs for connection testing
    probe_std = adapter._build_request_kwargs("gpt-4o-mini", messages, max_tokens=1, is_probe=True)
    assert probe_std["max_completion_tokens"] == 1
    assert "response_format" not in probe_std

    probe_reasoning = adapter._build_request_kwargs("o3-mini", messages, max_tokens=1, is_probe=True)
    assert probe_reasoning["max_completion_tokens"] >= 25  # Budget allocated for reasoning tokens


def test_custom_model_id_propagation(tmp_path):
    """Verify that custom model IDs cleanly propagate to evaluations and checkpoints."""
    from tests.conftest import build_mock_adapter
    custom_model = "ft:gpt-4o-mini:kuku-corp:custom-eval-v2"
    adapter = build_mock_adapter(model=custom_model)
    assert adapter.model == custom_model

    evaluator = PromoEvaluator(llm_client=adapter, checkpoint_dir=tmp_path)
    cand = CandidateWindow(
        candidate_id="CUSTOM_TEST_01", episode_num=1, scene_start=1, scene_end=1,
        start_beat_id=1, end_beat_id=2, start_sec=0.0, end_sec=25.0, start_str="00:00.00",
        end_str="00:25.00", duration_sec=25.0, lane="boundary", window_text="dialogue"
    )

    recs = evaluator.evaluate_candidates([cand])
    assert len(recs) == 1
    assert recs[0].evaluation.model_id == custom_model


def test_sanitization_of_api_keys_in_error_messages():
    """Verify that error sanitization redacts real keys and secret patterns."""
    from kuku_promo_system.llm_client import sanitize_error_text

    secret_key = "sk-proj-test1234567890abcdefghijklmnopqrstuvwxyz"
    raw_error = f"Error connecting with key {secret_key}: connection refused"
    clean_error = sanitize_error_text(raw_error, secret_key)

    assert secret_key not in clean_error
    assert "sk-" in clean_error
    assert "..." in clean_error or "[REDACTED]" in clean_error or "***" in clean_error


@pytest.mark.skipif(not FINAL_RANKED_PROMOS_JSON.exists(), reason="FINAL_RANKED_PROMOS_JSON absent")
def test_export_isolation_preserves_historical_files():
    """Verify that OPENAI_OUTPUT_DIR is isolated from RAW_OUTPUTS_DIR."""
    from kuku_promo_system.config import RAW_OUTPUTS_DIR, OPENAI_OUTPUT_DIR, FINAL_RANKED_PROMOS_JSON

    assert OPENAI_OUTPUT_DIR != RAW_OUTPUTS_DIR
    assert OPENAI_OUTPUT_DIR.parent == RAW_OUTPUTS_DIR
    assert FINAL_RANKED_PROMOS_JSON.parent == RAW_OUTPUTS_DIR
    assert FINAL_RANKED_PROMOS_JSON.exists()


def test_infer_provider():
    """Verify infer_provider returns correct provider for None, openai, openrouter, and another URL."""
    assert infer_provider(None) == "openai"
    assert infer_provider("https://api.openai.com/v1") == "openai"
    assert infer_provider("https://openrouter.ai/api/v1") == "openrouter"
    assert infer_provider("https://api.together.xyz/v1") == "custom"


def test_openrouter_adapter_kwargs():
    """Verify adapter with openrouter.ai base_url sets provider and uses max_tokens in kwargs."""
    adapter = LLMClientAdapter(
        api_key="test-dummy-key",
        base_url="https://openrouter.ai/api/v1",
    )
    assert adapter.provider == "openrouter"
    messages = [{"role": "user", "content": "Ping"}]
    kwargs = adapter._build_request_kwargs(
        target_model="meta-llama/llama-3.3-70b-instruct",
        messages=messages,
        max_tokens=500,
    )
    assert "max_tokens" in kwargs
    assert kwargs["max_tokens"] == 500
    assert "max_completion_tokens" not in kwargs


def test_is_reasoning_model_with_provider_prefix():
    """Verify is_reasoning_model handles model names with provider prefixes."""
    assert is_reasoning_model("openai/o3-mini") is True
    assert is_reasoning_model("meta-llama/llama-3.3-70b-instruct") is False
