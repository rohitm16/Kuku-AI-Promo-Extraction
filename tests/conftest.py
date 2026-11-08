"""Test configuration and shared fixtures for Kuku promo extraction test suite."""

from __future__ import annotations

import json
import os
import re
from unittest.mock import MagicMock
import pytest

from kuku_promo_system.llm_client import LLMClientAdapter


def build_mock_adapter(model: str = "gpt-4o-mini", api_key: str = "sk-test-dummy-key-12345") -> LLMClientAdapter:
    """Build an LLMClientAdapter with a dummy key and a mocked OpenAI client returning valid JSON."""
    adapter = LLMClientAdapter(api_key=api_key, model=model)
    mock_client = MagicMock()

    def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")
                break

        # Check for single test ping or probe
        if kwargs.get("is_probe") or "Ping" in user_content:
            choice = MagicMock()
            choice.message.content = json.dumps({"status": "ok"})
            mock_resp = MagicMock()
            mock_resp.choices = [choice]
            mock_resp.usage.prompt_tokens = 5
            mock_resp.usage.completion_tokens = 5
            mock_resp.usage.total_tokens = 10
            return mock_resp

        candidate_ids = re.findall(r"Candidate ID:\s*([A-Za-z0-9_]+)", user_content)
        if not candidate_ids:
            # Generic JSON response
            choice = MagicMock()
            choice.message.content = json.dumps({"message": "test_mock_response"})
            mock_resp = MagicMock()
            mock_resp.choices = [choice]
            mock_resp.usage.prompt_tokens = 10
            mock_resp.usage.completion_tokens = 10
            mock_resp.usage.total_tokens = 20
            return mock_resp

        evals = []
        for cid in candidate_ids:
            evals.append({
                "candidate_id": cid,
                "hook_score": 8.0,
                "hook_type": "visual",
                "hook_description": "Opening hook description",
                "cliffhanger_score": 8.5,
                "cliffhanger_type": "scene_terminal",
                "cliffhanger_question": "Dramatic unresolved tension?",
                "is_immediately_resolved": False,
                "standalone_clarity_score": 7.5,
                "identified_conflict": "Central dramatic conflict",
                "requires_prior_lore": False,
                "editorial_rationale": "High dramatic tension and strong cliffhanger",
            })

        choice = MagicMock()
        choice.message.content = json.dumps({"evaluations": evals})
        mock_resp = MagicMock()
        mock_resp.choices = [choice]
        mock_resp.usage.prompt_tokens = 100 * len(candidate_ids)
        mock_resp.usage.completion_tokens = 50 * len(candidate_ids)
        mock_resp.usage.total_tokens = 150 * len(candidate_ids)
        return mock_resp

    mock_client.chat.completions.create.side_effect = mock_create
    adapter._client = mock_client
    return adapter


@pytest.fixture
def mock_adapter():
    """Fixture providing a mock adapter."""
    return build_mock_adapter()
