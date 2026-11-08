"""Model configuration, credential management, and connection verification UI."""

from __future__ import annotations

import os
import streamlit as st
from typing import Tuple

from kuku_promo_system.config import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_VERIFIED_PRESETS,
    MAX_COMPLETION_TOKENS,
    OPENAI_MODEL_PRICING,
    mask_secret,
)
from kuku_promo_system.llm_client import LLMClientAdapter


def render_model_settings_sidebar() -> Tuple[str, str, int, str]:
    """Render the model settings section in the Streamlit sidebar.

    Returns:
        (api_key, selected_model_id, max_tokens, base_url)
    """
    st.sidebar.markdown("### 1. Model Provider")

    # 1. API Key Input (Masked Password field)
    # Check if a key was previously entered in session state or exists in env
    default_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))

    api_key_input = st.sidebar.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Key for the endpoint below. Stored in memory during this session.",
    )

    # Store strictly in session state (never written to disk or exported)
    st.session_state["openai_api_key"] = api_key_input.strip() if api_key_input else ""

    # API Base URL input
    default_base_url = os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
    base_url_input = st.sidebar.text_input(
        "API Base URL",
        value=default_base_url,
        help="Any OpenAI-compatible endpoint. Leave blank for OpenAI.",
    )
    base_url = base_url_input.strip()

    # 2. Model Selection: Verified Presets or Custom Model ID
    if "openrouter.ai" in base_url:
        model_options = (
            ["meta-llama/llama-3.3-70b-instruct"]
            + [f"openai/{m}" for m in OPENAI_VERIFIED_PRESETS]
            + ["Custom Model ID..."]
        )
    else:
        model_options = list(OPENAI_VERIFIED_PRESETS) + ["Custom Model ID..."]

    selected_preset = st.sidebar.selectbox(
        "Model",
        options=model_options,
        index=0,
        help="Select a standard verified model preset, or enter any custom model identifier.",
    )

    if selected_preset == "Custom Model ID...":
        custom_model = st.sidebar.text_input(
            "Enter Custom Model ID",
            value="",
            placeholder="e.g., gpt-4o-2024-08-06 or o3-mini",
            help="Enter any valid model slug.",
        )
        default_model = "meta-llama/llama-3.3-70b-instruct" if "openrouter.ai" in base_url else OPENAI_DEFAULT_MODEL
        selected_model = custom_model.strip() if custom_model.strip() else default_model
    else:
        selected_model = selected_preset

    # Display pricing note for selected model if known
    pricing = OPENAI_MODEL_PRICING.get(selected_model)
    if pricing:
        st.sidebar.caption(
            f"Pricing: \\${pricing['input_per_m']:.2f}/1M input &bull; \\${pricing['output_per_m']:.2f}/1M output tokens."
        )

    # 4. Token & Timeout Controls
    max_tokens = st.sidebar.slider(
        "Max Completion Tokens",
        min_value=500,
        max_value=3000,
        value=MAX_COMPLETION_TOKENS,
        step=100,
        help="Output token budget per batch (1500 is optimal for 3-candidate batches).",
    )

    # 4. "Test API Connection" Button
    if st.sidebar.button("Test API Connection", use_container_width=True):
        if not st.session_state["openai_api_key"]:
            st.sidebar.error("Please enter an API key before testing connection.")
        else:
            with st.sidebar.status("Connecting..."):
                adapter = LLMClientAdapter(
                    api_key=st.session_state["openai_api_key"],
                    model=selected_model,
                    base_url=base_url or None,
                    max_tokens=max_tokens,
                )
                success, msg, latency = adapter.verify_connection(selected_model)

            if success:
                st.sidebar.success(f"Connected to '{selected_model}' ({latency}s)")
            else:
                st.sidebar.error(f"Failed: {msg}")

    return st.session_state["openai_api_key"], selected_model, max_tokens, base_url
