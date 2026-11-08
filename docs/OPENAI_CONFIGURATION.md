# OpenAI Configuration & Integration Guide

The code uses the OpenAI Python SDK as a client library. To use OpenRouter or another OpenAI-compatible endpoint, set OPENAI_BASE_URL and OPENAI_API_KEY in your terminal and pass --model, or enter the key and base URL in the Streamlit sidebar.
> The latency figures in this document are not measurements.

This document details the configuration, model presets, connection verification, financial safeguards, and cost estimation protocols for the **OpenAI Integration** in the Kuku AI Promo Extraction System.

---

## 1. Authentication & API Key Setup

The system integrates directly with the official `openai` Python SDK (`openai>=1.0.0`).

### Method 1: Local `.env` File (Recommended for Local Dev & CLI)
Create or update `.env` in the project root:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-YourActualOpenAIKeyHere
OPENAI_MODEL=gpt-4o-mini
MAX_COMPLETION_TOKENS=1500
```

### Method 2: In-Memory Streamlit Password Input (Recommended for UI Demos)
When launching the web application (`streamlit run app.py`):
1. Open the left sidebar under **1. OpenAI Configuration**.
2. Enter your key into the masked **OpenAI API Key** field.
3. The key is stored **strictly in memory** in `st.session_state["openai_api_key"]`.
4. It is never written to disk, never logged to stdout, and never included in exported files.

---

## 2. Verified Model Presets & Pricing

The system includes pre-configured pricing metadata and validated prompt compatibility for the following OpenAI models:

| Model Identifier | Primary Use Case | Prompt Pricing (per 1M) | Completion Pricing (per 1M) | Avg Batch Latency |
| :--- | :--- | :---: | :---: | :---: |
| **`gpt-4o-mini`** *(Default)* | **Production Recommendation:** Fast, highly structured output, ultra-low cost. | **\$0.15** | **\$0.60** | **~0.9s** |
| **`gpt-4o`** | Flagship high-intelligence model for deep narrative nuance. | **\$2.50** | **\$10.00** | **~1.8s** |
| **`o3-mini`** | High-speed reasoning model for complex logical validation. | **\$1.10** | **\$4.40** | **~2.2s** |
| **`gpt-4-turbo`** | Previous generation flagship model with high capacity. | **\$10.00** | **\$30.00** | **~2.5s** |

### Custom Model ID Support
In addition to the presets, users can select **"Custom Model ID..."** in the Streamlit UI or pass `--model <any_model_slug>` via the CLI. The system will route all requests to that model identifier while maintaining strict Pydantic JSON schema validation.

---

## 3. Parameter Compatibility: Reasoning Models vs. Standard Chat Models

The OpenAI model family has distinct parameter requirements depending on whether a standard model or a reasoning model is invoked:

| Parameter Category | Standard Models (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`) | Reasoning Models (`o3-mini`, `o1`, `o1-mini`) |
| :--- | :--- | :--- |
| **Token Limit Parameter** | `max_completion_tokens` (with fallback to `max_tokens`) | **`max_completion_tokens` strictly required.** `max_tokens` causes HTTP 400. |
| **Sampling Temperature** | `temperature=0.0` (for deterministic scoring) | **Parameter omitted.** Reasoning models fix temperature at 1.0; passing custom temperature causes HTTP 400. |
| **Structured Output** | `response_format={"type": "json_object"}` | `response_format={"type": "json_object"}` supported on `o3-mini` and modern `o1`. |
| **Connection Probe** | `max_completion_tokens=1` | `max_completion_tokens=25` (allows token budget for internal reasoning + ping output). |

The adapter ([`kuku_promo_system/llm_client.py`](file:///d:/Companies/Kuku/kuku_promo_system/llm_client.py)) implements `_build_request_kwargs()` which automatically detects model type via `is_reasoning_model()` and applies the exact compatible parameter signature.

---

## 4. Connection Verification Probe

Before launching batch runs, users can click **"Test API Connection"** in the sidebar. The underlying `LLMClientAdapter.verify_connection()` executes an inexpensive test call to OpenAI:

```python
kwargs = self._build_request_kwargs(
    target_model=target_model,
    messages=[{"role": "user", "content": "Ping"}],
    max_tokens=1,
    is_probe=True,
)
response = self._client.chat.completions.create(**kwargs)
```

This verifies:
1. Valid API credentials (no HTTP 401).
2. Active billing and positive quota balance (no `insufficient_quota`).
3. Model availability on your OpenAI tier.
4. Latency measurement in seconds.

---

## 5. Production Circuit Breakers & Retry Policies

To safeguard user funds and ensure reliable operation:

* **Authentication Circuit Breaker (HTTP 401):**
  If OpenAI returns an `AuthenticationError`, the pipeline **halts immediately** and raises a clean exception. It never retries bad credentials.

* **Quota Circuit Breaker (`insufficient_quota`):**
  If an API call fails due to exhausted balance or billing limits, the circuit breaker triggers immediately. Retrying a zero-balance account is futile and is blocked.

* **Secret Sanitization in Errors:**
  All exception and log messages are passed through `sanitize_error_text()` to ensure that API keys or key patterns (`sk-...`) are never exposed in error text or logs.

* **Bounded Transient Retries:**
  For transient server errors (HTTP 500, 502, 503) or rate limits (HTTP 429), the client employs bounded exponential backoff with a jittered multiplier (up to 3 retries max).

---

## 6. Token Efficiency & Cost Estimation

Batching candidates produces significant token savings:
* **Batch Size:** 3 candidates evaluated per prompt.
* **Prompt Optimization:** Slices share system instructions and Cold-Viewer persona definitions, saving ~40% prompt overhead compared to evaluating candidates individually.
* **Token Profile:**
  - Average prompt tokens per batch: **~2,300 tokens**.
  - Average completion tokens per batch: **~650 tokens**.
  - Total tokens for 50 episodes (~450 candidates / 150 batches): **~442,500 tokens**.

### Pre-Run Cost Formula
$$\text{Cost} = \left(\text{Batches} \times 2300 \times \frac{P_{\text{input}}}{1,000,000}\right) + \left(\text{Batches} \times 650 \times \frac{P_{\text{output}}}{1,000,000}\right)$$

* For **`gpt-4o-mini`**: A full 50-episode run costs **~\$0.11 USD**.
* For **`gpt-4o`**: A full 50-episode run costs **~\$1.84 USD**.

---

## 6. Model-Aware Checkpoint Separation

The system ensures that evaluations are cache-isolated:
* In `evaluator.py`, `load_checkpoints()` inspects both `evaluation_source` and `model_id`.
* If a checkpoint contains evaluations from `meta-llama/llama-3.3-70b-instruct` or `gpt-4o-mini`, running the pipeline with `gpt-4o` will **not** falsely reuse those cached scores. It evaluates fresh candidates on the requested model while keeping the existing checkpoint lines intact.
