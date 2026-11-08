"""Official OpenAI client adapter with structured schema validation and circuit breaker.

Uses the OpenAI Python SDK. Set a base URL to reach any OpenAI-compatible endpoint, such as OpenRouter.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple
from pydantic import BaseModel, ValidationError
from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

from kuku_promo_system.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_DEFAULT_MODEL,
    MAX_RETRIES,
    RETRY_BASE_DELAY_SEC,
    EVALUATION_TIMEOUT_SEC,
    MAX_COMPLETION_TOKENS,
    mask_secret,
)
from kuku_promo_system.logger import logger

T = TypeVar("T", bound=BaseModel)


def clean_json_markdown_fences(raw_text: str) -> str:
    """Strip markdown code block fences if returned by the LLM."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def infer_provider(base_url: Optional[str] = None) -> str:
    """Infer the LLM provider based on the endpoint base URL."""
    if not base_url or not str(base_url).strip():
        return "openai"
    url_lower = str(base_url).lower()
    if "api.openai.com" in url_lower:
        return "openai"
    if "openrouter.ai" in url_lower:
        return "openrouter"
    return "custom"


def is_reasoning_model(model_name: Optional[str]) -> bool:
    """Determine whether the specified model belongs to the OpenAI reasoning series (e.g. o1, o3)."""
    if not model_name:
        return False
    slug = model_name.lower().strip()
    if "/" in slug:
        slug = slug.split("/")[-1]
    return slug.startswith("o1") or slug.startswith("o3")


def sanitize_error_text(error_text: str, secret: Optional[str] = None) -> str:
    """Ensure error messages never leak API keys or token fragments."""
    text = str(error_text)
    if secret and len(secret) >= 6:
        text = text.replace(secret, mask_secret(secret))
    # Redact any remaining standard sk- key patterns
    text = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "sk-***[REDACTED]", text)
    return text


class MissingAPIKeyError(RuntimeError):
    """Raised when OpenAI API key is missing or is an invalid placeholder."""
    pass


class LLMClientAdapter:
    """Official OpenAI API interface for structured JSON batch completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = MAX_COMPLETION_TOKENS,
        timeout_sec: float = EVALUATION_TIMEOUT_SEC,
    ):
        self.model = model or os.getenv("OPENAI_MODEL") or LLM_MODEL or OPENAI_DEFAULT_MODEL
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or LLM_BASE_URL
        self.provider = infer_provider(self.base_url)
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self._circuit_broken = False

        # Resolve OpenAI API key securely: parameter -> OPENAI_API_KEY -> LLM_API_KEY
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

        # Detect placeholder or missing credentials
        is_placeholder = (
            not self.api_key
            or not str(self.api_key).strip()
            or "PASTE_" in self.api_key
            or self.api_key in ("your-api-key-here", "your_api_key_here", "your_openrouter_api_key_here")
        )

        if is_placeholder:
            raise MissingAPIKeyError(
                "OpenAI API key is missing or invalid. Set the OPENAI_API_KEY environment variable or enter a key in the sidebar."
            )

        logger.info(
            f"Initialized OpenAI LLMClientAdapter: model='{self.model}', "
            f"max_tokens={self.max_tokens}, key={mask_secret(self.api_key)}"
        )

        self._client: Optional[OpenAI] = None
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout_sec,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

    def _build_request_kwargs(
        self,
        target_model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        is_probe: bool = False,
    ) -> Dict[str, Any]:
        """Construct model-compatible request parameters handling reasoning (o1/o3) vs standard chat models."""
        kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
        }
        token_param = "max_completion_tokens" if self.provider == "openai" else "max_tokens"

        if is_reasoning_model(target_model):
            # Reasoning models (o1, o3-mini):
            # 1. Strictly require max_completion_tokens (reject max_tokens)
            # 2. Reject custom temperature (only default 1.0 supported)
            # 3. For probes, provide sufficient token budget for reasoning + answer
            kwargs[token_param] = max(max_tokens, 25) if is_probe else max_tokens
            if not is_probe:
                kwargs["response_format"] = {"type": "json_object"}
        else:
            # Standard chat models (gpt-4o, gpt-4o-mini, gpt-4-turbo):
            # Both max_completion_tokens and temperature are supported
            kwargs[token_param] = max_tokens
            kwargs["temperature"] = temperature
            if not is_probe:
                kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    def verify_connection(self, model_id: Optional[str] = None) -> Tuple[bool, str, float]:
        """Verify OpenAI API connectivity and model accessibility with an inexpensive test request.

        Returns:
            (is_successful, status_message, latency_seconds)
        """
        if not self._client:
            return False, "No active OpenAI client connection.", 0.0

        target_model = model_id or self.model
        t0 = time.time()
        kwargs = self._build_request_kwargs(
            target_model=target_model,
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=1,
            is_probe=True,
        )

        try:
            # Minimal token test prompt to verify credentials and model access
            try:
                response = self._client.chat.completions.create(**kwargs)
            except BadRequestError as bre:
                if "max_completion_tokens" in str(bre).lower() and "unsupported" in str(bre).lower():
                    # Fallback to legacy max_tokens for older custom endpoints
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = 1
                    response = self._client.chat.completions.create(**kwargs)
                else:
                    raise

            latency = round(time.time() - t0, 3)
            return True, f"Connection verified successfully on model '{target_model}'.", latency
        except AuthenticationError as ae:
            latency = round(time.time() - t0, 3)
            clean_err = sanitize_error_text(str(ae.message if hasattr(ae, 'message') else ae), self.api_key)
            return False, f"Authentication Failed (HTTP 401): {clean_err}", latency
        except RateLimitError as rle:
            latency = round(time.time() - t0, 3)
            clean_err = sanitize_error_text(str(rle), self.api_key)
            if "insufficient_quota" in clean_err.lower() or "quota" in clean_err.lower() or "billing" in clean_err.lower():
                return False, f"Insufficient Quota / Billing Error: {clean_err}", latency
            return False, f"Rate Limit Encountered: {clean_err}", latency
        except BadRequestError as bre:
            latency = round(time.time() - t0, 3)
            clean_err = sanitize_error_text(str(bre), self.api_key)
            return False, f"Invalid Model or Request (HTTP 400): {clean_err}", latency
        except Exception as e:
            latency = round(time.time() - t0, 3)
            clean_err = sanitize_error_text(str(e), self.api_key)
            return False, f"Connection Error: {clean_err}", latency

    def generate_structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.0,
    ) -> Tuple[Optional[T], bool, Optional[str], Dict[str, Any], float]:
        """Execute chat completion with JSON schema enforcement, category-specific retries, and circuit breaker.

        Returns:
            (validated_pydantic_object, is_simulated, error_message, usage_dict, latency_sec)
        """
        # Circuit breaker check
        if self._circuit_broken:
            err_msg = (
                "[CIRCUIT BREAKER ACTIVE] OpenAI adapter encountered a fatal authentication "
                "or quota exhaustion error. Further network requests blocked."
            )
            logger.error(err_msg)
            return None, False, err_msg, {}, 0.0

        start_time = time.time()
        max_total_attempts = 4
        current_attempt = 0
        last_error = None

        schema_correction_attempts = 0
        server_error_attempts = 0
        timeout_attempts = 0
        rate_limit_attempts = 0

        working_messages = list(messages)

        while current_attempt < max_total_attempts:
            current_attempt += 1
            call_start = time.time()

            req_kwargs = self._build_request_kwargs(
                target_model=self.model,
                messages=working_messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                is_probe=False,
            )

            try:
                try:
                    response = self._client.chat.completions.create(**req_kwargs)
                except BadRequestError as bre:
                    if "max_completion_tokens" in str(bre).lower() and "unsupported" in str(bre).lower():
                        # Fallback to legacy max_tokens for older custom endpoints
                        req_kwargs.pop("max_completion_tokens", None)
                        req_kwargs["max_tokens"] = self.max_tokens
                        response = self._client.chat.completions.create(**req_kwargs)
                    else:
                        raise

                latency = round(time.time() - call_start, 2)

                raw_content = response.choices[0].message.content
                if not raw_content:
                    raise ValueError("Received empty response content from OpenAI API")

                # Extract token telemetry
                usage_dict = {}
                if hasattr(response, "usage") and response.usage:
                    usage_dict = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }

                # Clean markdown fences if any and validate schema
                cleaned_json = clean_json_markdown_fences(raw_content)
                data = json.loads(cleaned_json)
                validated = response_model.model_validate(data)
                return validated, False, None, usage_dict, latency

            except AuthenticationError as ae:
                self._circuit_broken = True
                clean_err = sanitize_error_text(str(ae), self.api_key)
                err_msg = f"[CIRCUIT BREAKER TRIPPED] OpenAI Authentication Error (HTTP 401): {clean_err}. Halting execution."
                logger.error(err_msg)
                return None, False, err_msg, {}, round(time.time() - start_time, 2)

            except RateLimitError as rle:
                clean_err = sanitize_error_text(str(rle), self.api_key)
                err_str = clean_err.lower()
                # If rate limit is actually due to credit/quota exhaustion, trip circuit breaker immediately
                if "insufficient_quota" in err_str or "quota" in err_str or "billing" in err_str:
                    self._circuit_broken = True
                    err_msg = f"[CIRCUIT BREAKER TRIPPED] OpenAI Insufficient Quota: {clean_err}. Halting execution."
                    logger.error(err_msg)
                    return None, False, err_msg, {}, round(time.time() - start_time, 2)

                rate_limit_attempts += 1
                last_error = str(rle)
                if rate_limit_attempts > 3:
                    logger.error(f"Exceeded max rate-limit retries (3) on OpenAI: {rle}")
                    break

                # Extract Retry-After header if present
                retry_after = None
                if hasattr(rle, "response") and rle.response and hasattr(rle.response, "headers"):
                    retry_after = rle.response.headers.get("retry-after")

                if retry_after:
                    try:
                        delay = float(retry_after) + random.uniform(0.5, 1.5)
                    except ValueError:
                        delay = (RETRY_BASE_DELAY_SEC ** rate_limit_attempts) + random.uniform(0.5, 1.5)
                else:
                    delay = (RETRY_BASE_DELAY_SEC ** rate_limit_attempts) + random.uniform(0.5, 1.5)

                delay = min(delay, 30.0)
                logger.warning(
                    f"Rate limit encountered on OpenAI (retry {rate_limit_attempts}/3). Backing off for {delay:.2f}s..."
                )
                time.sleep(delay)

            except BadRequestError as bre:
                # Fatal request malformation or missing model slug
                err_msg = f"Fatal OpenAI BadRequestError (HTTP 400): {bre}"
                logger.error(err_msg)
                return None, False, err_msg, {}, round(time.time() - start_time, 2)

            except (APITimeoutError, APIConnectionError) as te:
                timeout_attempts += 1
                last_error = str(te)
                if timeout_attempts > 2:
                    logger.error(f"Exceeded max connection/timeout retries (2) on OpenAI: {te}")
                    break

                delay = RETRY_BASE_DELAY_SEC * timeout_attempts
                logger.warning(f"Connection timeout encountered on OpenAI (retry {timeout_attempts}/2). Retrying in {delay:.2f}s...")
                time.sleep(delay)

            except (json.JSONDecodeError, ValidationError) as ve:
                schema_correction_attempts += 1
                last_error = str(ve)
                if schema_correction_attempts > 2:
                    logger.error(f"Exceeded max schema correction attempts (2) on OpenAI: {ve}")
                    break

                logger.warning(f"Schema validation error on OpenAI (attempt {schema_correction_attempts}/2): {ve}")
                working_messages.append({
                    "role": "user",
                    "content": f"Your previous response failed JSON schema validation: {ve}. Please respond with valid JSON strictly adhering to the schema."
                })

            except (InternalServerError, APIError) as ae:
                last_error = str(ae)
                status_code = getattr(ae, "status_code", None)
                err_str = str(ae).lower()

                # Fatal payment required
                if status_code == 402 or "402" in err_str:
                    self._circuit_broken = True
                    err_msg = f"[CIRCUIT BREAKER TRIPPED] Payment Required (HTTP 402): {ae}. Halting execution."
                    logger.error(err_msg)
                    return None, False, err_msg, {}, round(time.time() - start_time, 2)

                # Transient 5xx server errors
                server_error_attempts += 1
                if server_error_attempts > 2:
                    logger.error(f"Exceeded max server error retries (2) on OpenAI: {ae}")
                    break

                delay = RETRY_BASE_DELAY_SEC ** server_error_attempts
                logger.warning(f"Transient upstream server error ({status_code or 'unknown'}) on OpenAI. Retrying in {delay:.2f}s...")
                time.sleep(delay)

            except Exception as exc:
                last_error = str(exc)
                logger.error(f"Unexpected error during OpenAI completion: {exc}")
                break

        total_elapsed = round(time.time() - start_time, 2)
        return None, False, f"OpenAI request failed after {current_attempt} attempts: {last_error}", {}, total_elapsed

