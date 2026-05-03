"""Classifier — async LLM call with retry and fallback.

This module owns the single LLM call that drives the entire pipeline.
It uses the OpenAI SDK's structured-output parsing to guarantee a
valid ``ClassifierOutput`` or falls back gracefully.

Design decisions
~~~~~~~~~~~~~~~~
* **Injectable client** — ``classify()`` accepts an optional OpenAI
  client parameter so tests can pass a mock without monkeypatching.
* **Tenacity retry** — retries only on transient errors (rate limit,
  connection) with exponential backoff, max 2 retries.
* **Never raises** — on any unrecoverable failure, logs the error and
  returns ``FALLBACK_CLASSIFIER_OUTPUT``.
* **Token logging** — every successful call logs prompt + completion
  tokens via structlog for cost tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.classifier.prompts import (
    build_system_prompt,
    format_history_messages,
    summarise_user_context,
)
from src.classifier.schema import (
    ClassifierOutput,
    FALLBACK_CLASSIFIER_OUTPUT,
)
from src.config import get_settings

if TYPE_CHECKING:
    from src.models.user import UserProfile

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Retry-wrapped inner call — isolated so tenacity only retries the
# network call, not the prompt construction or fallback logic.
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(3),  # 1 initial + 2 retries
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _call_llm(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    messages: list[dict[str, str]],
) -> ClassifierOutput:
    """Make the actual LLM call with structured output parsing.

    This function is wrapped by tenacity and will be retried on
    transient errors.  It is NOT meant to be called directly —
    use ``classify()`` instead.

    Parameters
    ----------
    client:
        Async OpenAI client instance.
    model:
        Model identifier (e.g. ``"gpt-4o-mini"``).
    system_prompt:
        The full classifier system prompt.
    messages:
        The formatted conversation messages (history + current query).

    Returns
    -------
    ClassifierOutput
        Parsed and validated classifier output.

    Raises
    ------
    RateLimitError, APIConnectionError
        Transient errors — tenacity will retry these.
    Exception
        Any other error — will NOT be retried.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=full_messages,
        response_format=ClassifierOutput,
        temperature=0.0,
    )

    result = response.choices[0].message.parsed

    # Log token usage for cost tracking.
    if response.usage:
        await logger.ainfo(
            "classifier_llm_call",
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    if result is None:
        await logger.awarning("classifier_parsed_none", model=model)
        return FALLBACK_CLASSIFIER_OUTPUT

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify(
    query: str,
    session_history: list[dict[str, str]],
    user_profile: UserProfile,
    *,
    client: AsyncOpenAI | None = None,
) -> ClassifierOutput:
    """Classify a user query and return structured routing output.

    This is the only function external code should call.  It handles
    prompt construction, LLM invocation, retry, and graceful fallback.

    Parameters
    ----------
    query:
        The user's natural-language query.
    session_history:
        Prior conversation turns as
        ``[{"role": "user"|"assistant", "content": "..."}]``.
    user_profile:
        The full user profile for context-aware classification.
    client:
        Optional async OpenAI client.  Defaults to a new client
        constructed from settings.  Pass a mock in tests.

    Returns
    -------
    ClassifierOutput
        Always returns a valid output — never raises.  On LLM failure,
        returns ``FALLBACK_CLASSIFIER_OUTPUT``.
    """
    settings = get_settings()

    if client is None:
        client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Build the system prompt.
    system_prompt = build_system_prompt()

    # Build user context summary for the classifier.
    user_context = summarise_user_context(
        user_name=user_profile.name,
        risk_profile=user_profile.risk_profile,
        num_holdings=len(user_profile.positions),
        base_currency=user_profile.base_currency,
        country=user_profile.country,
    )

    # Format messages array.
    messages = format_history_messages(
        session_history=session_history,
        current_query=query,
        user_context_summary=user_context,
    )

    try:
        result = await _call_llm(
            client=client,
            model=settings.classifier_model,
            system_prompt=system_prompt,
            messages=messages,
        )
        await logger.ainfo(
            "classifier_success",
            intent=result.intent,
            target_agent=result.target_agent.value,
            confidence=result.confidence,
            safety_verdict=result.safety_verdict,
        )
        return result

    except (RateLimitError, APIConnectionError) as exc:
        # Tenacity exhausted all retries — fall back gracefully.
        await logger.aerror(
            "classifier_retries_exhausted",
            error_type=type(exc).__name__,
            error_msg=str(exc),
        )
        return FALLBACK_CLASSIFIER_OUTPUT

    except Exception as exc:
        # Any other error — schema validation failure, unexpected API
        # error, etc.  Log and fall back.
        await logger.aerror(
            "classifier_unexpected_error",
            error_type=type(exc).__name__,
            error_msg=str(exc),
        )
        return FALLBACK_CLASSIFIER_OUTPUT
