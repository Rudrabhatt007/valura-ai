"""Classifier schema and integration tests.

Two categories of tests:

1. **Schema tests** (no LLM needed) — validate that Pydantic models
   parse correctly, the ``FALLBACK_CLASSIFIER_OUTPUT`` is valid, and
   ``EntitySet`` handles all optional fields.

2. **Integration tests** (with mock LLM) — verify the ``classify()``
   function returns correct output on success, falls back on failure,
   and handles schema validation errors gracefully.

All tests run WITHOUT ``OPENAI_API_KEY`` — the LLM is always mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.classifier.classifier import classify
from src.classifier.prompts import (
    AGENT_DESCRIPTIONS,
    ENTITY_VOCABULARY,
    build_system_prompt,
    format_history_messages,
    summarise_user_context,
)
from src.classifier.schema import (
    AgentType,
    ClassifierOutput,
    EntitySet,
    FALLBACK_CLASSIFIER_OUTPUT,
)
from src.models.user import UserProfile


# ═══════════════════════════════════════════════════════════════════════════
# 1. SCHEMA TESTS — no LLM mock needed
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifierOutputSchema:
    """Validate the ClassifierOutput Pydantic model."""

    def test_valid_output(self) -> None:
        """A fully populated ClassifierOutput should validate."""
        output = ClassifierOutput(
            intent="stock_price_lookup",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["AAPL"]),
            safety_verdict="safe",
            confidence=0.92,
            reasoning="User wants current price of AAPL.",
        )
        assert output.target_agent == AgentType.MARKET_RESEARCH
        assert output.confidence == 0.92

    def test_minimal_output(self) -> None:
        """ClassifierOutput with only required fields should validate."""
        output = ClassifierOutput(
            intent="greeting",
            target_agent=AgentType.GENERAL_QUERY,
            confidence=0.99,
            reasoning="Simple greeting.",
        )
        assert output.entities.tickers == []
        assert output.safety_verdict == "safe"

    def test_all_agent_types_are_valid(self) -> None:
        """Every AgentType enum value should be accepted."""
        for agent in AgentType:
            output = ClassifierOutput(
                intent="test",
                target_agent=agent,
                confidence=0.5,
                reasoning="test",
            )
            assert output.target_agent == agent

    def test_safety_verdict_values(self) -> None:
        """Only 'safe', 'warning', 'unsafe' should be accepted."""
        for verdict in ("safe", "warning", "unsafe"):
            output = ClassifierOutput(
                intent="test",
                target_agent=AgentType.GENERAL_QUERY,
                safety_verdict=verdict,
                confidence=0.5,
                reasoning="test",
            )
            assert output.safety_verdict == verdict

    def test_invalid_safety_verdict_rejected(self) -> None:
        """Invalid safety verdict should raise validation error."""
        with pytest.raises(Exception):
            ClassifierOutput(
                intent="test",
                target_agent=AgentType.GENERAL_QUERY,
                safety_verdict="maybe",  # type: ignore[arg-type]
                confidence=0.5,
                reasoning="test",
            )

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        # Valid bounds
        ClassifierOutput(
            intent="test", target_agent=AgentType.GENERAL_QUERY,
            confidence=0.0, reasoning="test",
        )
        ClassifierOutput(
            intent="test", target_agent=AgentType.GENERAL_QUERY,
            confidence=1.0, reasoning="test",
        )

        # Out of bounds
        with pytest.raises(Exception):
            ClassifierOutput(
                intent="test", target_agent=AgentType.GENERAL_QUERY,
                confidence=1.5, reasoning="test",
            )
        with pytest.raises(Exception):
            ClassifierOutput(
                intent="test", target_agent=AgentType.GENERAL_QUERY,
                confidence=-0.1, reasoning="test",
            )


class TestEntitySet:
    """Validate the EntitySet model."""

    def test_empty_entity_set(self) -> None:
        """EntitySet with no fields should be valid."""
        es = EntitySet()
        assert es.tickers == []
        assert es.topics == []
        assert es.amount is None
        assert es.action is None

    def test_full_entity_set(self) -> None:
        """EntitySet with all fields populated should be valid."""
        es = EntitySet(
            tickers=["AAPL", "NVDA"],
            topics=["ETF", "compound interest"],
            sectors=["technology"],
            amount=2500.0,
            rate=0.08,
            period_years=20,
            currency="USD",
            index="S&P 500",
            action="buy",
            goal="retirement",
            frequency="monthly",
            horizon="5_years",
            time_period="this_month",
        )
        assert len(es.tickers) == 2
        assert es.amount == 2500.0
        assert es.action == "buy"

    def test_entity_set_serialization(self) -> None:
        """EntitySet should serialize to dict, excluding None values."""
        es = EntitySet(tickers=["AAPL"], action="sell")
        dumped = es.model_dump(exclude_none=True)
        assert "tickers" in dumped
        assert "action" in dumped
        assert "amount" not in dumped
        assert "currency" not in dumped

    def test_invalid_action_rejected(self) -> None:
        """Invalid action values should be rejected."""
        with pytest.raises(Exception):
            EntitySet(action="yolo")  # type: ignore[arg-type]

    def test_invalid_goal_rejected(self) -> None:
        """Invalid goal values should be rejected."""
        with pytest.raises(Exception):
            EntitySet(goal="vacation")  # type: ignore[arg-type]


class TestFallbackOutput:
    """Validate the FALLBACK_CLASSIFIER_OUTPUT constant."""

    def test_fallback_is_valid(self) -> None:
        """FALLBACK must be a valid ClassifierOutput instance."""
        assert isinstance(FALLBACK_CLASSIFIER_OUTPUT, ClassifierOutput)

    def test_fallback_routes_to_support(self) -> None:
        """FALLBACK must route to customer_support."""
        assert FALLBACK_CLASSIFIER_OUTPUT.target_agent == AgentType.CUSTOMER_SUPPORT

    def test_fallback_has_zero_confidence(self) -> None:
        """FALLBACK must have confidence == 0.0."""
        assert FALLBACK_CLASSIFIER_OUTPUT.confidence == 0.0

    def test_fallback_is_safe(self) -> None:
        """FALLBACK should have safety_verdict == 'safe'."""
        assert FALLBACK_CLASSIFIER_OUTPUT.safety_verdict == "safe"

    def test_fallback_has_empty_entities(self) -> None:
        """FALLBACK should have empty entities."""
        assert FALLBACK_CLASSIFIER_OUTPUT.entities.tickers == []
        assert FALLBACK_CLASSIFIER_OUTPUT.entities.amount is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. PROMPT TESTS — verify prompt construction
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptBuilder:
    """Verify the system prompt is well-formed."""

    def test_prompt_contains_all_agents(self) -> None:
        """System prompt must mention every agent type."""
        prompt = build_system_prompt()
        for agent in AgentType:
            assert agent.value in prompt, f"Agent {agent.value} missing from prompt"

    def test_prompt_contains_vocabulary(self) -> None:
        """System prompt must mention all closed vocabulary values."""
        prompt = build_system_prompt()
        for field, values in ENTITY_VOCABULARY.items():
            for value in values:
                assert value in prompt, f"Vocab value '{value}' for '{field}' missing"

    def test_prompt_contains_key_instructions(self) -> None:
        """System prompt must contain key classifier instructions."""
        prompt = build_system_prompt()
        assert "AGENT TAXONOMY" in prompt
        assert "ENTITY VOCABULARY" in prompt
        assert "FOLLOW-UP RESOLUTION" in prompt
        assert "SAFETY VERDICT" in prompt
        assert "OUTPUT FORMAT" in prompt

    def test_custom_agent_descriptions(self) -> None:
        """build_system_prompt should accept custom agent descriptions."""
        custom = {"test_agent": "A test agent for unit tests."}
        prompt = build_system_prompt(agent_descriptions=custom)
        assert "test_agent" in prompt
        assert "A test agent for unit tests." in prompt

    def test_history_formatting_with_context(self) -> None:
        """format_history_messages should include user context and history."""
        messages = format_history_messages(
            session_history=[
                {"role": "user", "content": "tell me about NVDA"},
                {"role": "assistant", "content": "NVDA is NVIDIA..."},
            ],
            current_query="how much do I own?",
            user_context_summary="Alex | aggressive | 9 holdings",
        )
        # System context + 2 history turns + current query = 4 messages
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert "Alex" in messages[0]["content"]
        assert messages[-1]["content"] == "how much do I own?"

    def test_history_formatting_without_context(self) -> None:
        """format_history_messages should work without user context."""
        messages = format_history_messages(
            session_history=[],
            current_query="hello",
        )
        assert len(messages) == 1
        assert messages[0]["content"] == "hello"

    def test_user_context_summary(self) -> None:
        """summarise_user_context should produce a readable summary."""
        ctx = summarise_user_context("Jamie", "moderate", 0, "USD", "US")
        assert "empty portfolio" in ctx
        assert "moderate" in ctx

        ctx2 = summarise_user_context("Alex", "aggressive", 9, "USD", "US")
        assert "9 holdings" in ctx2


# ═══════════════════════════════════════════════════════════════════════════
# 3. INTEGRATION TESTS — with mock LLM
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyIntegration:
    """Test the classify() function with a mocked OpenAI client."""

    @pytest.mark.asyncio
    async def test_classify_returns_output_on_success(
        self,
        mock_openai_client,
        sample_user_profile: UserProfile,
    ) -> None:
        """classify() should return the parsed ClassifierOutput on success."""
        result = await classify(
            query="how is my portfolio doing?",
            session_history=[],
            user_profile=sample_user_profile,
            client=mock_openai_client,
        )
        assert isinstance(result, ClassifierOutput)
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_classify_returns_fallback_on_exception(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """classify() should return FALLBACK on unexpected errors."""
        # Create a client that raises on parse().
        bad_client = AsyncMock()
        bad_client.beta.chat.completions.parse = AsyncMock(
            side_effect=ValueError("Unexpected schema error"),
        )

        result = await classify(
            query="test query",
            session_history=[],
            user_profile=sample_user_profile,
            client=bad_client,
        )
        assert result == FALLBACK_CLASSIFIER_OUTPUT
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_classify_returns_fallback_on_rate_limit(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """classify() should return FALLBACK after rate limit retries are exhausted."""
        from openai import RateLimitError

        # Build a mock response object for the RateLimitError.
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "rate limited"}}

        bad_client = AsyncMock()
        bad_client.beta.chat.completions.parse = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "rate limited"}},
            ),
        )

        result = await classify(
            query="test query",
            session_history=[],
            user_profile=sample_user_profile,
            client=bad_client,
        )
        assert result == FALLBACK_CLASSIFIER_OUTPUT

    @pytest.mark.asyncio
    async def test_classify_returns_fallback_on_none_parsed(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """classify() should return FALLBACK when parsed result is None."""
        client = AsyncMock()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150

        mock_message = MagicMock()
        mock_message.parsed = None  # LLM returned unparseable output

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        result = await classify(
            query="test query",
            session_history=[],
            user_profile=sample_user_profile,
            client=client,
        )
        assert result == FALLBACK_CLASSIFIER_OUTPUT

    @pytest.mark.asyncio
    async def test_classify_with_session_history(
        self,
        mock_openai_client,
        sample_user_profile: UserProfile,
    ) -> None:
        """classify() should accept and pass session history."""
        history = [
            {"role": "user", "content": "tell me about NVDA"},
            {"role": "assistant", "content": "NVDA is NVIDIA Corporation..."},
        ]

        result = await classify(
            query="how much do I own?",
            session_history=history,
            user_profile=sample_user_profile,
            client=mock_openai_client,
        )
        assert isinstance(result, ClassifierOutput)

        # Verify the mock was called with messages that include history.
        call_args = mock_openai_client.beta.chat.completions.parse.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        # Should have: system prompt + user context + 2 history + current = 5
        assert len(messages) >= 4

    @pytest.mark.asyncio
    async def test_classify_custom_output(
        self,
        mock_openai_client,
        sample_user_profile: UserProfile,
    ) -> None:
        """mock_openai_client.configure() should change the returned output."""
        custom_output = ClassifierOutput(
            intent="market_lookup",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["TSLA"]),
            safety_verdict="safe",
            confidence=0.88,
            reasoning="User wants TSLA info.",
        )
        mock_openai_client.configure(custom_output)

        result = await classify(
            query="tell me about Tesla",
            session_history=[],
            user_profile=sample_user_profile,
            client=mock_openai_client,
        )
        assert result.target_agent == AgentType.MARKET_RESEARCH
        assert result.entities.tickers == ["TSLA"]
        assert result.confidence == 0.88
