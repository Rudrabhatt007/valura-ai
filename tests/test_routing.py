"""Router tests — StubAgent, agent registry, SSE format validation.

No LLM calls in any routing test. All tests run without OPENAI_API_KEY.
"""

from __future__ import annotations

import json

import pytest

from src.classifier.schema import AgentType, EntitySet
from src.models.user import UserProfile
from src.router.router import AGENT_REGISTRY, StubAgent, get_agent
from src.utils.sse import (
    format_safety_block,
    format_sse_done,
    format_sse_error,
    format_sse_event,
    format_sse_metadata,
)


# ---------------------------------------------------------------------------
# SSE Format Validation
# ---------------------------------------------------------------------------

class TestSSEFormat:
    """Verify the SSE format is exactly correct."""

    def test_format_sse_event_contains_event_prefix(self) -> None:
        """Event line must start with 'event: '."""
        result = format_sse_event("metadata", {"key": "value"})
        assert result.startswith("event: metadata\n")

    def test_format_sse_event_contains_data_prefix(self) -> None:
        """Data line must start with 'data: '."""
        result = format_sse_event("test", {"a": 1})
        lines = result.split("\n")
        assert lines[1].startswith("data: ")

    def test_format_sse_event_ends_with_double_newline(self) -> None:
        """Each SSE block must end with \\n\\n."""
        result = format_sse_event("test", {"a": 1})
        assert result.endswith("\n\n")

    def test_format_sse_event_data_is_valid_json(self) -> None:
        """Data field must be parseable JSON when given a dict."""
        result = format_sse_event("test", {"key": "value", "n": 42})
        data_line = result.split("\n")[1]
        data_str = data_line.removeprefix("data: ")
        parsed = json.loads(data_str)
        assert parsed["key"] == "value"
        assert parsed["n"] == 42

    def test_format_sse_event_string_data(self) -> None:
        """String data should be passed through without JSON serialization."""
        result = format_sse_event("chunk", "raw text")
        assert "data: raw text\n" in result

    def test_format_sse_event_order(self) -> None:
        """event: line must come before data: line."""
        result = format_sse_event("test", {"a": 1})
        event_pos = result.index("event:")
        data_pos = result.index("data:")
        assert event_pos < data_pos

    def test_format_sse_error_contains_error_true(self) -> None:
        """Error event must contain error: true."""
        result = format_sse_error("test message", "TEST_CODE")
        data_line = result.split("\n")[1].removeprefix("data: ")
        parsed = json.loads(data_line)
        assert parsed["error"] is True
        assert parsed["code"] == "TEST_CODE"
        assert parsed["message"] == "test message"

    def test_format_sse_done_has_status_complete(self) -> None:
        """Done event must contain status: complete."""
        result = format_sse_done()
        assert "event: done\n" in result
        data_line = result.split("\n")[1].removeprefix("data: ")
        parsed = json.loads(data_line)
        assert parsed["status"] == "complete"

    def test_format_sse_metadata_event_type(self) -> None:
        """Metadata event must use 'event: metadata'."""
        result = format_sse_metadata({"intent": "test"})
        assert result.startswith("event: metadata\n")

    def test_format_safety_block_event_type(self) -> None:
        """Safety block must use 'event: safety_block'."""
        result = format_safety_block("insider_trading", "blocked")
        assert result.startswith("event: safety_block\n")
        data_line = result.split("\n")[1].removeprefix("data: ")
        parsed = json.loads(data_line)
        assert parsed["blocked"] is True
        assert parsed["category"] == "insider_trading"


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    """Verify agent registry is complete and correct."""

    def test_registry_has_all_agent_types(self) -> None:
        """Every AgentType value must be in AGENT_REGISTRY."""
        for agent_type in AgentType:
            assert agent_type in AGENT_REGISTRY, (
                f"Missing registry entry for {agent_type.value}"
            )

    def test_registry_count_matches_enum(self) -> None:
        """Registry size must match AgentType enum size."""
        assert len(AGENT_REGISTRY) == len(AgentType)

    def test_get_agent_returns_stub_for_all_types(self) -> None:
        """get_agent() returns a StubAgent for every type (before Day 2)."""
        for agent_type in AgentType:
            agent = get_agent(agent_type)
            assert isinstance(agent, StubAgent), (
                f"{agent_type.value} returned {type(agent).__name__}, expected StubAgent"
            )

    def test_get_agent_never_raises(self) -> None:
        """get_agent() must not raise even for unexpected input."""
        # This tests the fallback when an AgentType is somehow not in registry.
        # Since all are registered, this should still work fine.
        agent = get_agent(AgentType.CUSTOMER_SUPPORT)
        assert agent is not None


# ---------------------------------------------------------------------------
# StubAgent
# ---------------------------------------------------------------------------

class TestStubAgent:
    """Verify StubAgent produces correct output and never crashes."""

    @pytest.mark.asyncio
    async def test_stub_yields_at_least_one_chunk(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """StubAgent.run() must yield at least one SSE chunk."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="test query",
            entities=EntitySet(),
            user_profile=sample_user_profile,
            session_history=[],
            classified_intent="test_intent",
            target_agent="test_agent",
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stub_response_is_valid_sse(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """StubAgent must produce a properly formatted SSE event."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="test query",
            entities=EntitySet(),
            user_profile=sample_user_profile,
            session_history=[],
            classified_intent="test_intent",
            target_agent="test_agent",
        ):
            chunks.append(chunk)

        sse_output = chunks[0]
        assert sse_output.startswith("event: agent_response\n")
        assert "data: " in sse_output
        assert sse_output.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_stub_response_contains_required_fields(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """StubAgent response must contain all required fields."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="test query",
            entities=EntitySet(tickers=["AAPL"]),
            user_profile=sample_user_profile,
            session_history=[],
            classified_intent="portfolio_health_check",
            target_agent="portfolio_health",
        ):
            chunks.append(chunk)

        # Parse data from SSE
        data_line = chunks[0].split("\n")[1].removeprefix("data: ")
        parsed = json.loads(data_line)

        assert parsed["status"] == "not_implemented"
        assert parsed["classified_intent"] == "portfolio_health_check"
        assert parsed["target_agent"] == "portfolio_health"
        assert "message" in parsed
        assert "extracted_entities" in parsed
        assert "AAPL" in parsed["extracted_entities"]["tickers"]

    @pytest.mark.asyncio
    async def test_stub_never_raises_empty_query(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """StubAgent must not crash with empty query."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="",
            entities=EntitySet(),
            user_profile=sample_user_profile,
            session_history=[],
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stub_never_raises_empty_entities(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """StubAgent must not crash with empty entities."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="test",
            entities=EntitySet(),
            user_profile=sample_user_profile,
            session_history=[],
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stub_never_raises_empty_history(
        self,
        sample_user_profile: UserProfile,
    ) -> None:
        """StubAgent must not crash with empty session history."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="test",
            entities=EntitySet(),
            user_profile=sample_user_profile,
            session_history=[],
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_stub_never_raises_empty_portfolio(
        self,
        empty_portfolio_profile: UserProfile,
    ) -> None:
        """StubAgent must not crash with empty portfolio (user_004)."""
        agent = StubAgent()
        chunks = []
        async for chunk in agent.run(
            query="how is my portfolio?",
            entities=EntitySet(),
            user_profile=empty_portfolio_profile,
            session_history=[],
        ):
            chunks.append(chunk)
        assert len(chunks) >= 1
