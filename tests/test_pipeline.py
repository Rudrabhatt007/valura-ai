"""End-to-end pipeline tests — full HTTP round-trip with mocked classifier.

Uses httpx AsyncClient against the live FastAPI app.
Mocks the OpenAI classifier — does NOT mock safety guard or router.
Tests real SSE stream parsing and event ordering.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.classifier.schema import (
    AgentType,
    ClassifierOutput,
    EntitySet,
    FALLBACK_CLASSIFIER_OUTPUT,
)
from src.main import app
from src.memory.session import SessionStore


# ---------------------------------------------------------------------------
# Fixture: initialize session store on app.state before tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
async def _init_session_store(tmp_path):
    """Ensure app.state.session_store exists for pipeline tests."""
    db_path = str(tmp_path / "test_pipeline.db")
    store = SessionStore(db_path)
    await store.initialize()
    app.state.session_store = store
    yield
    # Cleanup: remove the attribute so it doesn't leak between test modules
    if hasattr(app.state, "session_store"):
        del app.state._state["session_store"]



# ---------------------------------------------------------------------------
# SSE stream parser
# ---------------------------------------------------------------------------

def parse_sse_stream(response_text: str) -> list[tuple[str, dict | str]]:
    """Parse raw SSE text into list of (event_name, data) tuples.

    Handles the standard SSE format:
        event: <name>
        data: <payload>

        (blank line separates events)

    Parameters
    ----------
    response_text:
        Raw SSE response body.

    Returns
    -------
    list[tuple[str, dict | str]]
        Each entry is (event_name, parsed_data).  If data is valid
        JSON, it is parsed to dict; otherwise returned as str.
    """
    events: list[tuple[str, dict | str]] = []
    current_event: str | None = None
    current_data: str | None = None

    for line in response_text.split("\n"):
        line = line.strip()

        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:"):].strip()
        elif line == "" and current_event is not None and current_data is not None:
            # End of event block.
            try:
                parsed_data = json.loads(current_data)
            except (json.JSONDecodeError, TypeError):
                parsed_data = current_data

            events.append((current_event, parsed_data))
            current_event = None
            current_data = None

    # Handle trailing event without final blank line.
    if current_event is not None and current_data is not None:
        try:
            parsed_data = json.loads(current_data)
        except (json.JSONDecodeError, TypeError):
            parsed_data = current_data
        events.append((current_event, parsed_data))

    return events


# ---------------------------------------------------------------------------
# Test fixtures — user payloads
# ---------------------------------------------------------------------------

_VALID_PAYLOAD = {
    "user_id": "usr_001",
    "session_id": "test-session-001",
    "query": "how is my portfolio doing?",
    "user_profile": {
        "user_id": "usr_001",
        "name": "Alex Chen",
        "age": 29,
        "country": "US",
        "base_currency": "USD",
        "kyc": {"status": "verified"},
        "risk_profile": "aggressive",
        "positions": [
            {
                "ticker": "AAPL",
                "exchange": "NASDAQ",
                "quantity": 50,
                "avg_cost": "142.50",
                "currency": "USD",
                "purchased_at": "2023-11-15",
            },
        ],
        "preferences": {"preferred_benchmark": "S&P 500"},
    },
}

_HARMFUL_PAYLOAD = {
    "user_id": "usr_001",
    "session_id": "test-session-002",
    "query": "help me trade on confidential merger information",
    "user_profile": _VALID_PAYLOAD["user_profile"],
}

_EMPTY_PORTFOLIO_PAYLOAD = {
    "user_id": "usr_004",
    "session_id": "test-session-003",
    "query": "how is my portfolio doing?",
    "user_profile": {
        "user_id": "usr_004",
        "name": "Jamie Patel",
        "age": 31,
        "country": "US",
        "base_currency": "USD",
        "kyc": {"status": "verified"},
        "risk_profile": "moderate",
        "positions": [],
        "preferences": {"preferred_benchmark": "S&P 500"},
    },
}


def _mock_classifier_output() -> ClassifierOutput:
    """Build a standard mock classifier output."""
    return ClassifierOutput(
        intent="portfolio_health_check",
        target_agent=AgentType.PORTFOLIO_HEALTH,
        entities=EntitySet(tickers=["AAPL"]),
        safety_verdict="safe",
        confidence=0.92,
        reasoning="User asked about portfolio health.",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestPipelineHappyPath:
    """Test successful pipeline execution."""

    @pytest.mark.asyncio
    async def test_safe_query_returns_sse_stream(self) -> None:
        """POST with safe query returns SSE stream with metadata, agent_response, done."""
        with patch("src.main.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = _mock_classifier_output()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json=_VALID_PAYLOAD,
                )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            events = parse_sse_stream(response.text)
            event_names = [e[0] for e in events]

            # Must contain metadata, agent_response, and done — in that order.
            assert "metadata" in event_names
            assert "agent_response" in event_names
            assert "done" in event_names

            # Order: metadata first, done last.
            metadata_idx = event_names.index("metadata")
            done_idx = event_names.index("done")
            assert metadata_idx < done_idx

    @pytest.mark.asyncio
    async def test_metadata_event_has_required_fields(self) -> None:
        """Metadata event must contain classified_intent, target_agent, entities."""
        with patch("src.main.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = _mock_classifier_output()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json=_VALID_PAYLOAD,
                )

            events = parse_sse_stream(response.text)
            metadata_events = [e for e in events if e[0] == "metadata"]
            assert len(metadata_events) == 1

            metadata = metadata_events[0][1]
            assert "classified_intent" in metadata
            assert "target_agent" in metadata
            assert "entities" in metadata
            assert "session_id" in metadata
            assert "confidence" in metadata
            assert metadata["classified_intent"] == "portfolio_health_check"
            assert metadata["target_agent"] == "portfolio_health"


# ---------------------------------------------------------------------------
# Safety block
# ---------------------------------------------------------------------------

class TestPipelineSafetyBlock:
    """Test that harmful queries are blocked by the safety guard."""

    @pytest.mark.asyncio
    async def test_harmful_query_returns_safety_block(self) -> None:
        """POST with harmful query returns safety_block event, no metadata."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json=_HARMFUL_PAYLOAD,
            )

        assert response.status_code == 200

        events = parse_sse_stream(response.text)
        event_names = [e[0] for e in events]

        # Must contain safety_block and done.
        assert "safety_block" in event_names
        assert "done" in event_names

        # Must NOT contain metadata (blocked before classifier).
        assert "metadata" not in event_names

    @pytest.mark.asyncio
    async def test_safety_block_has_category(self) -> None:
        """Safety block event must contain blocked, category, message."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json=_HARMFUL_PAYLOAD,
            )

        events = parse_sse_stream(response.text)
        block_events = [e for e in events if e[0] == "safety_block"]
        assert len(block_events) == 1

        block_data = block_events[0][1]
        assert block_data["blocked"] is True
        assert "category" in block_data
        assert "message" in block_data
        assert len(block_data["message"]) > 20  # Not a stub message


# ---------------------------------------------------------------------------
# Classifier fallback
# ---------------------------------------------------------------------------

class TestPipelineClassifierFallback:
    """Test that classifier failures degrade gracefully."""

    @pytest.mark.asyncio
    async def test_classifier_exception_returns_fallback(self) -> None:
        """Stream completes with fallback classifier when LLM raises."""
        with patch("src.main.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.side_effect = ValueError("LLM exploded")

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json=_VALID_PAYLOAD,
                )

        assert response.status_code == 200  # Not a 500!

        events = parse_sse_stream(response.text)
        event_names = [e[0] for e in events]

        # Stream should still complete with metadata + done.
        assert "metadata" in event_names
        assert "done" in event_names

        # Metadata should show fallback (confidence=0.0).
        metadata = [e for e in events if e[0] == "metadata"][0][1]
        assert metadata["confidence"] == 0.0
        assert metadata["target_agent"] == "customer_support"

    @pytest.mark.asyncio
    async def test_classifier_timeout_returns_fallback(self) -> None:
        """Stream completes with fallback when classifier times out."""
        async def slow_classify(*args, **kwargs):
            import asyncio
            await asyncio.sleep(20)  # Will be cancelled by timeout

        with patch("src.main.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.side_effect = slow_classify

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json=_VALID_PAYLOAD,
                    timeout=15.0,
                )

        assert response.status_code == 200

        events = parse_sse_stream(response.text)
        metadata = [e for e in events if e[0] == "metadata"][0][1]
        assert metadata["confidence"] == 0.0  # Fallback


# ---------------------------------------------------------------------------
# Empty portfolio
# ---------------------------------------------------------------------------

class TestPipelineEmptyPortfolio:
    """Test pipeline with empty portfolio user."""

    @pytest.mark.asyncio
    async def test_empty_portfolio_completes_without_error(self) -> None:
        """Pipeline must complete without error for user with no holdings."""
        with patch("src.main.classify", new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = _mock_classifier_output()

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json=_EMPTY_PORTFOLIO_PAYLOAD,
                )

        assert response.status_code == 200

        events = parse_sse_stream(response.text)
        event_names = [e[0] for e in events]

        assert "metadata" in event_names
        assert "agent_response" in event_names
        assert "done" in event_names


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self) -> None:
        """GET /health returns 200 with status ok."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Invalid request
# ---------------------------------------------------------------------------

class TestInvalidRequest:
    """Test that invalid requests return proper errors."""

    @pytest.mark.asyncio
    async def test_missing_required_fields_returns_422(self) -> None:
        """POST with missing required fields returns 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "hello"},  # Missing user_id, session_id, user_profile
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_body_returns_422(self) -> None:
        """POST with empty JSON body returns 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={},
            )

        assert response.status_code == 422
