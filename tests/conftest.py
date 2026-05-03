"""Shared pytest fixtures for the Valura AI assignment.

The most important fixture here is ``mock_openai_client`` — every test
that touches the classifier or any LLM-using code must use it.  CI runs
without ``OPENAI_API_KEY`` and unmocked LLM calls will fail.

Fixture categories:
- **Data loaders**: load fixture JSON files (safety pairs, intent queries, users)
- **Model factories**: pre-built Pydantic model instances for tests
- **LLM mocking**: configurable mock that mimics the OpenAI structured-output API
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.classifier.schema import (
    AgentType,
    ClassifierOutput,
    EntitySet,
)
from src.models.user import (
    Holding,
    KYCStatus,
    Preferences,
    UserProfile,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture data loaders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Root path to the fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def load_safety_pairs() -> list[dict]:
    """Load all safety query pairs from the gold set."""
    with open(FIXTURES_DIR / "test_queries" / "safety_pairs.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture(scope="session")
def load_intent_queries() -> list[dict]:
    """Load all intent classification queries from the gold set."""
    with open(FIXTURES_DIR / "test_queries" / "intent_classification.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_classifier_queries() -> list[dict]:
    """Load intent classification queries (per-test scope for compatibility)."""
    with open(FIXTURES_DIR / "test_queries" / "intent_classification.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_safety_queries() -> list[dict]:
    """Load safety query pairs (per-test scope for compatibility)."""
    with open(FIXTURES_DIR / "test_queries" / "safety_pairs.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def conversation_test_cases():
    """Returns a callable: ``conversation_test_cases('follow_up_session')``."""
    def _load(name: str) -> list[dict]:
        path = FIXTURES_DIR / "conversations" / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["test_cases"]
    return _load


def _load_user_fixture(user_id: str) -> dict:
    """Load a raw user fixture dict by user_id."""
    for path in (FIXTURES_DIR / "users").glob("*.json"):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        if user["user_id"] == user_id:
            return user
    raise FileNotFoundError(f"No fixture for user {user_id}")


@pytest.fixture
def load_user():
    """Load a user fixture by id, e.g. ``load_user('usr_001')``."""
    return _load_user_fixture


@pytest.fixture
def load_user_profile():
    """Load a user fixture and validate through ``UserProfile``.

    Usage::

        def test_something(load_user_profile):
            profile = load_user_profile("usr_001")
            assert len(profile.positions) == 9
    """
    def _load(user_id: str) -> UserProfile:
        raw = _load_user_fixture(user_id)
        return UserProfile.model_validate(raw)
    return _load


# ---------------------------------------------------------------------------
# Model factories — pre-built instances for tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_profile() -> UserProfile:
    """A minimal valid UserProfile for tests that don't need fixture data."""
    return UserProfile(
        user_id="usr_test",
        name="Test User",
        age=30,
        country="US",
        base_currency="USD",
        kyc=KYCStatus(status="verified"),
        risk_profile="moderate",
        positions=[
            Holding(
                ticker="AAPL",
                exchange="NASDAQ",
                quantity=10,
                avg_cost="150.00",
                currency="USD",
                purchased_at="2024-01-01",
            ),
        ],
        preferences=Preferences(preferred_benchmark="S&P 500"),
    )


@pytest.fixture
def sample_classifier_output() -> ClassifierOutput:
    """A valid ClassifierOutput for tests."""
    return ClassifierOutput(
        intent="portfolio_health_check",
        target_agent=AgentType.PORTFOLIO_HEALTH,
        entities=EntitySet(),
        safety_verdict="safe",
        confidence=0.95,
        reasoning="User asked about portfolio health.",
    )


@pytest.fixture
def empty_portfolio_profile() -> UserProfile:
    """UserProfile with zero holdings — mirrors user_004_empty."""
    return UserProfile(
        user_id="usr_004",
        name="Jamie Patel",
        age=31,
        country="US",
        base_currency="USD",
        kyc=KYCStatus(status="verified"),
        risk_profile="moderate",
        positions=[],
        preferences=Preferences(preferred_benchmark="S&P 500"),
    )


# ---------------------------------------------------------------------------
# LLM mocking — works WITHOUT OPENAI_API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client(sample_classifier_output: ClassifierOutput):
    """Mock AsyncOpenAI client that mimics structured-output parsing.

    Returns a configurable mock.  By default it returns
    ``sample_classifier_output``.  Override per-test::

        def test_custom(mock_openai_client):
            mock_openai_client.configure(ClassifierOutput(...))
            # now the mock returns your custom output

    The mock patches ``client.beta.chat.completions.parse()`` to
    return a response object with ``.choices[0].message.parsed`` set
    to the configured ``ClassifierOutput``, and ``.usage`` populated.
    """
    client = AsyncMock()

    # Build the mock response structure.
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_message = MagicMock()
    mock_message.parsed = sample_classifier_output

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    # Wire up the parse method.
    client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    # Helper to reconfigure the mock output.
    def configure(output: ClassifierOutput) -> None:
        mock_message.parsed = output

    client.configure = configure

    return client


@pytest.fixture
def mock_llm():
    """Legacy mock fixture — kept for backward compatibility with scaffold tests.

    Usage::

        def test_something(mock_llm):
            mock_llm.return_value = {"agent": "portfolio_health", "entities": {}}
    """
    return MagicMock()
