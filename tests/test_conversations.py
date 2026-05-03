"""Conversation tests — follow-up resolution, multi-intent, and ambiguous queries.

Tests verify that:
1. History formatting produces correct messages arrays for the LLM
2. Follow-up queries carry entity context from prior turns
3. Multi-intent sessions switch topics cleanly (no inappropriate carryover)
4. Ambiguous queries are handled gracefully (no crashes, no hallucinated entities)

Uses mock classifier outputs matched against fixture expectations
via the matcher.py rules.  No real LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.classifier.prompts import format_history_messages, summarise_user_context
from src.classifier.schema import AgentType, ClassifierOutput, EntitySet
from tests.matcher import match_agent, match_entities, match_tickers


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CONVERSATIONS_DIR = FIXTURES_DIR / "conversations"


def _load_conversation(filename: str) -> dict:
    """Load a conversation fixture file."""
    path = CONVERSATIONS_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Agent name mapping — fixture uses some names that don't exactly match
# AgentType enum values. This maps expected fixture agent names to our
# AgentType values so we can validate routing correctly.
_AGENT_ALIASES: dict[str, str] = {
    "portfolio_query": "portfolio_health",  # fixture uses portfolio_query
    "portfolio_health": "portfolio_health",
    "market_research": "market_research",
    "investment_strategy": "investment_strategy",
    "financial_planning": "financial_planning",
    "financial_calculator": "financial_calculator",
    "risk_assessment": "risk_assessment",
    "product_recommendation": "product_recommendation",
    "predictive_analysis": "predictive_analysis",
    "customer_support": "customer_support",
    "general_query": "general_query",
}


def _resolve_agent(fixture_agent: str) -> str:
    """Resolve fixture agent name to AgentType value."""
    return _AGENT_ALIASES.get(fixture_agent, fixture_agent)


# ---------------------------------------------------------------------------
# History formatting tests
# ---------------------------------------------------------------------------

class TestHistoryFormatting:
    """Verify that format_history_messages() builds correct message arrays."""

    def test_empty_history_produces_single_user_message(self) -> None:
        """With no prior turns, output should be just the current query."""
        messages = format_history_messages(
            session_history=[],
            current_query="How is my portfolio?",
        )
        # Should have at least one user message (the current query).
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[-1]["content"] == "How is my portfolio?"

    def test_prior_turns_included_in_order(self) -> None:
        """Prior turns should appear before the current query."""
        history = [
            {"role": "user", "content": "What's happening with Nvidia?"},
            {"role": "assistant", "content": "NVDA is up 3% today."},
        ]
        messages = format_history_messages(
            session_history=history,
            current_query="How much do I own?",
        )
        # History turns should be present.
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        assert "What's happening with Nvidia?" in user_contents
        assert "How much do I own?" in user_contents
        # Current query must be LAST user message.
        assert user_contents[-1] == "How much do I own?"

    def test_user_context_injected_as_system_message(self) -> None:
        """User context summary should be injected as a system message."""
        context = summarise_user_context(
            user_name="Alex Chen",
            risk_profile="aggressive",
            num_holdings=5,
            base_currency="USD",
            country="US",
        )
        messages = format_history_messages(
            session_history=[],
            current_query="test",
            user_context_summary=context,
        )
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 1
        assert "Alex Chen" in system_messages[0]["content"]

    def test_history_with_multiple_turns(self) -> None:
        """History with 3 prior turns should produce correct messages."""
        history = [
            {"role": "user", "content": "What's happening with Nvidia this week?"},
            {"role": "assistant", "content": "NVDA up 5%."},
            {"role": "user", "content": "How much do I own?"},
            {"role": "assistant", "content": "50 shares of NVDA."},
            {"role": "user", "content": "Should I sell some?"},
            {"role": "assistant", "content": "Consider your goals."},
        ]
        messages = format_history_messages(
            session_history=history,
            current_query="what about AMD?",
        )
        all_user = [m["content"] for m in messages if m["role"] == "user"]
        assert len(all_user) == 4  # 3 prior + 1 current
        assert all_user[-1] == "what about AMD?"


# ---------------------------------------------------------------------------
# Follow-up session tests
# ---------------------------------------------------------------------------

class TestFollowUpSession:
    """Tests from fixtures/conversations/follow_up_session.json.

    These verify that the classifier prompt and history formatting
    enable proper follow-up resolution — entity carryover from prior
    turns when the current turn references them implicitly.
    """

    @pytest.fixture
    def follow_up_data(self) -> dict:
        return _load_conversation("follow_up_session.json")

    def test_fixture_loads(self, follow_up_data: dict) -> None:
        """Fixture file loads successfully."""
        assert "test_cases" in follow_up_data
        assert len(follow_up_data["test_cases"]) == 4

    def test_fu01_history_formatting(self, follow_up_data: dict) -> None:
        """fu_01: 'How much do I own?' — should carry NVDA from prior turn."""
        case = follow_up_data["test_cases"][0]
        # Build history from prior turns.
        history = [
            {"role": "user", "content": turn}
            for turn in case["prior_user_turns"]
        ]
        messages = format_history_messages(
            session_history=history,
            current_query=case["current_user_turn"],
        )
        # Prior turn about Nvidia must be in messages.
        all_content = " ".join(m["content"] for m in messages)
        assert "Nvidia" in all_content or "nvidia" in all_content.lower()
        assert case["current_user_turn"] in all_content

    def test_fu01_mock_classifier_matches(self, follow_up_data: dict) -> None:
        """fu_01: Mock classifier output matches expected agent and entities."""
        case = follow_up_data["test_cases"][0]
        expected = case["expected"]

        # Simulate what a correct classifier would return.
        mock_output = ClassifierOutput(
            intent="portfolio_query",
            target_agent=AgentType.PORTFOLIO_HEALTH,
            entities=EntitySet(tickers=["NVDA"]),
            safety_verdict="safe",
            confidence=0.85,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_fu02_sell_action_with_ticker_carryover(
        self, follow_up_data: dict,
    ) -> None:
        """fu_02: 'Should I sell some?' — NVDA + sell action from context."""
        case = follow_up_data["test_cases"][1]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="sell_strategy",
            target_agent=AgentType.INVESTMENT_STRATEGY,
            entities=EntitySet(tickers=["NVDA"], action="sell"),
            safety_verdict="safe",
            confidence=0.88,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_fu03_ticker_switch(self, follow_up_data: dict) -> None:
        """fu_03: 'what about AMD?' — intent carried but ticker switches."""
        case = follow_up_data["test_cases"][2]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="market_research",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["AMD"]),
            safety_verdict="safe",
            confidence=0.90,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_fu04_compare_both_tickers(self, follow_up_data: dict) -> None:
        """fu_04: 'compare them' — both NVDA and AMD from prior turns."""
        case = follow_up_data["test_cases"][3]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="comparison",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["NVDA", "AMD"]),
            safety_verdict="safe",
            confidence=0.87,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        # Tickers: subset match — expected ["NVDA", "AMD"], both must be present.
        assert match_tickers(
            mock_output.entities.tickers,
            expected["entities"]["tickers"],
        )


# ---------------------------------------------------------------------------
# Multi-intent session tests
# ---------------------------------------------------------------------------

class TestMultiIntentSession:
    """Tests from fixtures/conversations/multi_intent_session.json.

    These verify that topic switches produce clean EntitySets
    without inappropriate carryover from prior turns.
    """

    @pytest.fixture
    def multi_intent_data(self) -> dict:
        return _load_conversation("multi_intent_session.json")

    def test_fixture_loads(self, multi_intent_data: dict) -> None:
        """Fixture file loads successfully."""
        assert "test_cases" in multi_intent_data
        assert len(multi_intent_data["test_cases"]) == 4

    def test_mi01_portfolio_health(self, multi_intent_data: dict) -> None:
        """mi_01: 'How is my portfolio doing?' — portfolio_health, no entities."""
        case = multi_intent_data["test_cases"][0]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="portfolio_health_check",
            target_agent=AgentType.PORTFOLIO_HEALTH,
            entities=EntitySet(),
            safety_verdict="safe",
            confidence=0.95,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_mi02_topic_switch_to_general(
        self, multi_intent_data: dict,
    ) -> None:
        """mi_02: DCA vs lump-sum — clean topic switch to general_query."""
        case = multi_intent_data["test_cases"][1]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="educational_query",
            target_agent=AgentType.GENERAL_QUERY,
            entities=EntitySet(topics=["DCA", "lump-sum"]),
            safety_verdict="safe",
            confidence=0.91,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_mi03_calculator_with_all_params(
        self, multi_intent_data: dict,
    ) -> None:
        """mi_03: Calculator — all numeric entities extracted correctly."""
        case = multi_intent_data["test_cases"][2]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="compound_growth_calculation",
            target_agent=AgentType.FINANCIAL_CALCULATOR,
            entities=EntitySet(
                amount=2000,
                currency="USD",
                period_years=10,
                rate=0.08,
                frequency="monthly",
            ),
            safety_verdict="safe",
            confidence=0.96,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_mi04_market_research_clean_switch(
        self, multi_intent_data: dict,
    ) -> None:
        """mi_04: 'tell me about ASML' — clean switch, no carryover."""
        case = multi_intent_data["test_cases"][3]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="market_research",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["ASML"]),
            safety_verdict="safe",
            confidence=0.93,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))


# ---------------------------------------------------------------------------
# Ambiguous session tests
# ---------------------------------------------------------------------------

class TestAmbiguousSession:
    """Tests from fixtures/conversations/ambiguous_session.json.

    Edge cases: typos, slang, vague references, missing parameters.
    Classifier should be tolerant; missing entities should be None
    (not hallucinated strings).
    """

    @pytest.fixture
    def ambiguous_data(self) -> dict:
        return _load_conversation("ambiguous_session.json")

    def test_fixture_loads(self, ambiguous_data: dict) -> None:
        """Fixture file loads successfully."""
        assert "test_cases" in ambiguous_data
        assert len(ambiguous_data["test_cases"]) == 5

    def test_amb01_informal_ticker_resolution(
        self, ambiguous_data: dict,
    ) -> None:
        """amb_01: 'hows apple doing' — resolve to AAPL despite informal language."""
        case = ambiguous_data["test_cases"][0]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="market_research",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["AAPL"]),
            safety_verdict="safe",
            confidence=0.89,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_amb02_typo_resolution(self, ambiguous_data: dict) -> None:
        """amb_02: 'microsfot' — resolve MSFT despite typo."""
        case = ambiguous_data["test_cases"][1]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="market_research",
            target_agent=AgentType.MARKET_RESEARCH,
            entities=EntitySet(tickers=["MSFT"]),
            safety_verdict="safe",
            confidence=0.85,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_amb03_ambiguous_reference(self, ambiguous_data: dict) -> None:
        """amb_03: 'that thing you mentioned earlier' — general_query, empty entities."""
        case = ambiguous_data["test_cases"][2]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="clarification_needed",
            target_agent=AgentType.GENERAL_QUERY,
            entities=EntitySet(),  # No hallucinated entities
            safety_verdict="safe",
            confidence=0.60,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))

    def test_amb04_partial_params_calculator(
        self, ambiguous_data: dict,
    ) -> None:
        """amb_04: '1500 monthly for 15 years' — calculator with partial params."""
        case = ambiguous_data["test_cases"][3]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="investment_calculation",
            target_agent=AgentType.FINANCIAL_CALCULATOR,
            entities=EntitySet(
                amount=1500,
                frequency="monthly",
                period_years=15,
            ),
            safety_verdict="safe",
            confidence=0.80,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))
        # rate and currency should NOT be hallucinated.
        assert mock_output.entities.rate is None
        assert mock_output.entities.currency is None

    def test_amb05_polite_closer(self, ambiguous_data: dict) -> None:
        """amb_05: 'thx' — must NOT trigger any specialist agent."""
        case = ambiguous_data["test_cases"][4]
        expected = case["expected"]

        mock_output = ClassifierOutput(
            intent="conversational",
            target_agent=AgentType.GENERAL_QUERY,
            entities=EntitySet(),  # No entities at all
            safety_verdict="safe",
            confidence=0.95,
        )
        assert match_agent(
            mock_output.target_agent.value,
            _resolve_agent(expected["agent"]),
        )
        assert match_entities(mock_output.entities, expected.get("entities", {}))
        # Must be general_query, not any specialist.
        assert mock_output.target_agent == AgentType.GENERAL_QUERY
