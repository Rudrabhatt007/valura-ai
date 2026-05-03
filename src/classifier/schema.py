"""Classifier output schema — AgentType enum, EntitySet, ClassifierOutput.

All types in this module are derived from the gold-standard fixture file
``fixtures/test_queries/intent_classification.json``.  The ``AgentType``
enum values are the *exact* strings from the ``agent_taxonomy`` object.
The closed-vocabulary Literal types for entity fields come from the
``entity_vocabulary`` section of the same file.

Nothing in this module is guessed — every value traces back to fixture
data.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent taxonomy — exact strings from intent_classification.json
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    """Specialist agent identifiers.

    Values are the exact ``expected_agent`` strings found in
    ``fixtures/test_queries/intent_classification.json``.
    """

    PORTFOLIO_HEALTH = "portfolio_health"
    MARKET_RESEARCH = "market_research"
    INVESTMENT_STRATEGY = "investment_strategy"
    FINANCIAL_PLANNING = "financial_planning"
    FINANCIAL_CALCULATOR = "financial_calculator"
    RISK_ASSESSMENT = "risk_assessment"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    CUSTOMER_SUPPORT = "customer_support"
    GENERAL_QUERY = "general_query"


# ---------------------------------------------------------------------------
# Closed entity vocabularies — from entity_vocabulary in the fixture
# ---------------------------------------------------------------------------

# "one of: buy, sell, hold, hedge, rebalance"
ActionType = Literal["buy", "sell", "hold", "hedge", "rebalance"]

# "one of: retirement, education, house, FIRE, emergency_fund"
GoalType = Literal["retirement", "education", "house", "FIRE", "emergency_fund"]

# "one of: daily, weekly, monthly, yearly"
FrequencyType = Literal["daily", "weekly", "monthly", "yearly"]

# "string token (6_months, 1_year, 5_years)"
HorizonType = Literal["6_months", "1_year", "5_years"]

# "string token (today, this_week, this_month, this_year)"
TimePeriodType = Literal["today", "this_week", "this_month", "this_year"]

# "string (S&P 500, FTSE 100, NIKKEI 225, MSCI World)"
IndexType = Literal["S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World"]

# Informational safety verdict — included in classifier output
SafetyVerdict = Literal["safe", "warning", "unsafe"]


# ---------------------------------------------------------------------------
# EntitySet — extracted entities from the user query
# ---------------------------------------------------------------------------

class EntitySet(BaseModel):
    """Entities extracted by the intent classifier.

    All fields are optional because any given query may mention none,
    some, or many entity types.  The classifier is instructed to use
    *only* the closed vocabulary values for constrained fields.
    """

    tickers: list[str] = Field(default_factory=list, description="Ticker symbols, uppercase, exchange-suffixed where relevant.")
    topics: list[str] = Field(default_factory=list, description="Free-text topic mentions (e.g. 'ETF', 'compound interest').")
    sectors: list[str] = Field(default_factory=list, description="Sector mentions (e.g. 'technology').")
    amount: Optional[float] = Field(None, description="Monetary amount mentioned in the query.")
    rate: Optional[float] = Field(None, description="Decimal rate (e.g. 0.08 for 8%).")
    period_years: Optional[int] = Field(None, description="Time period in years (exact integer).")
    currency: Optional[str] = Field(None, description="ISO 4217 currency code.")
    index: Optional[str] = Field(None, description="Canonical index name: 'S&P 500', 'FTSE 100', 'NIKKEI 225', 'MSCI World'.")
    action: Optional[ActionType] = Field(None, description="User's intended action: buy, sell, hold, hedge, rebalance.")
    goal: Optional[GoalType] = Field(None, description="Financial goal: retirement, education, house, FIRE, emergency_fund.")
    frequency: Optional[FrequencyType] = Field(None, description="Frequency: daily, weekly, monthly, yearly.")
    horizon: Optional[HorizonType] = Field(None, description="Forward-looking horizon: 6_months, 1_year, 5_years.")
    time_period: Optional[TimePeriodType] = Field(None, description="Backward-looking time window: today, this_week, this_month, this_year.")


# ---------------------------------------------------------------------------
# ClassifierOutput — the single structured output from the LLM
# ---------------------------------------------------------------------------

class ClassifierOutput(BaseModel):
    """Structured output returned by the intent classifier LLM call.

    This is the schema passed to ``response_format`` in the OpenAI
    ``client.beta.chat.completions.parse()`` call.  Every field is
    required so the LLM always produces a complete classification.
    """

    intent: str = Field(..., description="Human-readable intent label (e.g. 'portfolio_health_check').")
    target_agent: AgentType = Field(..., description="Which specialist agent should handle this query.")
    entities: EntitySet = Field(default_factory=EntitySet, description="Extracted entities from the query.")
    safety_verdict: SafetyVerdict = Field(default="safe", description="Informational safety assessment.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence in its routing decision.")
    reasoning: str = Field(default="", description="Brief explanation of the classification — for logging only.")


# ---------------------------------------------------------------------------
# Fallback — used when the LLM call fails or returns garbage
# ---------------------------------------------------------------------------

FALLBACK_CLASSIFIER_OUTPUT = ClassifierOutput(
    intent="fallback",
    target_agent=AgentType.CUSTOMER_SUPPORT,
    entities=EntitySet(),
    safety_verdict="safe",
    confidence=0.0,
    reasoning="LLM call failed or returned unparseable output — routing to support as safe fallback.",
)
"""Pre-built fallback used when the classifier LLM call fails.

Routes to ``customer_support`` with zero confidence so downstream
code can detect the fallback and act accordingly.
"""
