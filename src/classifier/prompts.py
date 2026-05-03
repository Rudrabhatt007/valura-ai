"""Classifier prompts — system prompt builder and history formatter.

This module builds the system prompt that drives the intent classifier
LLM call.  The prompt encodes:

1. The agent taxonomy (exact names + descriptions from fixtures)
2. The closed entity vocabulary (exact allowed values)
3. Follow-up resolution rules for multi-turn conversations
4. Entity extraction rules (tickers, amounts, etc.)
5. The safety verdict semantics (informational only)

The prompt is constructed as a function so tests can inspect it and
the vocabulary can be extended without touching string literals.
"""

from __future__ import annotations

from src.classifier.schema import AgentType


# ---------------------------------------------------------------------------
# Agent descriptions — from intent_classification.json agent_taxonomy
# ---------------------------------------------------------------------------

AGENT_DESCRIPTIONS: dict[str, str] = {
    AgentType.PORTFOLIO_HEALTH.value: (
        "Structured assessment of the user's portfolio — concentration risk, "
        "performance metrics, benchmark comparison, and actionable observations."
    ),
    AgentType.MARKET_RESEARCH.value: (
        "Factual / recent information about a specific instrument, sector, "
        "or market event. Price checks, news, comparisons."
    ),
    AgentType.INVESTMENT_STRATEGY.value: (
        "Advice and strategy questions — should I buy / sell / rebalance, "
        "allocation guidance, equity-bond splits."
    ),
    AgentType.FINANCIAL_PLANNING.value: (
        "Long-term financial planning — retirement, education savings, "
        "house deposit goals, FIRE plans, savings rates."
    ),
    AgentType.FINANCIAL_CALCULATOR.value: (
        "Deterministic numerical computations — DCA returns, mortgage "
        "payments, tax calculations, future value, FX conversion."
    ),
    AgentType.RISK_ASSESSMENT.value: (
        "Risk metrics and exposure analysis — beta, max drawdown, "
        "stress tests, what-if scenarios, currency exposure."
    ),
    AgentType.PRODUCT_RECOMMENDATION.value: (
        "Recommend specific products, funds, or ETFs that match the "
        "user's profile and stated criteria."
    ),
    AgentType.PREDICTIVE_ANALYSIS.value: (
        "Forward-looking analysis — forecasts, trend extrapolation, "
        "portfolio value projections."
    ),
    AgentType.CUSTOMER_SUPPORT.value: (
        "Platform issues, account questions, how-to-use-the-app, "
        "login problems, transaction history."
    ),
    AgentType.GENERAL_QUERY.value: (
        "Educational questions, conversational exchanges, definitions, "
        "greetings, and anything that doesn't fit another agent."
    ),
}

# ---------------------------------------------------------------------------
# Entity vocabulary — closed sets from intent_classification.json
# ---------------------------------------------------------------------------

ENTITY_VOCABULARY: dict[str, list[str]] = {
    "action": ["buy", "sell", "hold", "hedge", "rebalance"],
    "goal": ["retirement", "education", "house", "FIRE", "emergency_fund"],
    "frequency": ["daily", "weekly", "monthly", "yearly"],
    "horizon": ["6_months", "1_year", "5_years"],
    "time_period": ["today", "this_week", "this_month", "this_year"],
    "index": ["S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World"],
}


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    agent_descriptions: dict[str, str] | None = None,
    entity_vocabulary: dict[str, list[str]] | None = None,
) -> str:
    """Build the complete system prompt for the intent classifier.

    Parameters
    ----------
    agent_descriptions:
        Mapping of agent name → one-line description.  Defaults to
        ``AGENT_DESCRIPTIONS``.
    entity_vocabulary:
        Mapping of entity field → list of allowed values.  Defaults to
        ``ENTITY_VOCABULARY``.

    Returns
    -------
    str
        The full system prompt string.
    """
    agents = agent_descriptions or AGENT_DESCRIPTIONS
    vocab = entity_vocabulary or ENTITY_VOCABULARY

    # --- Agent taxonomy section ---
    agent_lines = "\n".join(
        f"  - {name}: {desc}" for name, desc in agents.items()
    )

    # --- Entity vocabulary section ---
    vocab_lines = "\n".join(
        f"  - {field}: allowed values are EXACTLY {values}"
        for field, values in vocab.items()
    )

    return f"""\
You are the intent classifier for Valura, a global wealth management platform.
Your job is to classify every user query into exactly ONE agent and extract structured entities.

═══ AGENT TAXONOMY ═══
Route each query to exactly one of these agents:
{agent_lines}

Routing rules:
- If a query spans multiple intents, pick the PRIMARY one (the main thing the user wants).
- If the query is just a ticker symbol with no verb, route to market_research.
- If the query is gibberish or unintelligible, route to general_query.
- Greetings ("hi", "hello", "thanks") go to general_query.

═══ ENTITY VOCABULARY (CLOSED SETS) ═══
For the following fields, you MUST use ONLY these exact values. Never paraphrase, synonymise, or invent new values:
{vocab_lines}

═══ ENTITY EXTRACTION RULES ═══
- tickers: UPPERCASE. Include exchange suffix if the user mentions it (e.g. ASML.AS, HSBA.L, 7203.T). If the user says a company name, map it to the correct ticker.
- topics: free-text array — extract the subject matter (e.g. "ETF", "compound interest", "beta").
- sectors: free-text array — extract sector mentions (e.g. "technology", "healthcare").
- amount: extract the number only, no currency symbol or units. Parse "200k" as 200000, "1.5M" as 1500000.
- rate: express as a decimal (8% → 0.08, 6.5% → 0.065).
- period_years: integer number of years only.
- currency: ISO 4217 code (USD, EUR, GBP, JPY).
- index: use the EXACT canonical name from the vocabulary above.
- Do NOT hallucinate entities that are not present in the query or conversation history.
- Only extract entities that are explicitly mentioned or clearly implied.

═══ FOLLOW-UP RESOLUTION ═══
You will receive prior conversation turns as context.
- If the current query uses pronouns ("it", "that stock", "the same one") or implicit references, resolve them from the most recent relevant turn in the history.
- If the user switches topic entirely, start fresh — do NOT carry entities from prior turns.
- If unsure whether the user is continuing or switching, prefer starting fresh.

═══ SAFETY VERDICT ═══
This is informational only — it does NOT block the query (the safety guard already ran).
- "safe": no concerns whatsoever.
- "warning": borderline content — passed the guard but worth flagging (e.g. aggressive leverage, concentrated position advocacy).
- "unsafe": content that the guard should have caught — use this defensively if something slipped through.

═══ OUTPUT FORMAT ═══
Return a JSON object with exactly these fields:
- intent: a short human-readable label for what the user wants (e.g. "portfolio_health_check", "stock_price_lookup")
- target_agent: one of the agent names listed above (exact string match required)
- entities: an object with the extracted entities (only include fields that have values)
- safety_verdict: one of "safe", "warning", "unsafe"
- confidence: a float between 0.0 and 1.0 indicating your confidence in the routing decision
- reasoning: a brief one-sentence explanation of why you chose this agent (for logging, not shown to user)
"""


# ---------------------------------------------------------------------------
# History formatter — prepares conversation turns for the LLM
# ---------------------------------------------------------------------------

def format_history_messages(
    session_history: list[dict[str, str]],
    current_query: str,
    user_context_summary: str | None = None,
) -> list[dict[str, str]]:
    """Format conversation history into the OpenAI messages array.

    Parameters
    ----------
    session_history:
        Prior turns as ``[{"role": "user"|"assistant", "content": "..."}]``.
        Already in OpenAI-compatible format from the session store.
    current_query:
        The user's current query to classify.
    user_context_summary:
        Optional one-line summary of the user's portfolio context
        (e.g. "User has 9 US tech holdings, aggressive risk profile").

    Returns
    -------
    list[dict[str, str]]
        Messages array ready for the OpenAI chat completion call.
        The system prompt is NOT included — it is set separately.
    """
    messages: list[dict[str, str]] = []

    # Inject user context as a system-adjacent message so the LLM
    # knows who it is classifying for, without bloating the main prompt.
    if user_context_summary:
        messages.append({
            "role": "system",
            "content": f"[User context] {user_context_summary}",
        })

    # Prior turns — the LLM needs these to resolve follow-up references.
    for turn in session_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"],
        })

    # Current query — the one being classified.
    messages.append({
        "role": "user",
        "content": current_query,
    })

    return messages


def summarise_user_context(
    user_name: str,
    risk_profile: str,
    num_holdings: int,
    base_currency: str,
    country: str,
) -> str:
    """Build a concise user context summary for the classifier.

    This is injected into the messages array so the LLM can make
    context-aware routing decisions (e.g. routing an empty-portfolio
    user's "how am I doing?" to a BUILD-oriented response).

    Parameters
    ----------
    user_name:
        Display name.
    risk_profile:
        Risk tolerance level.
    num_holdings:
        Number of positions in the portfolio.
    base_currency:
        Home currency.
    country:
        Country code.

    Returns
    -------
    str
        A single-line summary string.
    """
    if num_holdings == 0:
        portfolio_desc = "empty portfolio (no positions yet)"
    else:
        portfolio_desc = f"{num_holdings} holdings"

    return (
        f"{user_name} | {risk_profile} risk | {portfolio_desc} | "
        f"{base_currency} | {country}"
    )
