"""Safety guard — pure-Python pre-LLM filter for harmful financial queries.

This module runs **before** the classifier LLM call and blocks queries
that express harmful intent across these categories (from fixture data):

- ``insider_trading``       — acting on material non-public information
- ``market_manipulation``   — pump-and-dump, spoofing, wash trading
- ``money_laundering``      — structuring, hiding fund sources
- ``guaranteed_returns``    — promising specific returns on investments
- ``reckless_advice``       — all-in bets with retirement/emergency money
- ``sanctions_evasion``     — bypassing OFAC/trade sanctions
- ``fraud``                 — forging documents, fake claims

Design decisions
~~~~~~~~~~~~~~~~
* **No LLM, no network, no I/O** — pure regex + keyword matching.
* **All patterns compiled at module load time** — zero per-request cost.
* **Educational exception** — queries framed as educational ("what is",
  "explain", "how does") pass through unless they also contain
  first-person action intent ("help me", "i want to").
* **Each category has a distinct refusal** — not a generic message.
* **Under 10ms** for any input on commodity hardware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SafetyResult — returned by check()
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SafetyResult:
    """Result of the safety guard check.

    Attributes
    ----------
    passed:
        ``True`` if the query is safe to process; ``False`` if blocked.
    category:
        The blocking category (e.g. ``"insider_trading"``), or ``None``
        if the query passed.
    refusal_message:
        A professional, category-specific refusal message, or ``None``
        if the query passed.
    """

    passed: bool
    category: str | None = None
    refusal_message: str | None = None


# ---------------------------------------------------------------------------
# Category-specific refusal messages — professional, not generic
# ---------------------------------------------------------------------------

_REFUSAL_MESSAGES: dict[str, str] = {
    "insider_trading": (
        "I cannot assist with trading based on material non-public "
        "information. Insider trading is illegal under securities law "
        "in all major jurisdictions. If you have questions about "
        "insider trading regulations, I'm happy to explain them."
    ),
    "market_manipulation": (
        "I cannot help with strategies designed to artificially influence "
        "market prices or trading volumes. Market manipulation — including "
        "pump-and-dump schemes, spoofing, and wash trading — carries severe "
        "civil and criminal penalties."
    ),
    "money_laundering": (
        "I cannot assist with concealing the source, movement, or "
        "destination of funds, or with evading reporting requirements. "
        "Anti-money-laundering regulations exist to protect the financial "
        "system. If you have questions about AML compliance, I can help."
    ),
    "guaranteed_returns": (
        "I cannot promise or guarantee specific investment returns. "
        "All investments carry risk, and any claim of guaranteed returns "
        "is a hallmark of fraud. I can discuss historical performance "
        "and risk-adjusted expectations instead."
    ),
    "reckless_advice": (
        "I cannot recommend putting all your savings, emergency funds, "
        "or retirement assets into a single high-risk position. This "
        "level of concentration could cause serious financial harm. "
        "I can help you explore a diversified approach instead."
    ),
    "sanctions_evasion": (
        "I cannot assist with circumventing economic sanctions or OFAC "
        "restrictions. Sanctions evasion is a federal crime with severe "
        "penalties. If you have questions about sanctions compliance, "
        "I'm happy to explain how screening works."
    ),
    "fraud": (
        "I cannot help with creating fraudulent documents, false claims, "
        "or any form of financial deception. If you need help with "
        "legitimate financial documentation, I'm here to assist."
    ),
}

# ---------------------------------------------------------------------------
# Educational framing detection
# ---------------------------------------------------------------------------

# Queries that start with or contain these markers are likely educational.
_EDUCATIONAL_MARKERS: tuple[str, ...] = (
    "what is",
    "what are",
    "what's the difference",
    "what does",
    "explain",
    "how does",
    "how do regulators",
    "how do brokerages",
    "how do brokers",
    "teach me",
    "why do people",
    "why is",
    "why are",
    "history of",
    "definition of",
    "describe",
    "is it ever legitimate",
    "is it legal",
    "is it illegal",
    "what compliance",
    "what are the penalties",
    "what is the penalty",
    "what are the risks",
    "how should",
    "what factors should",
    "how do regulators",
    "how does the sec",
    "how does the fca",
    "what are the disclosure",
    "what are the aml",
    "are pump-and-dump",
)

# First-person action intent — if present alongside harmful content,
# the educational exception does NOT apply.
_ACTION_MARKERS: tuple[str, ...] = (
    "help me",
    "i want to",
    "i need to",
    "can you help",
    "show me how to",
    "do this for me",
    "do it",
    "execute",
    "place ",
    "make me",
    "tell me to",
    "tell me which",
    "tell me a stock",
    "i'm ",  # "i'm 70 and want to put all..."
    "should i put",
    "when should i buy",
    "should i load up",
    "give me a",
    "guarantee me",
    "promise me",
    "design a",
    "draft a",
    "how do i",
    "how can i",
)


def _has_educational_framing(text: str) -> bool:
    """Check if the query is framed as an educational question."""
    return any(marker in text for marker in _EDUCATIONAL_MARKERS)


def _has_action_intent(text: str) -> bool:
    """Check if the query contains first-person action intent."""
    return any(marker in text for marker in _ACTION_MARKERS)


# ---------------------------------------------------------------------------
# Harmful pattern definitions — compiled at module load time
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _HarmfulPattern:
    """A compiled regex pattern associated with a safety category."""

    category: str
    pattern: re.Pattern[str]


def _compile_patterns() -> list[_HarmfulPattern]:
    """Build and compile all harmful-content patterns.

    Called once at module load time.  Patterns are ordered by category.
    Each pattern is designed to catch the *intent* to act, not merely
    the mention of a topic.
    """
    raw: list[tuple[str, str]] = [
        # --- insider_trading ---
        ("insider_trading", r"\b(?:unannounced|non[- ]?public|confidential|inside)\b.*\b(?:buy|sell|trade|shares|stock|acquisition|merger)\b"),
        ("insider_trading", r"\b(?:tip|tipped off)\b.*\b(?:earnings|announcement|merger|acquisition)\b.*\b(?:buy|sell|load up|before)\b"),
        ("insider_trading", r"\b(?:buy|sell|load up|trade)\b.*\b(?:before|ahead of)\b.*\b(?:announcement|earnings|call)\b"),
        ("insider_trading", r"\b(?:earnings|announcement)\b.*\btomorrow\b"),
        ("insider_trading", r"\btrade on\b.*\b(?:confidential|non[- ]?public|inside|insider)\b"),
        ("insider_trading", r"\b(?:gave me a tip|friend at|work at)\b.*\b(?:buy|sell|shares|stock|load up)\b"),
        ("insider_trading", r"\bneed to know the earnings\b"),

        # --- market_manipulation ---
        ("market_manipulation", r"\bpump\s*(?:up|and dump)?\b.*\b(?:price|stock)\b"),
        ("market_manipulation", r"\bcoordinated\s+(?:buying|selling|scheme)\b"),
        ("market_manipulation", r"\bwash\s+trad(?:e|ing)\b(?!.*\billegal\b)(?!.*\bwhy\b)"),
        ("market_manipulation", r"\b(?:move|push)\s+(?:this|the|a)\s+stock\b"),
        ("market_manipulation", r"\bcreate\s+(?:fake\s+)?volume\b"),
        ("market_manipulation", r"\bartificially\s+(?:inflate|deflate|move)\b"),

        # --- money_laundering ---
        ("money_laundering", r"\bwithout\s+(?:it\s+being\s+)?report(?:ing|ed)\b"),
        ("money_laundering", r"\bavoid\b.*\breporting\s+threshold\b"),
        ("money_laundering", r"\bstructure\s+(?:deposits|transactions|payments)\b"),
        ("money_laundering", r"\b(?:obscure|hide|conceal)\b.*\b(?:source|funds|profits|money|origin)\b"),
        ("money_laundering", r"\b(?:move|transfer)\b.*\bcash\b.*\bwithout\b"),
        ("money_laundering", r"\bhide\b.*\b(?:trading|investment)\s+profits\b"),
        ("money_laundering", r"\bhide\b.*\bfrom\b.*\b(?:tax|authorit|irs|hmrc)\b"),
        ("money_laundering", r"\blayer\b.*\btrades?\b.*\bobscure\b"),

        # --- guaranteed_returns ---
        ("guaranteed_returns", r"\bguarantee\s+(?:me|us|my)\b"),
        ("guaranteed_returns", r"\bpromise\s+(?:me|us|my)\b"),
        ("guaranteed_returns", r"\b100\s*%\s*(?:certain|sure|guaranteed|safe)\b"),
        ("guaranteed_returns", r"\bfoolproof\s+(?:way|method|strategy)\b"),
        ("guaranteed_returns", r"\bcertain\s+to\s+(?:go up|increase|rise|double)\b"),
        ("guaranteed_returns", r"\bguaranteed?\s+(?:\d+\s*%|returns?|profit|gains?)\b"),
        ("guaranteed_returns", r"\bwill\s+(?:definitely|certainly|surely)\s+(?:go up|double|triple)\b"),
        ("guaranteed_returns", r"\bno[- ]?risk\s+(?:profit|returns?|investment)\b"),

        # --- reckless_advice ---
        ("reckless_advice", r"\b(?:all|entire|whole)\b.*\b(?:retirement|savings|emergency\s+fund|pension|nest\s+egg)\b.*\b(?:into|in|on|to buy)\b"),
        ("reckless_advice", r"\bmargin\s+loan\b.*\b(?:buy|more|load)\b"),
        ("reckless_advice", r"\bmortgage\s+(?:my|the)\s+house\b"),
        ("reckless_advice", r"\b(?:emergency\s+fund)\b.*\b(?:into|in)\b.*\b(?:options|crypto|bitcoin|meme)\b"),
        ("reckless_advice", r"\btell\s+me\s+(?:to\s+take|which\s+stock)\b.*\b(?:margin|loan|borrow|mortgage)\b"),
        ("reckless_advice", r"\btell\s+me\s+which\s+stock\s+to\s+mortgage\b"),

        # --- sanctions_evasion ---
        ("sanctions_evasion", r"\b(?:bypass|circumvent|evade|avoid|get around)\b.*\b(?:sanctions?|ofac)\b"),
        ("sanctions_evasion", r"\bsanctioned?\b.*\bwithout\s+(?:it\s+)?being\s+traced\b"),
        ("sanctions_evasion", r"\bshell\s+company\b.*\bsanctions?\b"),
        ("sanctions_evasion", r"\broute\s+(?:trades?|money|funds)\b.*\b(?:sanctions?|ofac)\b"),

        # --- fraud ---
        ("fraud", r"\b(?:draft|create|forge|fake)\s+(?:a\s+)?(?:fake|false|fraudulent|forged)?\s*(?:contract|document|note|receipt|statement)\b"),
        ("fraud", r"\bclaim\s+(?:fake|false|fraudulent)\s+(?:losses|gains|expenses)\b"),
        ("fraud", r"\bfake\s+(?:contract|document)\b"),
    ]

    return [
        _HarmfulPattern(category=cat, pattern=re.compile(pat, re.IGNORECASE))
        for cat, pat in raw
    ]


# Compiled at module load time — zero per-request compilation cost.
_HARMFUL_PATTERNS: list[_HarmfulPattern] = _compile_patterns()

# Safe result singleton — reused for every passing query.
_SAFE_RESULT = SafetyResult(passed=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(query: str) -> SafetyResult:
    """Check a query against the safety guard.

    Parameters
    ----------
    query:
        The user's natural-language query.  Handles ``None``, empty
        strings, and unicode without crashing.

    Returns
    -------
    SafetyResult
        ``passed=True`` if safe; ``passed=False`` with ``category``
        and ``refusal_message`` if blocked.
    """
    # Guard against None and empty input.
    if not query or not isinstance(query, str):
        return _SAFE_RESULT

    # Normalise: lowercase, collapse whitespace.
    text = " ".join(query.lower().split())

    if not text:
        return _SAFE_RESULT

    # Check for harmful patterns.
    for hp in _HARMFUL_PATTERNS:
        if hp.pattern.search(text):
            # Educational exception: if query is framed educationally
            # AND has no first-person action intent, let it through.
            if _has_educational_framing(text) and not _has_action_intent(text):
                continue

            return SafetyResult(
                passed=False,
                category=hp.category,
                refusal_message=_REFUSAL_MESSAGES.get(
                    hp.category,
                    "I cannot assist with this request as it may involve "
                    "prohibited financial activity.",
                ),
            )

    return _SAFE_RESULT
