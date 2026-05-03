"""Entity and agent matching utilities — implements fixture grading rules.

These matchers implement the **exact** rules from ``fixtures/README.md``:

- **Agent**: exact string match
- **Tickers**: case-folded, exchange-suffix stripped, subset match
- **Topics / Sectors**: case-folded substring match per element
- **Amount / Rate**: within ±5%
- **period_years**: exact integer match
- **currency**: exact ISO 4217 match
- **index**: exact match against canonical names
- **action, goal, frequency, horizon, time_period**: exact match
  against the closed vocabulary

All matchers follow the **subset** principle: every expected value must
be found in the predicted output, but extra predicted values are fine.
"""

from __future__ import annotations

from src.classifier.schema import EntitySet


# ---------------------------------------------------------------------------
# Atomic matchers
# ---------------------------------------------------------------------------

def match_agent(predicted: str, expected: str) -> bool:
    """Exact string match for agent routing.

    Parameters
    ----------
    predicted:
        The agent name produced by the classifier.
    expected:
        The gold-standard agent name from the fixture.

    Returns
    -------
    bool
    """
    return predicted == expected


def _normalise_ticker(ticker: str) -> str:
    """Normalise a ticker for comparison.

    Rules from fixtures/README.md:
    - Case-fold to uppercase
    - Strip exchange suffix after '.' (ASML.AS → ASML)
    """
    upper = ticker.upper().strip()
    if "." in upper:
        return upper.split(".")[0]
    return upper


def match_tickers(predicted_list: list[str], expected_list: list[str]) -> bool:
    """Subset match for tickers with normalisation.

    Every expected ticker must appear in the predicted list after
    normalisation. Extra predicted tickers are allowed.

    Parameters
    ----------
    predicted_list:
        Tickers from the classifier output.
    expected_list:
        Tickers from the fixture gold standard.

    Returns
    -------
    bool
    """
    normalised_predicted = {_normalise_ticker(t) for t in predicted_list}
    for expected in expected_list:
        if _normalise_ticker(expected) not in normalised_predicted:
            return False
    return True


def match_string_list(predicted_list: list[str], expected_list: list[str]) -> bool:
    """Subset match for topics / sectors with case-folded substring matching.

    For each expected element, at least one predicted element must
    contain it as a substring (case-insensitive).

    Parameters
    ----------
    predicted_list:
        Values from the classifier output.
    expected_list:
        Values from the fixture gold standard.

    Returns
    -------
    bool
    """
    lower_predicted = [p.lower() for p in predicted_list]
    for expected in expected_list:
        expected_lower = expected.lower()
        if not any(expected_lower in p for p in lower_predicted):
            return False
    return True


def match_numeric(predicted: float | None, expected: float | None, tolerance: float = 0.05) -> bool:
    """Numeric match within ±tolerance (default 5%).

    Parameters
    ----------
    predicted:
        Value from the classifier output.
    expected:
        Value from the fixture gold standard.
    tolerance:
        Relative tolerance (0.05 = 5%).

    Returns
    -------
    bool
    """
    if expected is None:
        return True  # Nothing to check.
    if predicted is None:
        return False  # Expected a value but got nothing.
    denominator = max(abs(expected), 1e-9)
    return abs(predicted - expected) / denominator <= tolerance


def match_exact(predicted: str | None, expected: str | None) -> bool:
    """Case-insensitive exact match.

    Parameters
    ----------
    predicted:
        Value from the classifier output.
    expected:
        Value from the fixture gold standard.

    Returns
    -------
    bool
    """
    if expected is None:
        return True
    if predicted is None:
        return False
    return predicted.lower() == expected.lower()


def match_index(predicted: str | None, expected: str | None) -> bool:
    """Exact match for canonical index names.

    Index names must match exactly (case-insensitive) against the
    canonical names: 'S&P 500', 'FTSE 100', 'NIKKEI 225', 'MSCI World'.

    Parameters
    ----------
    predicted:
        Index name from classifier output.
    expected:
        Index name from fixture.

    Returns
    -------
    bool
    """
    if expected is None:
        return True
    if predicted is None:
        return False
    return predicted.strip().lower() == expected.strip().lower()


# ---------------------------------------------------------------------------
# Composite entity matcher
# ---------------------------------------------------------------------------

# Maps entity field names to their match functions.
# Each entry: (field_name, matcher_function, is_list_field)
_FIELD_MATCHERS = {
    "tickers": ("list", match_tickers),
    "topics": ("list", match_string_list),
    "sectors": ("list", match_string_list),
    "amount": ("numeric", match_numeric),
    "rate": ("numeric", match_numeric),
    "period_years": ("exact_int", None),  # Special: exact integer match
    "currency": ("exact", match_exact),
    "index": ("exact", match_index),
    "action": ("exact", match_exact),
    "goal": ("exact", match_exact),
    "frequency": ("exact", match_exact),
    "horizon": ("exact", match_exact),
    "time_period": ("exact", match_exact),
}


def match_entities(predicted: EntitySet, expected: dict) -> bool:
    """Match predicted entities against expected gold-standard entities.

    Runs the correct matcher per field type.  Missing optional fields
    in ``expected`` are skipped.  Returns ``True`` only if ALL present
    expected fields match.

    Parameters
    ----------
    predicted:
        The ``EntitySet`` from the classifier output.
    expected:
        The ``expected_entities`` dict from the fixture.

    Returns
    -------
    bool
    """
    if not expected:
        return True  # No entities to check.

    predicted_dict = predicted.model_dump()

    for field, expected_value in expected.items():
        if field.startswith("_"):
            continue  # Skip metadata fields like _notes.

        if field not in _FIELD_MATCHERS:
            continue  # Unknown field — skip gracefully.

        match_type, matcher = _FIELD_MATCHERS[field]
        predicted_value = predicted_dict.get(field)

        if match_type == "list":
            pred_list = predicted_value if predicted_value else []
            exp_list = expected_value if isinstance(expected_value, list) else [expected_value]
            if not matcher(pred_list, exp_list):
                return False

        elif match_type == "numeric":
            if not matcher(predicted_value, expected_value):
                return False

        elif match_type == "exact_int":
            # period_years: exact integer match.
            if expected_value is not None:
                if predicted_value is None or predicted_value != expected_value:
                    return False

        elif match_type == "exact":
            if not matcher(predicted_value, expected_value):
                return False

    return True
