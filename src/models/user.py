"""User profile models — derived from fixture data in ``fixtures/users/``.

These Pydantic v2 models are the canonical representation of user-side
data throughout the pipeline.  Every field was derived by inspecting all
five fixture profiles:

- ``user_001_active_trader_us.json``  — aggressive, 9 US holdings
- ``user_003_concentrated.json``      — moderate, NVDA-heavy
- ``user_004_empty.json``             — zero positions (edge case)
- ``user_006_multi_currency.json``    — multi-currency, Singapore-based
- ``user_008_retiree.json``           — conservative, dividend-focused

Design decisions
~~~~~~~~~~~~~~~~
* **Decimal for monetary values** — ``avg_cost`` uses ``Decimal`` to
  avoid floating-point artefacts in financial maths.
* **Optional fields** — anything that appears in fewer than all five
  profiles is ``Optional`` (e.g. ``reporting_currency``, ``income_focus``).
* **Empty positions list** — ``user_004`` proves this must be valid.
* **Flat risk_profile** — fixtures use a plain string, not a nested
  object.  We model it as a ``Literal`` with known values.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Holding — a single position in the user's portfolio
# ---------------------------------------------------------------------------

class Holding(BaseModel):
    """A single portfolio position.

    Mirrors the ``positions[]`` entries in fixture user profiles.
    ``currency`` is per-holding because ``user_006`` holds positions
    in USD, EUR, GBP, and JPY simultaneously.
    """

    ticker: str = Field(..., description="Instrument ticker, with exchange suffix if non-US (e.g. 'ASML.AS').")
    exchange: str = Field(..., description="Exchange identifier (e.g. 'NASDAQ', 'LSE', 'TSE').")
    quantity: int | float = Field(..., description="Number of shares/units held.")
    avg_cost: Decimal = Field(..., description="Average cost basis per share in the holding's currency.")
    currency: str = Field(..., description="ISO 4217 currency code for this holding.")
    purchased_at: date = Field(..., description="Date the position was opened.")


# ---------------------------------------------------------------------------
# KYC — know-your-customer status
# ---------------------------------------------------------------------------

class KYCStatus(BaseModel):
    """KYC verification status.

    All five fixture profiles have ``{"status": "verified"}``.
    We keep the model flexible for future values (e.g. ``pending``,
    ``rejected``).
    """

    status: str = Field(..., description="KYC status — e.g. 'verified', 'pending', 'rejected'.")


# ---------------------------------------------------------------------------
# Preferences — user-configurable display/reporting options
# ---------------------------------------------------------------------------

class Preferences(BaseModel):
    """User display and reporting preferences.

    ``preferred_benchmark`` appears in all profiles.  Other fields are
    profile-specific and therefore ``Optional``.
    """

    preferred_benchmark: str = Field(..., description="Canonical benchmark name (e.g. 'S&P 500', 'QQQ', 'MSCI World').")
    reporting_currency: Optional[str] = Field(None, description="Override currency for reporting (ISO 4217). Only user_006 sets this.")
    income_focus: Optional[bool] = Field(None, description="Whether the user prioritises income/yield. Only user_008 sets this.")


# ---------------------------------------------------------------------------
# UserProfile — top-level container
# ---------------------------------------------------------------------------

# Risk tolerance values observed across the five fixture profiles.
RiskTolerance = Literal["aggressive", "moderate", "conservative"]


class UserProfile(BaseModel):
    """Full user profile as passed into the pipeline.

    Fields are derived from the union of all five fixture files.
    ``positions`` may be an empty list (``user_004_empty``).
    """

    user_id: str = Field(..., description="Stable user identifier (e.g. 'usr_001').")
    name: str = Field(..., description="Display name.")
    age: int = Field(..., description="User's age in years.")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code.")
    base_currency: str = Field(..., description="User's home currency (ISO 4217).")
    kyc: KYCStatus = Field(..., description="KYC verification status.")
    risk_profile: RiskTolerance = Field(..., description="Self-reported risk tolerance level.")
    positions: list[Holding] = Field(default_factory=list, description="Current portfolio holdings. May be empty.")
    preferences: Preferences = Field(..., description="User display and reporting preferences.")
