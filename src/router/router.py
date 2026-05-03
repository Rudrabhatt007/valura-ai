"""Router — maps classified intents to agent instances.

Pure Python, no LLM.  The ``AGENT_REGISTRY`` maps every ``AgentType``
to a concrete ``BaseAgent`` subclass.  Initially all agents are stubbed;
``PortfolioHealthAgent`` will replace its stub in Day 2.

The ``StubAgent`` returns a structured "not implemented" SSE response
so the pipeline never crashes on unimplemented agents — the router's
job is to route correctly even when the destination is a stub.
"""

from __future__ import annotations

from typing import AsyncGenerator

import structlog

from src.agents.base import BaseAgent
from src.classifier.schema import AgentType, EntitySet
from src.models.user import UserProfile
from src.utils.sse import format_sse_event

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# StubAgent — placeholder for unimplemented agents
# ---------------------------------------------------------------------------

class StubAgent(BaseAgent):
    """Structured stub for unimplemented agents.

    Returns an informational not-implemented response as a properly
    formatted SSE event.  Never crashes.  Never raises.
    """

    async def run(
        self,
        query: str,
        entities: EntitySet,
        user_profile: UserProfile,
        session_history: list[dict[str, str]],
        *,
        classified_intent: str = "unknown",
        target_agent: str = "unknown",
    ) -> AsyncGenerator[str, None]:
        """Yield a single structured 'not implemented' SSE event."""
        payload = {
            "status": "not_implemented",
            "classified_intent": classified_intent,
            "extracted_entities": entities.model_dump(exclude_none=True),
            "target_agent": target_agent,
            "message": (
                f"The {target_agent} agent is not available in this build. "
                f"Your query has been classified and entities extracted. "
                f"This agent will be implemented in a future version."
            ),
            "build_version": "v0.1.0",
        }
        yield format_sse_event("agent_response", payload)


# ---------------------------------------------------------------------------
# Agent registry — maps AgentType → BaseAgent subclass
# ---------------------------------------------------------------------------

# Initially all agents map to StubAgent.  As agents are implemented,
# replace entries here (e.g. AgentType.PORTFOLIO_HEALTH → PortfolioHealthAgent).
AGENT_REGISTRY: dict[AgentType, type[BaseAgent]] = {
    AgentType.PORTFOLIO_HEALTH: StubAgent,
    AgentType.MARKET_RESEARCH: StubAgent,
    AgentType.INVESTMENT_STRATEGY: StubAgent,
    AgentType.FINANCIAL_PLANNING: StubAgent,
    AgentType.FINANCIAL_CALCULATOR: StubAgent,
    AgentType.RISK_ASSESSMENT: StubAgent,
    AgentType.PRODUCT_RECOMMENDATION: StubAgent,
    AgentType.PREDICTIVE_ANALYSIS: StubAgent,
    AgentType.CUSTOMER_SUPPORT: StubAgent,
    AgentType.GENERAL_QUERY: StubAgent,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_agent(agent_type: AgentType) -> BaseAgent:
    """Return an agent instance for the given agent type.

    Never raises — unknown or unregistered types fall back to
    ``StubAgent``.

    Parameters
    ----------
    agent_type:
        The ``AgentType`` enum value from the classifier output.

    Returns
    -------
    BaseAgent
        A concrete agent instance ready to call ``.run()``.
    """
    agent_cls = AGENT_REGISTRY.get(agent_type, StubAgent)

    if agent_cls is StubAgent and agent_type not in AGENT_REGISTRY:
        logger.warning(
            "agent_type_not_in_registry",
            agent_type=agent_type.value,
            fallback="StubAgent",
        )

    logger.info(
        "router_dispatch",
        agent_type=agent_type.value,
        agent_class=agent_cls.__name__,
        is_stub=issubclass(agent_cls, StubAgent),
    )

    return agent_cls()
