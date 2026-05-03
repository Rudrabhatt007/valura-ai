"""Router — maps classified intents to agent instances.

Pure Python, no LLM.  The ``AGENT_REGISTRY`` maps every ``AgentType``
to a concrete ``BaseAgent`` subclass.  Initially all agents are stubbed;
``PortfolioHealthAgent`` will replace its stub in Day 2.

The ``StubAgent`` returns a structured "not implemented" SSE response
so the pipeline never crashes on unimplemented agents — the router's
job is to route correctly even when the destination is a stub.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import structlog

from src.agents.base import BaseAgent
from src.classifier.schema import AgentType, EntitySet
from src.models.user import UserProfile

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# StubAgent — placeholder for unimplemented agents
# ---------------------------------------------------------------------------

class StubAgent(BaseAgent):
    """Placeholder agent for intents that are not yet implemented.

    Returns a single structured JSON chunk indicating which agent
    *would* have handled the query, along with the classified intent
    and extracted entities.  This satisfies the assignment requirement
    that unimplemented agents return structured "not implemented"
    responses without crashing or returning errors.
    """

    async def run(
        self,
        query: str,
        entities: EntitySet,
        user_profile: UserProfile,
        session_history: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Yield a single structured "not implemented" response."""
        # Determine which agent this stub is standing in for.
        # The target_agent is injected by get_agent() via _agent_type.
        agent_name = getattr(self, "_agent_type", "unknown")

        response = {
            "status": "not_implemented",
            "classified_intent": query,
            "extracted_entities": entities.model_dump(exclude_none=True),
            "target_agent": agent_name,
            "message": (
                f"The {agent_name} agent is not available in this build. "
                f"Your query has been correctly classified and would be "
                f"handled by this agent in the full system."
            ),
        }
        yield json.dumps(response)


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
    ``StubAgent``.  The returned agent has ``_agent_type`` set so
    ``StubAgent`` can report which agent it is standing in for.

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
    agent = agent_cls()

    # Tag the agent so StubAgent can report its identity.
    agent._agent_type = agent_type.value  # type: ignore[attr-defined]

    logger.info(
        "router_dispatch",
        agent_type=agent_type.value,
        agent_class=agent_cls.__name__,
        is_stub=isinstance(agent, StubAgent),
    )

    return agent
