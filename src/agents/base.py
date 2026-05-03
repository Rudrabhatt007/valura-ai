"""Base agent — abstract interface for all specialist agents.

Every specialist agent (portfolio health, market research, etc.) must
subclass ``BaseAgent`` and implement the ``run()`` async generator.
This guarantees a uniform streaming contract across the entire system.

The ``run()`` method yields SSE-formatted strings so the HTTP layer
can forward them directly to the client without transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from src.classifier.schema import EntitySet
from src.models.user import UserProfile


class BaseAgent(ABC):
    """Abstract base class for all specialist agents.

    Subclasses must implement ``run()`` as an async generator that
    yields SSE-formatted string chunks.  The HTTP layer iterates
    over this generator and forwards each chunk to the client.

    Example subclass::

        class PortfolioHealthAgent(BaseAgent):
            async def run(self, query, entities, user_profile, session_history):
                analysis = await self._analyse(user_profile)
                yield json.dumps(analysis)
    """

    @abstractmethod
    async def run(
        self,
        query: str,
        entities: EntitySet,
        user_profile: UserProfile,
        session_history: list[dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        """Execute the agent and stream response chunks.

        Parameters
        ----------
        query:
            The user's natural-language query.
        entities:
            Structured entities extracted by the classifier.
        user_profile:
            The full user profile, including portfolio holdings.
        session_history:
            Prior conversation turns as
            ``[{"role": "user"|"assistant", "content": "..."}]``.

        Yields
        ------
        str
            SSE-formatted response chunks.  Each yielded string becomes
            the ``data:`` field of an ``event: chunk`` SSE frame.
        """
        # Abstract generator — yield to satisfy the type checker.
        yield ""  # pragma: no cover
