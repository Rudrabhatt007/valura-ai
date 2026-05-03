"""API request/response models — QueryRequest, SSEEvent, PipelineMetadata.

These Pydantic models define the HTTP contract between clients and the
Valura AI microservice.  All responses flow through Server-Sent Events
(SSE), so the models here describe the *payloads* inside SSE frames,
not traditional JSON response bodies.

SSE event lifecycle for a successful request::

    event: metadata   → PipelineMetadata (classifier result + routing info)
    event: chunk      → streamed agent output (one or more)
    event: done       → empty payload, signals stream end

SSE event for a blocked or errored request::

    event: error      → {"error": "...", "category": "..."}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.models.user import UserProfile


# ---------------------------------------------------------------------------
# Inbound — what the client sends
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for ``POST /api/v1/query``.

    The client is responsible for attaching the full ``UserProfile`` so
    the pipeline has all user context without an extra DB lookup.
    """

    user_id: str = Field(..., description="Stable user identifier.")
    session_id: str = Field(..., description="Conversation session identifier — groups turns.")
    query: str = Field(..., min_length=1, description="The user's natural-language query.")
    user_profile: UserProfile = Field(..., description="Full user profile including portfolio and preferences.")


# ---------------------------------------------------------------------------
# SSE building blocks — internal, not exposed as HTTP schemas
# ---------------------------------------------------------------------------

class SSEEvent(BaseModel):
    """Internal helper for constructing SSE payloads.

    ``event`` maps to the SSE ``event:`` field.  ``data`` is serialised
    to JSON and placed in the SSE ``data:`` field.

    Recognised event types:

    * ``metadata`` — first event, carries classifier + routing info
    * ``chunk``    — streamed agent output fragment
    * ``error``    — blocked by safety guard or pipeline failure
    * ``done``     — stream complete, no further events
    """

    event: str = Field(..., description="SSE event name: 'metadata', 'chunk', 'error', or 'done'.")
    data: dict[str, Any] | str = Field(..., description="Event payload — dict is JSON-serialised, str sent as-is.")


# ---------------------------------------------------------------------------
# Pipeline metadata — first SSE event on a successful request
# ---------------------------------------------------------------------------

class PipelineMetadata(BaseModel):
    """Sent as the first ``event: metadata`` SSE frame.

    Gives the client immediate visibility into how the query was
    classified and where it is being routed, before the agent starts
    streaming its response.
    """

    session_id: str = Field(..., description="Echo of the request session ID.")
    classified_intent: str = Field(..., description="Intent label produced by the classifier.")
    target_agent: str = Field(..., description="Agent that will handle this query.")
    entities: dict[str, Any] = Field(default_factory=dict, description="Extracted entities from the classifier.")
    safety_verdict: str = Field(default="safe", description="Informational safety verdict: 'safe', 'warning', or 'unsafe'.")
