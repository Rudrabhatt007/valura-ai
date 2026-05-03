"""SSE formatting utilities — single source of truth for all SSE output.

Every module that constructs Server-Sent Events imports from here.
This ensures a consistent format across the entire pipeline:

    event: <event_name>
    data: <json_payload>
    \\n

Each event block ends with TWO newlines (``\\n\\n``).
The ``event:`` line comes before the ``data:`` line.
No extra whitespace. No BOM. No Content-Length header.

Architectural decision: SSE is the ONLY response mode.  Errors,
safety blocks, and pipeline state are all communicated through the
SSE stream — never via HTTP 4xx/5xx status codes for pipeline
failures (only for malformed requests caught by FastAPI validation).
"""

from __future__ import annotations

import json


def format_sse_event(event: str, data: dict | str) -> str:
    """Format a single SSE event string.

    Parameters
    ----------
    event:
        Event name (e.g. ``"metadata"``, ``"agent_response"``,
        ``"done"``).
    data:
        Payload — dict is serialised to JSON, str is used as-is.

    Returns
    -------
    str
        A properly formatted SSE event string ending with ``\\n\\n``.

    Examples
    --------
    >>> format_sse_event("metadata", {"intent": "portfolio_health"})
    'event: metadata\\ndata: {"intent": "portfolio_health"}\\n\\n'
    """
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = data
    return f"event: {event}\ndata: {data_str}\n\n"


def format_sse_error(
    message: str,
    code: str = "PIPELINE_ERROR",
) -> str:
    """Format a structured SSE error event.

    Parameters
    ----------
    message:
        Human-readable error message.
    code:
        Machine-readable error code for client-side handling.

    Returns
    -------
    str
        SSE event with ``event: error``.
    """
    return format_sse_event("error", {
        "error": True,
        "code": code,
        "message": message,
    })


def format_sse_done() -> str:
    """Format the terminal SSE done event.

    Signals to the client that the stream is complete and no
    further events will be sent.

    Returns
    -------
    str
        SSE event with ``event: done`` and ``status: complete``.
    """
    return format_sse_event("done", {"status": "complete"})


def format_sse_metadata(metadata: dict) -> str:
    """Format the pipeline metadata SSE event.

    This is always the first event in a successful pipeline
    execution.  Contains classifier output, routing info, and
    extracted entities.

    Parameters
    ----------
    metadata:
        Pipeline metadata dict (session_id, classified_intent,
        target_agent, entities, safety_verdict, confidence).

    Returns
    -------
    str
        SSE event with ``event: metadata``.
    """
    return format_sse_event("metadata", metadata)


def format_safety_block(category: str, message: str) -> str:
    """Format a safety guard refusal as an SSE event.

    Distinct from ``format_sse_error`` — clients can differentiate
    a safety block (content policy) from a pipeline error (infra).

    Parameters
    ----------
    category:
        The safety category that triggered the block (e.g.
        ``"insider_trading"``).
    message:
        The professional refusal message.

    Returns
    -------
    str
        SSE event with ``event: safety_block``.
    """
    return format_sse_event("safety_block", {
        "blocked": True,
        "category": category,
        "message": message,
    })
