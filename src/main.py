"""FastAPI application — SSE pipeline endpoint.

Single endpoint ``POST /api/v1/query`` that runs the full pipeline:

1. Safety guard → block if harmful
2. Session history retrieval
3. Intent classification (LLM call)
4. Agent routing
5. Streamed response via Server-Sent Events

All responses flow through SSE — there is no JSON fallback path.
Errors are returned as structured ``event: error`` SSE frames.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.classifier.classifier import classify
from src.classifier.schema import ClassifierOutput
from src.config import get_settings
from src.memory.session import SessionStore
from src.models.api import PipelineMetadata, QueryRequest
from src.router.router import get_agent
from src.safety.guard import check as safety_check

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

import logging

def _configure_logging(log_level: str) -> None:
    """Configure structlog for structured JSON logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[log_level.upper()]
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — initialize shared resources on startup."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    # Initialize session store and attach to app state.
    db_url = settings.database_url
    if db_url and "sqlite" in db_url:
        db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
    else:
        db_path = ""

    if not db_path:
        db_path = "valura.db"  # Sensible default

    session_store = SessionStore(db_path)
    await session_store.initialize()
    app.state.session_store = session_store

    await logger.ainfo(
        "app_startup",
        environment=settings.environment,
        classifier_model=settings.classifier_model,
        pipeline_timeout=settings.pipeline_timeout_seconds,
    )

    yield  # Application runs here.

    await logger.ainfo("app_shutdown")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Valura AI",
    description="Financial intelligence microservice — AI co-investor for every user.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(event: str, data: dict | str) -> dict:
    """Build an SSE event dict for EventSourceResponse."""
    payload = json.dumps(data) if isinstance(data, dict) else data
    return {"event": event, "data": payload}


# ---------------------------------------------------------------------------
# Pipeline endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/query")
async def query_endpoint(request_body: QueryRequest, request: Request) -> EventSourceResponse:
    """Run the full AI pipeline and stream the response via SSE.

    Pipeline order:
    1. Safety guard check
    2. Session history retrieval
    3. Intent classification
    4. Agent routing + streaming
    5. Save turns to session

    All errors are returned as structured SSE error events.
    """
    settings = get_settings()
    session_store: SessionStore = request.app.state.session_store

    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events for the pipeline."""
        try:
            # ── Step 1: Safety guard ──────────────────────────────
            safety_result = safety_check(request_body.query)

            if not safety_result.passed:
                await logger.awarning(
                    "query_blocked",
                    category=safety_result.category,
                    query=request_body.query[:100],
                )
                yield _sse_event("error", {
                    "error": "query_blocked",
                    "category": safety_result.category,
                    "message": safety_result.refusal_message,
                })
                return

            # ── Step 2: Session history ───────────────────────────
            session_history = await session_store.get_recent_turns(
                request_body.session_id,
                limit=settings.max_session_history_turns,
            )

            # ── Step 3: Intent classification ─────────────────────
            classifier_output: ClassifierOutput = await asyncio.wait_for(
                classify(
                    query=request_body.query,
                    session_history=session_history,
                    user_profile=request_body.user_profile,
                ),
                timeout=settings.pipeline_timeout_seconds,
            )

            # ── Step 4: Route to agent ────────────────────────────
            agent = get_agent(classifier_output.target_agent)

            # ── Step 5a: Stream metadata ──────────────────────────
            metadata = PipelineMetadata(
                session_id=request_body.session_id,
                classified_intent=classifier_output.intent,
                target_agent=classifier_output.target_agent.value,
                entities=classifier_output.entities.model_dump(exclude_none=True),
                safety_verdict=classifier_output.safety_verdict,
            )
            yield _sse_event("metadata", metadata.model_dump())

            # ── Step 5b: Stream agent response ────────────────────
            agent_response_chunks: list[str] = []

            async for chunk in agent.run(
                query=request_body.query,
                entities=classifier_output.entities,
                user_profile=request_body.user_profile,
                session_history=session_history,
            ):
                agent_response_chunks.append(chunk)
                yield _sse_event("chunk", chunk)

            # ── Step 5c: Signal completion ────────────────────────
            yield _sse_event("done", "")

            # ── Step 6: Save turns (fire and forget) ──────────────
            full_response = "".join(agent_response_chunks)
            asyncio.create_task(
                _save_turns(
                    session_store,
                    request_body.session_id,
                    request_body.query,
                    full_response,
                )
            )

        except asyncio.TimeoutError:
            await logger.aerror(
                "pipeline_timeout",
                timeout_seconds=settings.pipeline_timeout_seconds,
                query=request_body.query[:100],
            )
            yield _sse_event("error", {
                "error": "timeout",
                "message": (
                    f"The request timed out after {settings.pipeline_timeout_seconds} seconds. "
                    "Please try again."
                ),
            })

        except Exception as exc:
            await logger.aerror(
                "pipeline_error",
                error_type=type(exc).__name__,
                error_msg=str(exc),
            )
            yield _sse_event("error", {
                "error": "internal_error",
                "message": "An unexpected error occurred. Please try again.",
            })

    return EventSourceResponse(event_generator())


async def _save_turns(
    store: SessionStore,
    session_id: str,
    user_query: str,
    assistant_response: str,
) -> None:
    """Save user and assistant turns — fire-and-forget task.

    Errors are logged but never propagated — saving turns must not
    block or crash the response stream.
    """
    try:
        await store.save_turn(session_id, "user", user_query)
        await store.save_turn(session_id, "assistant", assistant_response)
    except Exception as exc:
        await logger.aerror(
            "save_turns_failed",
            session_id=session_id,
            error_type=type(exc).__name__,
            error_msg=str(exc),
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "valura-ai"}


@app.get("/")
async def root() -> dict:
    """Root endpoint — service info."""
    return {
        "service": "Valura AI",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /api/v1/query",
    }
