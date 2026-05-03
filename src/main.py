"""FastAPI application — SSE pipeline endpoint.

Single endpoint ``POST /api/v1/query`` that runs the full pipeline:

1. Safety guard → block if harmful
2. Session history retrieval (degrades gracefully)
3. Intent classification (LLM call with timeout + fallback)
4. Agent routing
5. Streamed response via Server-Sent Events

**Architectural decision**: SSE is the ONLY response mode.  There is
no JSON fallback path.  Errors, safety blocks, and pipeline state are
all communicated through SSE events — never via HTTP 4xx/5xx status
codes for pipeline failures.  The reason: once a streaming response
starts, the HTTP status code cannot change.  Only malformed requests
(caught by FastAPI validation) return HTTP 422.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from src.classifier.classifier import classify
from src.classifier.schema import ClassifierOutput, FALLBACK_CLASSIFIER_OUTPUT
from src.config import get_settings
from src.memory.session import SessionStore
from src.models.api import QueryRequest
from src.router.router import get_agent
from src.safety.guard import check as safety_check
from src.utils.sse import (
    format_safety_block,
    format_sse_done,
    format_sse_error,
    format_sse_metadata,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

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
        "valura_ai_startup",
        environment=settings.environment,
        classifier_model=settings.classifier_model,
        pipeline_timeout=settings.pipeline_timeout_seconds,
        db_path=db_path,
    )

    yield  # Application runs here.

    await logger.ainfo("valura_ai_shutdown")


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
# Pipeline endpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/query")
async def query_pipeline(
    request_body: QueryRequest,
    req: Request,
) -> StreamingResponse:
    """Run the full AI pipeline and stream the response via SSE.

    Pipeline order:
    1. Safety guard check
    2. Session history retrieval (degrades gracefully)
    3. Intent classification (timeout + fallback)
    4. Stream pipeline metadata (first SSE event)
    5. Route to agent + stream response
    6. Save session turn (fire and forget)
    7. Done event

    All pipeline errors are returned as structured SSE events.
    Only malformed requests return HTTP 4xx (via FastAPI validation).
    """
    settings = get_settings()
    session_store: SessionStore = req.app.state.session_store

    async def generate() -> AsyncGenerator[str, None]:
        """Inner async generator — produces raw SSE strings."""

        # ── Step 1: Safety guard ──────────────────────────────────
        try:
            safety_result = safety_check(request_body.query)
        except Exception as exc:
            logger.error("safety_guard_error", error=str(exc))
            yield format_sse_error("Safety check failed", "SAFETY_ERROR")
            yield format_sse_done()
            return

        if not safety_result.passed:
            logger.info(
                "query_blocked",
                category=safety_result.category,
                query=request_body.query[:100],
            )
            yield format_safety_block(
                safety_result.category or "unknown",
                safety_result.refusal_message or "Request blocked by safety guard.",
            )
            yield format_sse_done()
            return

        # ── Step 2: Load session history (degrade gracefully) ─────
        try:
            history = await session_store.get_recent_turns(
                request_body.session_id,
                limit=settings.max_session_history_turns,
            )
        except Exception as exc:
            logger.error(
                "session_load_error",
                error=str(exc),
                session_id=request_body.session_id,
            )
            history = []  # History is not critical — continue without it

        # ── Step 3: Intent classification ─────────────────────────
        try:
            classifier_output: ClassifierOutput = await asyncio.wait_for(
                classify(
                    query=request_body.query,
                    session_history=history,
                    user_profile=request_body.user_profile,
                ),
                timeout=10.0,  # Classifier gets 10s of the total budget
            )
        except asyncio.TimeoutError:
            logger.warning(
                "classifier_timeout",
                session_id=request_body.session_id,
            )
            classifier_output = FALLBACK_CLASSIFIER_OUTPUT
        except Exception as exc:
            logger.error(
                "classifier_error",
                error=str(exc),
                session_id=request_body.session_id,
            )
            classifier_output = FALLBACK_CLASSIFIER_OUTPUT

        # ── Step 4: Stream pipeline metadata (FIRST SSE event) ────
        metadata = {
            "session_id": request_body.session_id,
            "classified_intent": classifier_output.intent,
            "target_agent": classifier_output.target_agent.value,
            "entities": classifier_output.entities.model_dump(exclude_none=True),
            "safety_verdict": classifier_output.safety_verdict,
            "confidence": classifier_output.confidence,
        }
        yield format_sse_metadata(metadata)

        # ── Step 5: Route to agent and stream response ────────────
        try:
            agent = get_agent(classifier_output.target_agent)
        except Exception as exc:
            logger.error("routing_error", error=str(exc))
            yield format_sse_error("Routing failed", "ROUTING_ERROR")
            yield format_sse_done()
            return

        try:
            async for chunk in agent.run(
                query=request_body.query,
                entities=classifier_output.entities,
                user_profile=request_body.user_profile,
                session_history=history,
                classified_intent=classifier_output.intent,
                target_agent=classifier_output.target_agent.value,
            ):
                yield chunk
        except asyncio.CancelledError:
            logger.info(
                "client_disconnected",
                session_id=request_body.session_id,
            )
            return
        except Exception as exc:
            logger.error(
                "agent_error",
                error=str(exc),
                agent=classifier_output.target_agent.value,
            )
            yield format_sse_error(
                f"Agent execution failed: {type(exc).__name__}",
                "AGENT_ERROR",
            )

        # ── Step 6: Save session turn (fire and forget) ───────────
        asyncio.create_task(
            _save_session_turn(
                session_store,
                request_body.session_id,
                request_body.query,
                classifier_output,
            )
        )

        # ── Step 7: Done ──────────────────────────────────────────
        yield format_sse_done()

    # Wrap in pipeline-level timeout.
    async def timeout_wrapper() -> AsyncGenerator[str, None]:
        """Enforce total pipeline timeout around generate()."""
        try:
            async for event in generate():
                yield event
        except asyncio.TimeoutError:
            logger.error(
                "pipeline_timeout",
                timeout_seconds=settings.pipeline_timeout_seconds,
                session_id=request_body.session_id,
            )
            yield format_sse_error(
                f"Pipeline timed out after {settings.pipeline_timeout_seconds}s",
                "PIPELINE_TIMEOUT",
            )
            yield format_sse_done()

    return StreamingResponse(
        timeout_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Session save helper — fire and forget
# ---------------------------------------------------------------------------

async def _save_session_turn(
    store: SessionStore,
    session_id: str,
    query: str,
    classifier_output: ClassifierOutput,
) -> None:
    """Save user and assistant turns — fire-and-forget task.

    Errors are logged but never propagated — saving turns must not
    block or crash the response stream.
    """
    try:
        await store.save_turn(session_id, "user", query)
        summary = (
            f"[{classifier_output.target_agent.value}] "
            f"Intent: {classifier_output.intent}"
        )
        await store.save_turn(session_id, "assistant", summary)
    except Exception as exc:
        logger.error(
            "session_save_error",
            error=str(exc),
            session_id=session_id,
        )
        # Never raise — session save failure must not affect the response


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint — used by infrastructure for liveness."""
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}


@app.get("/")
async def root() -> dict:
    """Root endpoint — service info and navigation."""
    return {
        "service": "Valura AI",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /api/v1/query",
    }
