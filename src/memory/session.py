"""Session memory — SQLite-backed turn history for conversation context.

Uses ``aiosqlite`` directly (no ORM) for full transparency and minimal
overhead.  The schema is a single ``session_turns`` table indexed by
``session_id``.

Design decisions
~~~~~~~~~~~~~~~~
* **SQLite + aiosqlite** — zero-config persistence that survives
  restarts.  Justified in README: this demo doesn't need multi-process
  write concurrency, and SQLite is the simplest option that still gives
  real persistence.
* **No ORM** — the schema is one table; an ORM would be overhead for
  no benefit.  Raw SQL keeps the storage layer fully visible.
* **OpenAI-compatible output** — ``get_recent_turns()`` returns dicts
  in ``{"role": ..., "content": ...}`` format, ready to drop into the
  messages array.
"""

from __future__ import annotations

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL statements — defined once, reused by every method call.
# ---------------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS session_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_session_id ON session_turns(session_id);
"""

_SELECT_RECENT = """\
SELECT role, content
FROM   session_turns
WHERE  session_id = ?
ORDER  BY turn_index DESC
LIMIT  ?;
"""

_SELECT_MAX_INDEX = """\
SELECT COALESCE(MAX(turn_index), -1)
FROM   session_turns
WHERE  session_id = ?;
"""

_INSERT_TURN = """\
INSERT INTO session_turns (session_id, turn_index, role, content)
VALUES (?, ?, ?, ?);
"""


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """Async SQLite-backed conversation turn storage.

    Usage::

        store = SessionStore("valura.db")
        await store.initialize()           # creates table if needed
        await store.save_turn(sid, "user", "hello")
        history = await store.get_recent_turns(sid, limit=5)

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Will be created if it does
        not exist.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Create the ``session_turns`` table and index if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE)
            await db.execute(_CREATE_INDEX)
            await db.commit()
        await logger.ainfo("session_store_initialized", db_path=self._db_path)

    async def get_recent_turns(
        self,
        session_id: str,
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """Retrieve the most recent turns for a session.

        Parameters
        ----------
        session_id:
            The conversation session identifier.
        limit:
            Maximum number of turns to return (most recent first in
            storage, but returned in chronological order).

        Returns
        -------
        list[dict[str, str]]
            Turns in chronological order, each as
            ``{"role": "user"|"assistant", "content": "..."}``.
            Format is directly compatible with the OpenAI messages array.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(_SELECT_RECENT, (session_id, limit))
            rows = await cursor.fetchall()

        # Rows come back newest-first (ORDER BY turn_index DESC);
        # reverse to chronological order.
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

    async def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Persist a single conversation turn.

        Automatically determines the next ``turn_index`` for the
        session by reading the current maximum.

        Parameters
        ----------
        session_id:
            The conversation session identifier.
        role:
            ``"user"`` or ``"assistant"``.
        content:
            The turn's text content.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(_SELECT_MAX_INDEX, (session_id,))
            row = await cursor.fetchone()
            next_index = (row[0] if row else -1) + 1

            await db.execute(
                _INSERT_TURN,
                (session_id, next_index, role, content),
            )
            await db.commit()
