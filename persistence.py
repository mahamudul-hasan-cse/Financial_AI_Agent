"""Persistence layer for sessions, watchlists, and alerts.

Two backends share the same public functions:

* **SQLite** (default): the file configured via ``DATABASE_PATH``.
* **PostgreSQL** (e.g. Neon): used when ``DATABASE_URL`` is a ``postgres://``
  or ``postgresql://`` URL. Connections come from a small ``psycopg_pool``
  pool; ``sslmode`` and other options are taken from the URL, and server-side
  prepared statements are disabled so Neon's pooled (PgBouncer) URL works.

Queries are written once with ``?`` placeholders; the Postgres adapter
rewrites them to ``%s``. Every session row records the anonymous
``client_id`` that created it so the API can restrict listing/reading/
modifying to the owning browser.
"""

from __future__ import annotations

import atexit
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import APP_DIR, settings
from db_schema import _PG_SCHEMA, _SQLITE_SCHEMA

_db_lock = threading.Lock()

DB_PATH: Path = settings.sqlite_path
_DEFAULT_DB_PATH: Path = APP_DIR / "data" / "financial_agent.db"


def is_postgres_url(url: str | None) -> bool:
    """Return True when *url* points at PostgreSQL."""

    return bool(url) and url.split("://", 1)[0] in ("postgres", "postgresql")


class _PgConnection:
    """Tiny adapter giving a psycopg connection the sqlite3 ``execute`` API."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: tuple | list = ()) -> Any:
        return self._conn.execute(sql.replace("?", "%s"), params)


_pool: Any = None
_conn: sqlite3.Connection | None = None
DATABASE_URL: str | None = None


def _connect_sqlite() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _open_pool(url: str) -> Any:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        url,
        min_size=int(os.getenv("DB_POOL_MIN_SIZE", "1")),
        max_size=int(os.getenv("DB_POOL_MAX_SIZE", "5")),
        # prepare_threshold=None: no server-side prepared statements, which
        # transaction-mode poolers (Neon's -pooler host / PgBouncer) reject.
        kwargs={"row_factory": dict_row, "prepare_threshold": None},
        check=ConnectionPool.check_connection,
        open=True,
        name="financial-agent",
    )


def configure(database_url: str | None = None) -> None:
    """(Re)select the backend: Postgres for a postgres URL, else SQLite.

    Called once at import with ``settings.database_url``; tests call it to
    switch backends.
    """

    global _pool, _conn, DATABASE_URL
    if _pool is not None:
        _pool.close()
        _pool = None
    if _conn is not None:
        _conn.close()
        _conn = None
    if is_postgres_url(database_url):
        DATABASE_URL = database_url
        _pool = _open_pool(database_url)
    else:
        DATABASE_URL = None
        _conn = _connect_sqlite()
    init_db()


def _close_pool() -> None:
    if _pool is not None:
        _pool.close()


atexit.register(_close_pool)


def using_postgres() -> bool:
    return _pool is not None


@contextmanager
def _tx() -> Iterator[Any]:
    """Yield a connection-like object inside one committed transaction."""

    if _pool is not None:
        with _pool.connection() as conn:  # commits on success, rolls back on error
            yield _PgConnection(conn)
    else:
        with _db_lock, _conn:
            yield _conn


def init_db() -> None:
    """Create required tables if they do not already exist."""

    if _pool is not None:
        with _tx() as db:
            for statement in _PG_SCHEMA:
                db.execute(statement)
            # Migration: databases created before client ownership existed.
            columns = {
                row["column_name"]
                for row in db.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = 'sessions'"
                )
            }
            if "client_id" not in columns:
                db.execute("ALTER TABLE sessions ADD COLUMN client_id TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions (client_id)")
        return

    with _db_lock, _conn:
        _conn.executescript(_SQLITE_SCHEMA)
        # Migration: databases created before client ownership existed.
        columns = {row["name"] for row in _conn.execute("PRAGMA table_info(sessions)")}
        if "client_id" not in columns:
            _conn.execute("ALTER TABLE sessions ADD COLUMN client_id TEXT")
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions (client_id)")


configure(settings.database_url)


def reset_for_tests() -> None:
    """Delete persisted rows while keeping the schema intact.

    Refuses to run against the default application database so a test run
    can never wipe real data (tests set ``DATABASE_PATH`` to a temp file).
    """

    if _pool is not None:
        if DATABASE_URL != os.getenv("TEST_DATABASE_URL"):
            raise RuntimeError(
                "reset_for_tests() only wipes a Postgres database named by TEST_DATABASE_URL."
            )
    elif DB_PATH.resolve() == _DEFAULT_DB_PATH.resolve():
        raise RuntimeError(
            "reset_for_tests() refused to wipe the default database "
            f"({DB_PATH}); set DATABASE_PATH to a temporary file."
        )
    with _tx() as db:
        db.execute("DELETE FROM alerts")
        db.execute("DELETE FROM watchlist")
        db.execute("DELETE FROM messages")
        db.execute("DELETE FROM sessions")


def _session_title_from_messages(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            text = msg["content"].strip()
            return text[:60] + ("..." if len(text) > 60 else "")
    return "New research"


def upsert_session(
    session_id: str,
    *,
    client_id: str | None = None,
    title: str | None = None,
    touch: bool = True,
) -> None:
    """Ensure a session exists and optionally refresh its timestamps.

    ``client_id`` is only recorded when the session row is created.
    """

    now = time.time()
    with _tx() as db:
        row = db.execute(
            "SELECT session_id, title FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            db.execute(
                """
                INSERT INTO sessions (session_id, client_id, title, created_at, updated_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, client_id, title or "New research", now, now, now),
            )
        elif touch or title is not None:
            db.execute(
                """
                UPDATE sessions
                SET title = COALESCE(?, title),
                    updated_at = CASE WHEN ? = 1 THEN ? ELSE updated_at END,
                    last_active = CASE WHEN ? = 1 THEN ? ELSE last_active END
                WHERE session_id = ?
                """,
                (title, int(touch), now, int(touch), now, session_id),
            )


_SESSION_SUMMARY_SQL = """
    SELECT s.session_id, s.title, s.updated_at,
           COALESCE((
               SELECT SUBSTR(m.content, 1, 140)
               FROM messages m
               WHERE m.session_id = s.session_id AND m.role = 'user'
               ORDER BY m.timestamp DESC, m.id DESC
               LIMIT 1
           ), '') AS preview,
           (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS message_count
    FROM sessions s
"""


def list_sessions(client_id: str) -> list[dict[str, Any]]:
    """Return the sessions owned by *client_id*, most recently updated first.

    The preview is the latest user message in each session.
    """

    with _tx() as db:
        rows = db.execute(
            _SESSION_SUMMARY_SQL + " WHERE s.client_id = ? ORDER BY s.updated_at DESC",
            (client_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    """Return the sidebar summary row for one session."""

    with _tx() as db:
        row = db.execute(
            _SESSION_SUMMARY_SQL + " WHERE s.session_id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def get_session_owner(session_id: str) -> tuple[bool, str | None]:
    """Return ``(exists, client_id)`` for a session."""

    with _tx() as db:
        row = db.execute(
            "SELECT client_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return False, None
    return True, row["client_id"]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return one saved session with all messages."""

    with _tx() as db:
        session_row = db.execute(
            "SELECT session_id, client_id, title, updated_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            return None
        message_rows = db.execute(
            """
            SELECT role, content, timestamp, nlp_metadata_json, structured_response_json
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (session_id,),
        ).fetchall()
    messages = []
    for row in message_rows:
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "nlp_metadata": (
                    json.loads(row["nlp_metadata_json"]) if row["nlp_metadata_json"] else None
                ),
                "structured_response": (
                    json.loads(row["structured_response_json"])
                    if row["structured_response_json"]
                    else None
                ),
            }
        )
    payload = dict(session_row)
    payload["messages"] = messages
    return payload


def append_message(
    session_id: str,
    *,
    role: str,
    content: str,
    timestamp: float | None = None,
    nlp_metadata: dict[str, Any] | None = None,
    structured_response: dict[str, Any] | None = None,
) -> None:
    """Persist one session message."""

    ts = timestamp or time.time()
    upsert_session(session_id)
    with _tx() as db:
        db.execute(
            """
            INSERT INTO messages (session_id, role, content, timestamp, nlp_metadata_json, structured_response_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                ts,
                json.dumps(nlp_metadata) if nlp_metadata else None,
                json.dumps(structured_response) if structured_response else None,
            ),
        )
        if role == "user":
            user_rows = db.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
            current_messages = [
                {"role": row["role"], "content": row["content"]} for row in user_rows
            ]
            title = _session_title_from_messages(current_messages)
            db.execute(
                "UPDATE sessions SET title = ?, updated_at = ?, last_active = ? WHERE session_id = ?",
                (title, ts, ts, session_id),
            )
        else:
            db.execute(
                "UPDATE sessions SET updated_at = ?, last_active = ? WHERE session_id = ?",
                (ts, ts, session_id),
            )


def rename_session(session_id: str, title: str) -> bool:
    """Rename an existing session."""

    with _tx() as db:
        cursor = db.execute(
            "UPDATE sessions SET title = ?, updated_at = ?, last_active = ? WHERE session_id = ?",
            (title, time.time(), time.time(), session_id),
        )
    return cursor.rowcount > 0


def delete_session(session_id: str) -> None:
    """Delete a persisted session and all related entities."""

    with _tx() as db:
        db.execute("DELETE FROM alerts WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM watchlist WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def list_watchlist(session_id: str) -> list[str]:
    """Return watchlist symbols for a session."""

    with _tx() as db:
        rows = db.execute(
            "SELECT symbol FROM watchlist WHERE session_id = ? ORDER BY symbol ASC",
            (session_id,),
        ).fetchall()
    return [row["symbol"] for row in rows]


def add_watchlist_symbol(session_id: str, symbol: str) -> None:
    """Insert one watchlist symbol."""

    upsert_session(session_id)
    with _tx() as db:
        db.execute(
            """
            INSERT INTO watchlist (session_id, symbol, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (session_id, symbol.upper(), time.time()),
        )


def remove_watchlist_symbol(session_id: str, symbol: str) -> None:
    """Delete one symbol from the watchlist."""

    with _tx() as db:
        db.execute(
            "DELETE FROM watchlist WHERE session_id = ? AND symbol = ?",
            (session_id, symbol.upper()),
        )


def list_alerts(session_id: str) -> list[dict[str, Any]]:
    """Return all alerts for a session."""

    with _tx() as db:
        rows = db.execute(
            """
            SELECT id, session_id, symbol, alert_type, threshold, label,
                   enabled, last_triggered_at, last_price
            FROM alerts
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (session_id,),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
        items.append(item)
    return items


def create_alert(
    session_id: str,
    symbol: str,
    alert_type: str,
    *,
    threshold: float | None,
    label: str,
) -> dict[str, Any]:
    """Persist a new alert and return it."""

    upsert_session(session_id)
    now = time.time()
    with _tx() as db:
        cursor = db.execute(
            """
            INSERT INTO alerts (session_id, symbol, alert_type, threshold, label, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            RETURNING id
            """,
            (session_id, symbol.upper(), alert_type, threshold, label, now),
        )
        alert_id = cursor.fetchone()["id"]
    return get_alert(alert_id)


def get_alert(alert_id: int) -> dict[str, Any]:
    """Return one alert by id."""

    with _tx() as db:
        row = db.execute(
            """
            SELECT id, session_id, symbol, alert_type, threshold, label,
                   enabled, last_triggered_at, last_price
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,),
        ).fetchone()
    if row is None:
        raise KeyError(alert_id)
    item = dict(row)
    item["enabled"] = bool(item["enabled"])
    return item


def update_alert(
    alert_id: int, *, enabled: bool | None = None, label: str | None = None
) -> dict[str, Any]:
    """Update alert fields and return the updated row."""

    current = get_alert(alert_id)  # acquires the (non-reentrant) lock itself
    with _tx() as db:
        db.execute(
            """
            UPDATE alerts
            SET enabled = ?, label = ?
            WHERE id = ?
            """,
            (
                int(enabled if enabled is not None else current["enabled"]),
                label if label is not None else current["label"],
                alert_id,
            ),
        )
    return get_alert(alert_id)


def delete_alert(alert_id: int) -> None:
    """Delete one alert."""

    with _tx() as db:
        db.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))


def mark_alert_triggered(alert_id: int, *, last_price: float | None = None) -> None:
    """Update last trigger timestamps for an alert."""

    with _tx() as db:
        db.execute(
            """
            UPDATE alerts
            SET last_triggered_at = ?, last_price = COALESCE(?, last_price)
            WHERE id = ?
            """,
            (time.time(), last_price, alert_id),
        )


def prune_old_sessions(max_sessions: int, session_ttl: int) -> list[str]:
    """Delete expired or oldest sessions and return the removed session ids."""

    now = time.time()
    removed: list[str] = []
    with _tx() as db:
        expired_rows = db.execute(
            "SELECT session_id FROM sessions WHERE ? - last_active > ?",
            (now, session_ttl),
        ).fetchall()
        for row in expired_rows:
            removed.append(row["session_id"])
    for session_id in removed:
        delete_session(session_id)

    with _tx() as db:
        total = db.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        if total <= max_sessions:
            return removed
        overflow = total - max_sessions
        oldest_rows = db.execute(
            """
            SELECT session_id FROM sessions
            ORDER BY last_active ASC
            LIMIT ?
            """,
            (overflow,),
        ).fetchall()
    for row in oldest_rows:
        removed.append(row["session_id"])
        delete_session(row["session_id"])
    return removed
