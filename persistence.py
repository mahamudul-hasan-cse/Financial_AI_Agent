"""SQLite persistence layer for sessions, watchlists, and alerts."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from config import settings

_db_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_conn = _connect()


def init_db() -> None:
    """Create required tables if they do not already exist."""

    with _db_lock, _conn:
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New research',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_active REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                nlp_metadata_json TEXT,
                structured_response_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (session_id, symbol),
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                threshold REAL,
                label TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_triggered_at REAL,
                last_price REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_time
            ON messages (session_id, timestamp);

            CREATE INDEX IF NOT EXISTS idx_alerts_session
            ON alerts (session_id);
            """
        )


init_db()


def reset_for_tests() -> None:
    """Delete persisted rows while keeping the schema intact."""

    with _db_lock, _conn:
        _conn.execute("DELETE FROM alerts")
        _conn.execute("DELETE FROM watchlist")
        _conn.execute("DELETE FROM messages")
        _conn.execute("DELETE FROM sessions")


def _session_title_from_messages(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            text = msg["content"].strip()
            return text[:60] + ("..." if len(text) > 60 else "")
    return "New research"


def upsert_session(session_id: str, *, title: str | None = None, touch: bool = True) -> None:
    """Ensure a session exists and optionally refresh its timestamps."""

    now = time.time()
    with _db_lock, _conn:
        row = _conn.execute(
            "SELECT session_id, title FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            _conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, last_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, title or "New research", now, now, now),
            )
        elif touch or title is not None:
            _conn.execute(
                """
                UPDATE sessions
                SET title = COALESCE(?, title),
                    updated_at = CASE WHEN ? THEN ? ELSE updated_at END,
                    last_active = CASE WHEN ? THEN ? ELSE last_active END
                WHERE session_id = ?
                """,
                (title, int(touch), now, int(touch), now, session_id),
            )


def list_sessions() -> list[dict[str, Any]]:
    """Return saved sessions ordered by activity."""

    with _db_lock:
        rows = _conn.execute(
            """
            SELECT s.session_id, s.title, s.updated_at,
                   COALESCE(m.preview, '') AS preview,
                   COALESCE(m.message_count, 0) AS message_count
            FROM sessions s
            LEFT JOIN (
                SELECT session_id,
                       MAX(timestamp) AS max_ts,
                       SUBSTR(MAX(CASE WHEN role = 'user' THEN content ELSE '' END), 1, 140) AS preview,
                       COUNT(*) AS message_count
                FROM messages
                GROUP BY session_id
            ) m ON m.session_id = s.session_id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return one saved session with all messages."""

    with _db_lock:
        session_row = _conn.execute(
            "SELECT session_id, title, updated_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            return None
        message_rows = _conn.execute(
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
                "nlp_metadata": json.loads(row["nlp_metadata_json"]) if row["nlp_metadata_json"] else None,
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
    with _db_lock, _conn:
        _conn.execute(
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
            user_rows = _conn.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
            current_messages = [
                {"role": row["role"], "content": row["content"]}
                for row in user_rows
            ]
            title = _session_title_from_messages(current_messages)
            _conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ?, last_active = ? WHERE session_id = ?",
                (title, ts, ts, session_id),
            )
        else:
            _conn.execute(
                "UPDATE sessions SET updated_at = ?, last_active = ? WHERE session_id = ?",
                (ts, ts, session_id),
            )


def rename_session(session_id: str, title: str) -> bool:
    """Rename an existing session."""

    with _db_lock, _conn:
        cursor = _conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ?, last_active = ? WHERE session_id = ?",
            (title, time.time(), time.time(), session_id),
        )
    return cursor.rowcount > 0


def delete_session(session_id: str) -> None:
    """Delete a persisted session and all related entities."""

    with _db_lock, _conn:
        _conn.execute("DELETE FROM alerts WHERE session_id = ?", (session_id,))
        _conn.execute("DELETE FROM watchlist WHERE session_id = ?", (session_id,))
        _conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        _conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def list_watchlist(session_id: str) -> list[str]:
    """Return watchlist symbols for a session."""

    upsert_session(session_id)
    with _db_lock:
        rows = _conn.execute(
            "SELECT symbol FROM watchlist WHERE session_id = ? ORDER BY symbol ASC",
            (session_id,),
        ).fetchall()
    return [row["symbol"] for row in rows]


def add_watchlist_symbol(session_id: str, symbol: str) -> None:
    """Insert one watchlist symbol."""

    upsert_session(session_id)
    with _db_lock, _conn:
        _conn.execute(
            """
            INSERT OR IGNORE INTO watchlist (session_id, symbol, created_at)
            VALUES (?, ?, ?)
            """,
            (session_id, symbol.upper(), time.time()),
        )


def remove_watchlist_symbol(session_id: str, symbol: str) -> None:
    """Delete one symbol from the watchlist."""

    with _db_lock, _conn:
        _conn.execute(
            "DELETE FROM watchlist WHERE session_id = ? AND symbol = ?",
            (session_id, symbol.upper()),
        )


def list_alerts(session_id: str) -> list[dict[str, Any]]:
    """Return all alerts for a session."""

    upsert_session(session_id)
    with _db_lock:
        rows = _conn.execute(
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
    with _db_lock, _conn:
        cursor = _conn.execute(
            """
            INSERT INTO alerts (session_id, symbol, alert_type, threshold, label, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (session_id, symbol.upper(), alert_type, threshold, label, now),
        )
        alert_id = cursor.lastrowid
    return get_alert(alert_id)


def get_alert(alert_id: int) -> dict[str, Any]:
    """Return one alert by id."""

    with _db_lock:
        row = _conn.execute(
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


def update_alert(alert_id: int, *, enabled: bool | None = None, label: str | None = None) -> dict[str, Any]:
    """Update alert fields and return the updated row."""

    with _db_lock, _conn:
        current = get_alert(alert_id)
        _conn.execute(
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

    with _db_lock, _conn:
        _conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))


def mark_alert_triggered(alert_id: int, *, last_price: float | None = None) -> None:
    """Update last trigger timestamps for an alert."""

    with _db_lock, _conn:
        _conn.execute(
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
    with _db_lock:
        expired_rows = _conn.execute(
            "SELECT session_id FROM sessions WHERE ? - last_active > ?",
            (now, session_ttl),
        ).fetchall()
        for row in expired_rows:
            removed.append(row["session_id"])
    for session_id in removed:
        delete_session(session_id)

    with _db_lock:
        total = _conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        if total <= max_sessions:
            return removed
        overflow = total - max_sessions
        oldest_rows = _conn.execute(
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
