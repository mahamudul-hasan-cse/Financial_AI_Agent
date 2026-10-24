"""One-off copy of the SQLite database into PostgreSQL (e.g. Neon).

Usage::

    DATABASE_URL=postgresql://...  python scripts/migrate_sqlite_to_postgres.py \\
        [--sqlite data/financial_agent.db] [--dry-run]

* Creates the schema on the target (same DDL as the app).
* Copies sessions, messages, watchlist and alerts preserving ids.
* Idempotent: rows that already exist (same primary key) are skipped, so the
  script can be re-run safely.
* Resets the identity sequences of ``messages`` and ``alerts`` past the
  highest copied id so new inserts don't collide.
* Prints per-table row counts (source, inserted, target).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (table, columns, conflict target). Parents before children (foreign keys).
TABLES: list[tuple[str, list[str], str]] = [
    (
        "sessions",
        ["session_id", "client_id", "title", "created_at", "updated_at", "last_active"],
        "(session_id)",
    ),
    (
        "messages",
        [
            "id",
            "session_id",
            "role",
            "content",
            "timestamp",
            "nlp_metadata_json",
            "structured_response_json",
        ],
        "(id)",
    ),
    ("watchlist", ["session_id", "symbol", "created_at"], "(session_id, symbol)"),
    (
        "alerts",
        [
            "id",
            "session_id",
            "symbol",
            "alert_type",
            "threshold",
            "label",
            "enabled",
            "created_at",
            "last_triggered_at",
            "last_price",
        ],
        "(id)",
    ),
]
IDENTITY_TABLES = ("messages", "alerts")


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    return f"{scheme}://***@{rest.split('@', 1)[1]}"


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def migrate(sqlite_path: Path, database_url: str, *, dry_run: bool = False) -> dict:
    import psycopg

    # Build the target schema with the app's own DDL.
    sys.path.insert(0, str(ROOT))
    from db_schema import _PG_SCHEMA  # noqa: E402  (no app/config side effects)

    src = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    counts: dict[str, dict[str, int]] = {}
    with psycopg.connect(database_url, prepare_threshold=None) as dst:
        with dst.cursor() as cur:
            for statement in _PG_SCHEMA:
                cur.execute(statement)
            cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS client_id TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_client ON sessions (client_id)")

            for table, columns, conflict in TABLES:
                available = _sqlite_columns(src, table)
                select_cols = [c if c in available else f"NULL AS {c}" for c in columns]
                rows = src.execute(f"SELECT {', '.join(select_cols)} FROM {table}").fetchall()
                inserted = 0
                if rows:
                    placeholders = ", ".join(["%s"] * len(columns))
                    sql = (
                        f"INSERT INTO {table} ({', '.join(columns)}) "
                        f"{'OVERRIDING SYSTEM VALUE ' if table in IDENTITY_TABLES else ''}"
                        f"VALUES ({placeholders}) ON CONFLICT {conflict} DO NOTHING"
                    )
                    for row in rows:
                        cur.execute(sql, row)
                        inserted += cur.rowcount
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = {
                    "source": len(rows),
                    "inserted": inserted,
                    "target": cur.fetchone()[0],
                }

            for table in IDENTITY_TABLES:
                cur.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)"
                )
        if dry_run:
            dst.rollback()
        else:
            dst.commit()
    src.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sqlite",
        default=os.getenv("DATABASE_PATH") or str(ROOT / "data" / "financial_agent.db"),
        help="source SQLite file (default: DATABASE_PATH or data/financial_agent.db)",
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--dry-run", action="store_true", help="copy, report, then roll back")
    args = parser.parse_args()

    if not args.database_url or not args.database_url.startswith(("postgres://", "postgresql://")):
        parser.error("set DATABASE_URL (or --database-url) to a postgres:// URL")
    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        parser.error(f"SQLite file not found: {sqlite_path}")

    print(f"Copying {sqlite_path} -> {_redact(args.database_url)}")
    counts = migrate(sqlite_path, args.database_url, dry_run=args.dry_run)
    for table, c in counts.items():
        print(
            f"  {table:<10} source={c['source']:<6} inserted={c['inserted']:<6} "
            f"target={c['target']}"
        )
    if args.dry_run:
        print("Dry run: rolled back.")
    mismatched = [t for t, c in counts.items() if c["target"] < c["source"]]
    return 1 if mismatched and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
