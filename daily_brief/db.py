"""SQLite storage: connection handling and schema migrations.

Why SQLite: the whole dataset is a few thousand short documents on one machine.
SQLite is in the standard library, needs no server, gives ACID writes, and ships
FTS5 -- a BM25 full-text index -- which is exactly the retrieval primitive this
application needs. Anything heavier would be infrastructure for its own sake.

Connections are thread-local, so the API thread pool and the background
scheduler can each hold their own without sharing a handle across threads.
WAL mode lets the scheduler write while the UI reads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Ordered migrations. Append only -- never edit a released statement block.
MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE preferences (
            id           INTEGER PRIMARY KEY CHECK (id = 1),
            data         TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        );

        CREATE TABLE sources (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            url          TEXT NOT NULL,
            site         TEXT NOT NULL DEFAULT '',
            categories   TEXT NOT NULL DEFAULT '[]',
            lean         TEXT NOT NULL DEFAULT 'unknown',
            confidence   TEXT NOT NULL DEFAULT 'low',
            source_type  TEXT NOT NULL DEFAULT '',
            country      TEXT NOT NULL DEFAULT '',
            ownership    TEXT NOT NULL DEFAULT '',
            notes        TEXT NOT NULL DEFAULT '',
            enabled      INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE feed_state (
            source_id            TEXT PRIMARY KEY,
            etag                 TEXT,
            last_modified        TEXT,
            last_fetch_at        TEXT,
            last_success_at      TEXT,
            last_status          TEXT,
            last_error           TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE articles (
            id              INTEGER PRIMARY KEY,
            canonical_url   TEXT NOT NULL UNIQUE,
            url             TEXT NOT NULL,
            title           TEXT NOT NULL,
            source_id       TEXT NOT NULL,
            source_name     TEXT NOT NULL DEFAULT '',
            summary         TEXT NOT NULL DEFAULT '',
            content         TEXT NOT NULL DEFAULT '',
            author          TEXT NOT NULL DEFAULT '',
            published_at    TEXT,
            retrieved_at    TEXT NOT NULL,
            language        TEXT NOT NULL DEFAULT '',
            image_url       TEXT NOT NULL DEFAULT '',
            feed_categories TEXT NOT NULL DEFAULT '[]',
            topics          TEXT NOT NULL DEFAULT '[]',
            entities        TEXT NOT NULL DEFAULT '[]',
            bias            TEXT,
            content_hash    TEXT NOT NULL DEFAULT '',
            simhash         INTEGER NOT NULL DEFAULT 0,
            cluster_id      TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX idx_articles_published  ON articles(published_at DESC);
        CREATE INDEX idx_articles_hash       ON articles(content_hash);
        CREATE INDEX idx_articles_cluster    ON articles(cluster_id);
        CREATE INDEX idx_articles_source     ON articles(source_id);

        CREATE VIRTUAL TABLE articles_fts USING fts5(
            title, summary, content, entities, topics,
            content='articles', content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TRIGGER articles_ai AFTER INSERT ON articles BEGIN
            INSERT INTO articles_fts(rowid, title, summary, content, entities, topics)
            VALUES (new.id, new.title, new.summary, new.content, new.entities, new.topics);
        END;
        CREATE TRIGGER articles_ad AFTER DELETE ON articles BEGIN
            INSERT INTO articles_fts(articles_fts, rowid, title, summary, content, entities, topics)
            VALUES ('delete', old.id, old.title, old.summary, old.content, old.entities, old.topics);
        END;
        CREATE TRIGGER articles_au AFTER UPDATE ON articles BEGIN
            INSERT INTO articles_fts(articles_fts, rowid, title, summary, content, entities, topics)
            VALUES ('delete', old.id, old.title, old.summary, old.content, old.entities, old.topics);
            INSERT INTO articles_fts(rowid, title, summary, content, entities, topics)
            VALUES (new.id, new.title, new.summary, new.content, new.entities, new.topics);
        END;

        CREATE TABLE clusters (
            id          TEXT PRIMARY KEY,
            key_title   TEXT NOT NULL DEFAULT '',
            size        INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE briefings (
            id            INTEGER PRIMARY KEY,
            brief_date    TEXT NOT NULL,
            generated_at  TEXT NOT NULL,
            engine        TEXT NOT NULL DEFAULT '',
            interests     TEXT NOT NULL DEFAULT '[]',
            story_count   INTEGER NOT NULL DEFAULT 0,
            payload       TEXT NOT NULL
        );
        CREATE INDEX idx_briefings_generated ON briefings(generated_at DESC);

        CREATE TABLE chat_sessions (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE chat_messages (
            id          INTEGER PRIMARY KEY,
            session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            sources     TEXT NOT NULL DEFAULT '[]',
            meta        TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_messages_session ON chat_messages(session_id, id);
        """,
    ),
]

SCHEMA_VERSION = MIGRATIONS[-1][0]


class Database:
    """Owns the SQLite file and hands out thread-local connections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialised = False

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        self._local.conn = conn

        with self._lock:
            if not self._initialised:
                self._migrate(conn)
                self._initialised = True
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        for version, script in MIGRATIONS:
            if version <= current:
                continue
            log.info("applying database migration %s", version)
            # executescript() commits any open transaction before it runs, so the
            # BEGIN/COMMIT has to live inside the script itself to be atomic.
            conn.executescript(
                f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;"
            )

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ helpers
    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        return self.connect().execute(sql, params)

    def query(self, sql: str, params: Any = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Any = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def transaction(self) -> "_Transaction":
        return _Transaction(self.connect())


class _Transaction:
    """Context manager for an explicit write transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
        return False


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default if default is not None else None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default if default is not None else None
