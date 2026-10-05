"""Data access. All SQL lives here; the rest of the app talks in domain objects."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from .db import Database, dumps, loads
from .models import (
    Article,
    BiasAssessment,
    Confidence,
    Lean,
    Preferences,
    Source,
    utcnow,
)

log = logging.getLogger(__name__)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class SourceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(self, sources: Iterable[Source]) -> int:
        rows = [
            (
                s.id,
                s.name,
                s.url,
                s.site,
                dumps(s.categories),
                str(s.lean),
                str(s.confidence),
                s.source_type,
                s.country,
                s.ownership,
                s.notes,
                int(s.enabled),
            )
            for s in sources
        ]
        if not rows:
            return 0
        with self.db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO sources (id, name, url, site, categories, lean, confidence,
                                     source_type, country, ownership, notes, enabled)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, url=excluded.url, site=excluded.site,
                    categories=excluded.categories, lean=excluded.lean,
                    confidence=excluded.confidence, source_type=excluded.source_type,
                    country=excluded.country, ownership=excluded.ownership,
                    notes=excluded.notes, enabled=excluded.enabled
                """,
                rows,
            )
        return len(rows)

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            name=row["name"],
            url=row["url"],
            site=row["site"],
            categories=loads(row["categories"], []) or [],
            lean=Lean.parse(row["lean"]),
            confidence=Confidence.parse(row["confidence"]),
            source_type=row["source_type"],
            country=row["country"],
            ownership=row["ownership"],
            notes=row["notes"],
            enabled=bool(row["enabled"]),
        )

    def all(self, enabled_only: bool = False) -> list[Source]:
        sql = "SELECT * FROM sources"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY name COLLATE NOCASE"
        return [self._row_to_source(r) for r in self.db.query(sql)]

    def get(self, source_id: str) -> Source | None:
        row = self.db.query_one("SELECT * FROM sources WHERE id = ?", (source_id,))
        return self._row_to_source(row) if row else None

    def by_id_map(self) -> dict[str, Source]:
        return {s.id: s for s in self.all()}

    # ------------------------------------------------------------- feed state
    def get_feed_state(self, source_id: str) -> dict[str, Any]:
        row = self.db.query_one(
            "SELECT * FROM feed_state WHERE source_id = ?", (source_id,)
        )
        return dict(row) if row else {}

    def all_feed_state(self) -> dict[str, dict[str, Any]]:
        return {r["source_id"]: dict(r) for r in self.db.query("SELECT * FROM feed_state")}

    def record_fetch(
        self,
        source_id: str,
        *,
        status: str,
        etag: str | None = None,
        last_modified: str | None = None,
        error: str | None = None,
    ) -> None:
        now = _iso(utcnow())
        success = error is None
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO feed_state (source_id, etag, last_modified, last_fetch_at,
                                        last_success_at, last_status, last_error,
                                        consecutive_failures)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id) DO UPDATE SET
                    etag = COALESCE(excluded.etag, feed_state.etag),
                    last_modified = COALESCE(excluded.last_modified, feed_state.last_modified),
                    last_fetch_at = excluded.last_fetch_at,
                    last_success_at = CASE WHEN excluded.last_error IS NULL
                                           THEN excluded.last_fetch_at
                                           ELSE feed_state.last_success_at END,
                    last_status = excluded.last_status,
                    last_error = excluded.last_error,
                    consecutive_failures = CASE WHEN excluded.last_error IS NULL
                                                THEN 0
                                                ELSE feed_state.consecutive_failures + 1 END
                """,
                (
                    source_id,
                    etag,
                    last_modified,
                    now,
                    now if success else None,
                    status,
                    error,
                    0 if success else 1,
                ),
            )


class ArticleRepository:
    _COLUMNS = (
        "id, canonical_url, url, title, source_id, source_name, summary, content, "
        "author, published_at, retrieved_at, language, image_url, feed_categories, "
        "topics, entities, bias, content_hash, simhash, cluster_id"
    )

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------ (de)serialise
    @staticmethod
    def row_to_article(row: sqlite3.Row) -> Article:
        return Article(
            id=row["id"],
            canonical_url=row["canonical_url"],
            url=row["url"],
            title=row["title"],
            source_id=row["source_id"],
            source_name=row["source_name"],
            summary=row["summary"],
            content=row["content"],
            author=row["author"],
            published_at=_parse_dt(row["published_at"]),
            retrieved_at=_parse_dt(row["retrieved_at"]) or utcnow(),
            language=row["language"],
            image_url=row["image_url"],
            feed_categories=loads(row["feed_categories"], []) or [],
            topics=loads(row["topics"], []) or [],
            entities=loads(row["entities"], []) or [],
            bias=BiasAssessment.from_dict(loads(row["bias"], {})),
            content_hash=row["content_hash"],
            simhash=row["simhash"],
            cluster_id=row["cluster_id"],
        )

    @staticmethod
    def _article_params(article: Article) -> tuple:
        return (
            article.canonical_url or article.url,
            article.url,
            article.title,
            article.source_id,
            article.source_name,
            article.summary,
            article.content,
            article.author,
            _iso(article.published_at),
            _iso(article.retrieved_at),
            article.language,
            article.image_url,
            dumps(article.feed_categories),
            dumps(article.topics),
            dumps(article.entities),
            dumps(article.bias.to_dict()) if article.bias else None,
            article.content_hash,
            article.simhash,
            article.cluster_id,
        )

    # -------------------------------------------------------------------- write
    def upsert_many(self, articles: Sequence[Article]) -> tuple[int, int]:
        """Insert new articles, refresh existing ones. Returns (inserted, updated)."""
        if not articles:
            return (0, 0)
        inserted = updated = 0
        with self.db.transaction() as conn:
            for article in articles:
                params = self._article_params(article)
                existing = conn.execute(
                    "SELECT id FROM articles WHERE canonical_url = ?", (params[0],)
                ).fetchone()
                conn.execute(
                    """
                    INSERT INTO articles (canonical_url, url, title, source_id, source_name,
                        summary, content, author, published_at, retrieved_at, language,
                        image_url, feed_categories, topics, entities, bias, content_hash,
                        simhash, cluster_id)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(canonical_url) DO UPDATE SET
                        title=excluded.title,
                        summary=CASE WHEN length(excluded.summary) > length(articles.summary)
                                     THEN excluded.summary ELSE articles.summary END,
                        content=CASE WHEN length(excluded.content) > length(articles.content)
                                     THEN excluded.content ELSE articles.content END,
                        published_at=COALESCE(articles.published_at, excluded.published_at),
                        topics=excluded.topics,
                        entities=excluded.entities,
                        bias=excluded.bias,
                        simhash=excluded.simhash,
                        cluster_id=CASE WHEN excluded.cluster_id != ''
                                        THEN excluded.cluster_id ELSE articles.cluster_id END
                    """,
                    params,
                )
                if existing is None:
                    inserted += 1
                    row = conn.execute(
                        "SELECT id FROM articles WHERE canonical_url = ?", (params[0],)
                    ).fetchone()
                    article.id = row["id"] if row else None
                else:
                    updated += 1
                    article.id = existing["id"]
        return (inserted, updated)

    def set_cluster(self, assignments: dict[int, str]) -> None:
        if not assignments:
            return
        with self.db.transaction() as conn:
            conn.executemany(
                "UPDATE articles SET cluster_id = ? WHERE id = ?",
                [(cid, aid) for aid, cid in assignments.items()],
            )

    def update_analysis(self, article: Article) -> None:
        if article.id is None:
            return
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE articles SET topics=?, entities=?, bias=?, summary=? WHERE id=?",
                (
                    dumps(article.topics),
                    dumps(article.entities),
                    dumps(article.bias.to_dict()) if article.bias else None,
                    article.summary,
                    article.id,
                ),
            )

    def prune(self, retention_days: int) -> int:
        cutoff = _iso(utcnow() - timedelta(days=retention_days))
        with self.db.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM articles WHERE COALESCE(published_at, retrieved_at) < ?",
                (cutoff,),
            )
            deleted = cur.rowcount or 0
            conn.execute(
                "DELETE FROM clusters WHERE id NOT IN "
                "(SELECT DISTINCT cluster_id FROM articles WHERE cluster_id != '')"
            )
        return deleted

    # --------------------------------------------------------------------- read
    def get(self, article_id: int) -> Article | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM articles WHERE id = ?", (article_id,)
        )
        return self.row_to_article(row) if row else None

    def get_many(self, ids: Sequence[int]) -> list[Article]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.db.query(
            f"SELECT {self._COLUMNS} FROM articles WHERE id IN ({marks})", tuple(ids)
        )
        by_id = {r["id"]: self.row_to_article(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def get_by_url(self, url: str) -> Article | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM articles WHERE canonical_url = ?", (url,)
        )
        return self.row_to_article(row) if row else None

    def recent(
        self,
        *,
        hours: int | None = None,
        limit: int = 200,
        source_ids: Sequence[str] | None = None,
    ) -> list[Article]:
        sql = f"SELECT {self._COLUMNS} FROM articles"
        params: list[Any] = []
        where: list[str] = []
        if hours is not None:
            where.append("COALESCE(published_at, retrieved_at) >= ?")
            params.append(_iso(utcnow() - timedelta(hours=hours)))
        if source_ids:
            where.append(f"source_id IN ({','.join('?' * len(source_ids))})")
            params.extend(source_ids)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY COALESCE(published_at, retrieved_at) DESC LIMIT ?"
        params.append(limit)
        return [self.row_to_article(r) for r in self.db.query(sql, tuple(params))]

    def find_by_hash(self, content_hash: str) -> Article | None:
        row = self.db.query_one(
            f"SELECT {self._COLUMNS} FROM articles WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        return self.row_to_article(row) if row else None

    def existing_hashes(self, hashes: Sequence[str]) -> set[str]:
        if not hashes:
            return set()
        found: set[str] = set()
        # Chunk to stay well under SQLITE_MAX_VARIABLE_NUMBER.
        for start in range(0, len(hashes), 400):
            chunk = hashes[start : start + 400]
            marks = ",".join("?" * len(chunk))
            rows = self.db.query(
                f"SELECT content_hash FROM articles WHERE content_hash IN ({marks})",
                tuple(chunk),
            )
            found.update(r["content_hash"] for r in rows)
        return found

    def existing_urls(self, urls: Sequence[str]) -> set[str]:
        if not urls:
            return set()
        found: set[str] = set()
        for start in range(0, len(urls), 400):
            chunk = urls[start : start + 400]
            marks = ",".join("?" * len(chunk))
            rows = self.db.query(
                f"SELECT canonical_url FROM articles WHERE canonical_url IN ({marks})",
                tuple(chunk),
            )
            found.update(r["canonical_url"] for r in rows)
        return found

    def cluster_members(self, cluster_id: str, exclude_id: int | None = None) -> list[Article]:
        if not cluster_id:
            return []
        sql = f"SELECT {self._COLUMNS} FROM articles WHERE cluster_id = ?"
        params: list[Any] = [cluster_id]
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        sql += " ORDER BY COALESCE(published_at, retrieved_at) DESC"
        return [self.row_to_article(r) for r in self.db.query(sql, tuple(params))]

    def search_fts(
        self, match_query: str, *, limit: int = 120, hours: int | None = None
    ) -> list[tuple[Article, float]]:
        """BM25 full-text search. Returns (article, relevance in 0..1) pairs.

        Column weights favour titles, then entities/topics, then body text.
        """
        sql = f"""
            SELECT {', '.join('a.' + c.strip() for c in self._COLUMNS.split(','))},
                   bm25(articles_fts, 8.0, 3.0, 1.0, 5.0, 4.0) AS rank
            FROM articles_fts
            JOIN articles a ON a.id = articles_fts.rowid
            WHERE articles_fts MATCH ?
        """
        params: list[Any] = [match_query]
        if hours is not None:
            sql += " AND COALESCE(a.published_at, a.retrieved_at) >= ?"
            params.append(_iso(utcnow() - timedelta(hours=hours)))
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = self.db.query(sql, tuple(params))
        except sqlite3.OperationalError as exc:
            # A malformed MATCH expression is a user-input problem, not a bug.
            log.debug("FTS query rejected (%s): %s", exc, match_query)
            return []

        results: list[tuple[Article, float]] = []
        for row in rows:
            # bm25() returns a negative score; more negative = better.
            raw = -float(row["rank"])
            results.append((self.row_to_article(row), raw))
        if not results:
            return []
        top = max(score for _, score in results) or 1.0
        return [(article, min(1.0, score / top)) for article, score in results]

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM articles")
        return int(row["n"]) if row else 0

    def stats(self) -> dict[str, Any]:
        row = self.db.query_one(
            """
            SELECT COUNT(*) AS total,
                   COUNT(DISTINCT source_id) AS sources,
                   MAX(retrieved_at) AS last_ingest,
                   SUM(CASE WHEN COALESCE(published_at, retrieved_at) >= ? THEN 1 ELSE 0 END) AS last_24h
            FROM articles
            """,
            (_iso(utcnow() - timedelta(hours=24)),),
        )
        return dict(row) if row else {}


class ClusterRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, cluster_id: str, key_title: str, size: int) -> None:
        now = _iso(utcnow())
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO clusters (id, key_title, size, created_at, updated_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    size = excluded.size, updated_at = excluded.updated_at,
                    key_title = CASE WHEN clusters.key_title = '' THEN excluded.key_title
                                     ELSE clusters.key_title END
                """,
                (cluster_id, key_title, size, now, now),
            )

    def upsert_many(self, items: Sequence[tuple[str, str, int]]) -> None:
        if not items:
            return
        now = _iso(utcnow())
        with self.db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO clusters (id, key_title, size, created_at, updated_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    size = excluded.size, updated_at = excluded.updated_at
                """,
                [(cid, title, size, now, now) for cid, title, size in items],
            )


class PreferencesRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def load(self) -> Preferences | None:
        row = self.db.query_one("SELECT data, updated_at FROM preferences WHERE id = 1")
        if not row:
            return None
        data = loads(row["data"], {}) or {}
        return Preferences(
            interests=list(data.get("interests") or []),
            raw_interests_text=data.get("raw_interests_text") or "",
            onboarded=bool(data.get("onboarded")),
            extras=dict(data.get("extras") or {}),
            updated_at=_parse_dt(row["updated_at"]) or utcnow(),
        )

    def save(self, prefs: Preferences) -> Preferences:
        prefs.updated_at = utcnow()
        payload = {
            "interests": prefs.interests,
            "raw_interests_text": prefs.raw_interests_text,
            "onboarded": prefs.onboarded,
            "extras": prefs.extras,
        }
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO preferences (id, data, updated_at) VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
                """,
                (dumps(payload), _iso(prefs.updated_at)),
            )
        return prefs

    def reset(self) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM preferences WHERE id = 1")


class BriefingRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, payload: dict[str, Any]) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO briefings (brief_date, generated_at, engine, interests,
                                       story_count, payload)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    payload.get("date", ""),
                    payload.get("generated_at", _iso(utcnow())),
                    payload.get("engine", ""),
                    dumps(payload.get("interests", [])),
                    int(payload.get("story_count", 0)),
                    dumps(payload),
                ),
            )
            return int(cur.lastrowid)

    def latest(self) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT id, payload FROM briefings ORDER BY generated_at DESC, id DESC LIMIT 1"
        )
        if not row:
            return None
        payload = loads(row["payload"], {}) or {}
        payload["id"] = row["id"]
        return payload

    def prune(self, keep: int = 30) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM briefings WHERE id NOT IN "
                "(SELECT id FROM briefings ORDER BY generated_at DESC, id DESC LIMIT ?)",
                (keep,),
            )


class ChatRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_session(self, session_id: str, title: str = "") -> None:
        now = _iso(utcnow())
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at,
                    title = CASE WHEN chat_sessions.title = '' THEN excluded.title
                                 ELSE chat_sessions.title END
                """,
                (session_id, title[:120], now, now),
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at, sources, meta)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    session_id,
                    role,
                    content,
                    _iso(utcnow()),
                    dumps(sources or []),
                    dumps(meta or {}),
                ),
            )
            return int(cur.lastrowid)

    def history(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT role, content, created_at, sources, meta FROM chat_messages
            WHERE session_id = ? ORDER BY id DESC LIMIT ?
            """,
            (session_id, limit),
        )
        out = [
            {
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
                "sources": loads(r["sources"], []) or [],
                "meta": loads(r["meta"], {}) or {},
            }
            for r in rows
        ]
        out.reverse()
        return out

    def clear(self, session_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


__all__ = [
    "ArticleRepository",
    "BriefingRepository",
    "ChatRepository",
    "ClusterRepository",
    "PreferencesRepository",
    "SourceRepository",
    "json",
]
