"""Retrieval.

Hybrid lexical retrieval over the SQLite FTS5 index, re-ranked with recency,
entity match, interest affinity and source diversity, then grouped into stories.

Why no vector embeddings: the corpus is a rolling window of a few thousand short
news items, queries are entity-heavy ("NVIDIA", "OpenAI", "Linux 6.x"), and BM25
over titles plus extracted proper nouns handles that extremely well at zero
model cost. A dense stage would add a multi-hundred-megabyte dependency for a
marginal gain on this workload. :class:`Retriever` is a single seam, so a dense
or reranking stage can be added later without touching callers.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Sequence

from ..analysis.relevance import Interest, score_article as score_interest
from ..config import SearchConfig
from ..models import Article, ScoredArticle, StoryGroup
from ..repository import ArticleRepository
from ..text import contains_term, fold
from .query import ParsedQuery

log = logging.getLogger(__name__)

#: Characters FTS5 treats as operators; terms containing them get quoted.
_FTS_SAFE = re.compile(r"^[A-Za-z0-9_]+$")

#: Minimum weighted share of the query an article must contain to be a hit,
#: unless it matched one of the query's named entities outright.
MIN_COVERAGE = 0.34

#: Widest fallback window. Nothing older than the retention horizon is stored,
#: so this effectively means "search everything we have".
MAX_WINDOW_HOURS = 24 * 60


def _quote(term: str) -> str:
    """Quote a term as an FTS5 string literal."""
    return '"' + term.replace('"', '""') + '"'


def _clause(term: str) -> str:
    cleaned = " ".join(term.split())
    if _FTS_SAFE.match(cleaned) and len(cleaned) >= 4:
        # Prefix match catches inflections ("tariff" -> "tariffs", "tariffed").
        return f"{cleaned}*"
    return _quote(cleaned)


def build_match_expression(parsed: ParsedQuery, *, broad: bool = False) -> str:
    """Build an FTS5 MATCH expression from a parsed query.

    Two modes. The default is **entity-anchored**: if the question names any
    proper nouns, only those are matched. Asking "what's happening with NVIDIA
    today?" should not return every hardware article merely because the query
    also carries the topic ``hardware``. ``broad=True`` ORs everything and is
    used as a fallback when the anchored expression finds nothing.

    Terms are OR-ed rather than AND-ed within a mode: BM25 already rewards
    documents matching more of them, and AND would fail on multi-concept
    questions where no single article contains every word.

    Topics are deliberately never part of the expression -- they are far too
    broad as free-text terms, and :func:`_topic_score` already applies them
    during re-ranking.
    """
    clauses: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        key = fold(term)
        if not key or key in seen or len(key) < 2:
            return
        seen.add(key)
        clauses.append(_clause(term))

    for entity in parsed.entities:
        add(entity)

    if broad or not clauses:
        for keyword in parsed.keywords:
            add(keyword)

    return " OR ".join(clauses)


def recency_score(article: Article, now: datetime, half_life_hours: float = 20.0) -> float:
    """Exponential decay on article age. Undated articles get a neutral score."""
    if article.published_at is None:
        return 0.35
    age = article.age_hours(now)
    return math.pow(0.5, age / half_life_hours)


def _entity_score(article: Article, parsed: ParsedQuery) -> float:
    if not parsed.entities:
        return 0.0
    title = fold(article.title)
    body = fold(article.summary or article.content[:800])
    article_entities = {fold(e) for e in article.entities}
    hits = 0.0
    for entity in parsed.entities:
        needle = fold(entity)
        if not needle:
            continue
        if needle in title:
            hits += 1.0
        elif needle in article_entities:
            hits += 0.8
        elif needle in body:
            hits += 0.45
    return min(1.0, hits / max(1, len(parsed.entities)))


def _coverage(article: Article, parsed: ParsedQuery) -> tuple[float, bool]:
    """How much of the query the article actually contains.

    Returns ``(weighted coverage in 0..1, matched at least one entity)``.
    Without this, an OR-ed expression lets an article through on a single
    incidental common word -- "quantum banana treaty" matching a story about a
    python found in a banana plantation.
    """
    terms = parsed.all_terms
    if not terms:
        return (0.0, False)

    haystack = fold(
        " ".join(
            [
                article.title,
                article.summary,
                article.content[:1500],
                " ".join(article.entities),
                " ".join(article.topics),
            ]
        )
    )
    entity_keys = {fold(e) for e in parsed.entities}

    total = matched = 0.0
    entity_hit = False
    for term in terms:
        key = fold(term)
        weight = 1.5 if key in entity_keys else 1.0
        total += weight
        if contains_term(haystack, key):
            matched += weight
            if key in entity_keys:
                entity_hit = True
    return (matched / total if total else 0.0, entity_hit)


def _topic_score(article: Article, parsed: ParsedQuery) -> float:
    if not parsed.topics:
        return 0.0
    overlap = set(article.topics) & set(parsed.topics)
    if not overlap:
        return 0.0
    # A leading topic match is worth more than an incidental one.
    lead_bonus = 0.4 if article.topics and article.topics[0] in overlap else 0.0
    return min(1.0, 0.6 * len(overlap) / len(parsed.topics) + lead_bonus)


class Retriever:
    """Selects the articles a grounded answer will be built from."""

    def __init__(self, articles: ArticleRepository, config: SearchConfig) -> None:
        self.articles = articles
        self.config = config

    def retrieve(
        self,
        parsed: ParsedQuery,
        *,
        interests: Sequence[Interest] | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[ScoredArticle]:
        """Return ranked articles. An empty list means "no relevant results"."""
        now = now or datetime.now(timezone.utc)
        limit = limit or self.config.context_articles

        # Try progressively broader retrievals, stopping at the first that
        # yields articles which actually cover the query.
        for broad in (False, True):
            match = build_match_expression(parsed, broad=broad)
            if not match:
                continue
            for hours in self._windows(parsed.time_window_hours):
                hits = self.articles.search_fts(
                    match, limit=self.config.candidate_limit, hours=hours
                )
                scored = self._score(hits, parsed, interests, now)
                if scored:
                    return self._diversify(scored, limit)
        return []

    @staticmethod
    def _windows(window: int) -> list[int]:
        """The requested window, then the whole index before giving up.

        A story the user asks about may simply be older than their phrasing
        implied. Preferring recent matches but falling back to everything stored
        beats answering "no results" for an article that is sitting in the index.
        """
        return [window] if window >= MAX_WINDOW_HOURS else [window, MAX_WINDOW_HOURS]

    def _score(
        self,
        hits: Sequence[tuple[Article, float]],
        parsed: ParsedQuery,
        interests: Sequence[Interest] | None,
        now: datetime,
    ) -> list[ScoredArticle]:
        scored: list[ScoredArticle] = []
        for article, lexical in hits:
            coverage, entity_hit = _coverage(article, parsed)
            # An article must genuinely cover the query, not merely share one
            # incidental word with it.
            if not entity_hit and coverage < MIN_COVERAGE:
                continue

            recency = recency_score(article, now)
            entity = _entity_score(article, parsed)
            topic = _topic_score(article, parsed)
            interest_score = 0.0
            if interests:
                interest_score, _ = score_interest(article, list(interests))

            blended = (
                0.32 * lexical
                + 0.22 * coverage
                + 0.20 * entity
                + 0.10 * topic
                + 0.11 * recency
                + 0.05 * interest_score
            )

            reasons: list[str] = []
            if entity >= 0.5:
                reasons.append("matches the entities in the query")
            if topic >= 0.5:
                reasons.append("matches the query topic")
            if recency >= 0.5:
                reasons.append("published recently")

            scored.append(
                ScoredArticle(
                    article=article,
                    score=blended,
                    lexical=lexical,
                    recency=recency,
                    interest=interest_score,
                    entity=entity,
                    reasons=reasons,
                )
            )

        scored = [s for s in scored if s.score >= self.config.min_score]
        scored.sort(key=lambda s: -s.score)
        return scored

    def _diversify(self, scored: list[ScoredArticle], limit: int) -> list[ScoredArticle]:
        """Cap per-source and per-cluster dominance so one outlet cannot fill the answer."""
        per_source_cap = max(2, limit // 3)
        per_cluster_cap = 2
        source_counts: dict[str, int] = {}
        cluster_counts: dict[str, int] = {}
        chosen: list[ScoredArticle] = []
        overflow: list[ScoredArticle] = []

        for item in scored:
            sid = item.article.source_id
            cid = item.article.cluster_id or f"solo:{item.article.id}"
            if source_counts.get(sid, 0) >= per_source_cap or (
                cluster_counts.get(cid, 0) >= per_cluster_cap
            ):
                overflow.append(item)
                continue
            source_counts[sid] = source_counts.get(sid, 0) + 1
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            chosen.append(item)
            if len(chosen) >= limit:
                return chosen

        # Backfill from overflow rather than returning fewer results than asked.
        for item in overflow:
            if len(chosen) >= limit:
                break
            chosen.append(item)
        return chosen

    def group_into_stories(
        self, scored: Sequence[ScoredArticle], *, include_related: bool = True
    ) -> list[StoryGroup]:
        """Fold ranked articles into story groups using stored cluster ids."""
        groups: list[StoryGroup] = []
        seen_clusters: set[str] = set()

        for item in scored:
            article = item.article
            cid = article.cluster_id
            if cid and cid in seen_clusters:
                continue
            related: list[Article] = []
            if include_related and cid:
                seen_clusters.add(cid)
                # Enough to enumerate every outlet in a coverage list; the
                # answer prompt still only receives the lead article.
                related = self.articles.cluster_members(cid, exclude_id=article.id)[:12]
            groups.append(StoryGroup(lead=article, related=related, score=item.score))
        return groups

    def coverage_for(self, article: Article, extra: Sequence[Article] = ()) -> list[Article]:
        """All known coverage of one story: its cluster plus supplied extras."""
        members = [article]
        if article.cluster_id:
            members.extend(
                self.articles.cluster_members(article.cluster_id, exclude_id=article.id)
            )
        seen = {a.canonical_url or a.url for a in members}
        for other in extra:
            key = other.canonical_url or other.url
            if key not in seen:
                seen.add(key)
                members.append(other)
        return members
