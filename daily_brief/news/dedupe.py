"""Deduplication and story clustering.

Three layers, cheapest first:

1. **Exact**  -- identical canonical URL or identical (title, url) hash.
2. **Near**   -- same publication, near-identical headline (SimHash + token
   containment). Catches a feed re-emitting a story after a small edit.
3. **Story clustering** -- different publications covering the same event.
   Grouped by headline-token overlap plus shared entities inside a time window,
   resolved with union-find.

No embedding model is involved. For short news headlines within a rolling window
of a day or two, lexical overlap plus proper-noun overlap is both effective and
two orders of magnitude cheaper than vector similarity.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime
from typing import Iterable, Sequence

from ..models import Article
from ..text import containment, hamming, jaccard, tokenize

log = logging.getLogger(__name__)

#: SimHash distance under which two headlines from one source are "the same".
SIMHASH_NEAR_THRESHOLD = 6
#: Headline token containment above which two articles are the same story.
STORY_TITLE_THRESHOLD = 0.62
#: Lower headline threshold accepted when entities also overlap strongly.
STORY_TITLE_WITH_ENTITY_THRESHOLD = 0.42
STORY_ENTITY_THRESHOLD = 0.34
#: Articles more than this many hours apart are not clustered together.
STORY_WINDOW_HOURS = 48


def dedupe_exact(articles: Sequence[Article]) -> tuple[list[Article], int]:
    """Collapse articles that share a canonical URL or content hash.

    When duplicates are found, the richest version wins (longest body text,
    then the one that has a publication date).
    """
    best: dict[str, Article] = {}
    order: list[str] = []
    removed = 0

    for article in articles:
        key = article.canonical_url or article.url
        alt = article.content_hash
        existing_key = key if key in best else (alt if alt in best else None)
        if existing_key is None:
            best[key] = article
            if alt and alt != key:
                best[alt] = article
            order.append(key)
            continue

        removed += 1
        incumbent = best[existing_key]
        if _richness(article) > _richness(incumbent):
            for k, v in list(best.items()):
                if v is incumbent:
                    best[k] = article

    seen: set[int] = set()
    out: list[Article] = []
    for key in order:
        article = best.get(key)
        if article is not None and id(article) not in seen:
            seen.add(id(article))
            out.append(article)
    return out, removed


def _richness(article: Article) -> tuple[int, int, int]:
    return (
        len(article.content),
        len(article.summary),
        1 if article.published_at else 0,
    )


def dedupe_near(articles: Sequence[Article]) -> tuple[list[Article], int]:
    """Drop near-identical headlines from the *same* source.

    Cross-source near-duplicates are legitimate independent coverage and are
    handled by clustering instead, not removal.
    """
    kept: list[Article] = []
    by_source: dict[str, list[Article]] = defaultdict(list)
    removed = 0

    for article in articles:
        tokens = tokenize(article.title)
        duplicate_of: Article | None = None
        for other in by_source[article.source_id]:
            if hamming(article.simhash, other.simhash) <= SIMHASH_NEAR_THRESHOLD:
                duplicate_of = other
                break
            if containment(tokens, tokenize(other.title)) >= 0.9:
                duplicate_of = other
                break
        if duplicate_of is None:
            by_source[article.source_id].append(article)
            kept.append(article)
        else:
            removed += 1
            if _richness(article) > _richness(duplicate_of):
                # Replace the weaker copy in place, preserving order.
                kept[kept.index(duplicate_of)] = article
                by_source[article.source_id][
                    by_source[article.source_id].index(duplicate_of)
                ] = article
    return kept, removed


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _hours_apart(a: Article, b: Article) -> float:
    left = a.published_at or a.retrieved_at
    right = b.published_at or b.retrieved_at
    if not isinstance(left, datetime) or not isinstance(right, datetime):
        return 0.0
    return abs((left - right).total_seconds()) / 3600.0


def _same_story(a: Article, b: Article, ta: set[str], tb: set[str]) -> bool:
    if _hours_apart(a, b) > STORY_WINDOW_HOURS:
        return False
    title_overlap = containment(ta, tb)
    if title_overlap >= STORY_TITLE_THRESHOLD:
        return True
    if title_overlap < STORY_TITLE_WITH_ENTITY_THRESHOLD:
        return False
    ea = {e.lower() for e in a.entities}
    eb = {e.lower() for e in b.entities}
    return bool(ea and eb) and jaccard(ea, eb) >= STORY_ENTITY_THRESHOLD


def cluster_id_for(articles: Iterable[Article]) -> str:
    """Deterministic cluster id from the member URLs, so reruns are stable."""
    urls = sorted((a.canonical_url or a.url) for a in articles)
    digest = hashlib.blake2b("\n".join(urls).encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def cluster_stories(articles: Sequence[Article]) -> list[list[Article]]:
    """Group articles into story clusters. Singletons form their own cluster.

    O(n^2) over the candidate window. With a rolling window of a few hundred
    articles that is a handful of milliseconds; blocking by shared token would
    be the next step if the window ever grew.
    """
    n = len(articles)
    if n <= 1:
        return [[a] for a in articles]

    token_sets = [set(tokenize(a.title)) for a in articles]
    # Inverted index over title tokens: only compare articles sharing a token.
    postings: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            postings[token].append(index)

    uf = _UnionFind(n)
    compared: set[tuple[int, int]] = set()
    for indices in postings.values():
        # A token appearing nearly everywhere carries no signal.
        if len(indices) > 60:
            continue
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                pair = (left, right)
                if pair in compared:
                    continue
                compared.add(pair)
                if _same_story(
                    articles[left], articles[right], token_sets[left], token_sets[right]
                ):
                    uf.union(left, right)

    groups: dict[int, list[Article]] = defaultdict(list)
    for index, article in enumerate(articles):
        groups[uf.find(index)].append(article)

    clusters = list(groups.values())
    for cluster in clusters:
        cluster.sort(
            key=lambda a: (a.published_at or a.retrieved_at).timestamp(), reverse=True
        )
    clusters.sort(key=len, reverse=True)
    return clusters


def assign_clusters(articles: Sequence[Article]) -> dict[str, list[Article]]:
    """Cluster and stamp ``cluster_id`` onto each article."""
    result: dict[str, list[Article]] = {}
    for cluster in cluster_stories(articles):
        cid = cluster_id_for(cluster)
        for article in cluster:
            article.cluster_id = cid
        result[cid] = cluster
    return result
