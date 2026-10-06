"""The ingestion pipeline.

    sources -> fetch -> parse -> normalize -> dedupe -> classify -> analyse -> store

Each stage is a separate module; this file only sequences them and records what
happened. Failures are per-source: one dead feed reduces coverage, it never
fails the run.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Sequence

from ..analysis import bias as bias_analysis
from ..analysis import topics as topic_analysis
from ..config import Config
from ..models import Article, Source
from ..repository import ArticleRepository, ClusterRepository, SourceRepository
from . import dedupe, normalize
from .feeds import FeedFetcher, FetchResult
from .parser import FeedParseError, parse_feed

log = logging.getLogger(__name__)


@dataclass
class SourceReport:
    source_id: str
    source_name: str
    status: str
    entries: int = 0
    accepted: int = 0
    error: str = ""


@dataclass
class IngestReport:
    """What one ingestion run did. Surfaced by the CLI and the /api/status route."""

    sources_total: int = 0
    sources_ok: int = 0
    sources_not_modified: int = 0
    sources_failed: int = 0
    entries_seen: int = 0
    articles_normalized: int = 0
    duplicates_exact: int = 0
    duplicates_near: int = 0
    inserted: int = 0
    updated: int = 0
    clusters: int = 0
    pruned: int = 0
    per_source: list[SourceReport] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.inserted} new, {self.updated} updated, "
            f"{self.duplicates_exact + self.duplicates_near} duplicates dropped, "
            f"{self.clusters} story clusters, "
            f"{self.sources_ok + self.sources_not_modified}/{self.sources_total} sources ok "
            f"({self.sources_not_modified} unchanged since the last fetch, "
            f"{self.sources_failed} failed)"
        )


class NewsIngestor:
    def __init__(
        self,
        config: Config,
        articles: ArticleRepository,
        sources: SourceRepository,
        clusters: ClusterRepository,
        fetcher: FeedFetcher | None = None,
    ) -> None:
        self.config = config
        self.articles = articles
        self.sources = sources
        self.clusters = clusters
        self.fetcher = fetcher or FeedFetcher(config.ingest)

    # ------------------------------------------------------------------ public
    async def run(self, source_ids: Sequence[str] | None = None) -> IngestReport:
        registry = self.sources.all(enabled_only=True)
        if source_ids:
            wanted = set(source_ids)
            registry = [s for s in registry if s.id in wanted]

        state = self.sources.all_feed_state()
        limit = self.config.ingest.max_consecutive_failures
        active = [
            s
            for s in registry
            if state.get(s.id, {}).get("consecutive_failures", 0) < limit
        ]
        skipped = len(registry) - len(active)
        if skipped:
            log.info("skipping %d source(s) with too many consecutive failures", skipped)

        report = IngestReport(sources_total=len(active))
        if not active:
            return report

        log.info("fetching %d feeds", len(active))
        results = await self.fetcher.fetch_many(active, state)
        return await asyncio.to_thread(self._process, results, report)

    # ----------------------------------------------------------------- private
    def _process(self, results: list[FetchResult], report: IngestReport) -> IngestReport:
        by_source = {s.id: s for s in self.sources.all()}
        candidates: list[Article] = []

        for result in results:
            source = result.source
            entry = SourceReport(source.id, source.name, result.status)

            if result.status == "not-modified":
                report.sources_not_modified += 1
                self.sources.record_fetch(
                    source.id,
                    status="not-modified",
                    etag=result.etag,
                    last_modified=result.last_modified,
                )
                report.per_source.append(entry)
                continue

            if not result.ok:
                report.sources_failed += 1
                entry.error = result.error
                self.sources.record_fetch(
                    source.id, status="error", error=result.error or "unknown error"
                )
                report.per_source.append(entry)
                continue

            try:
                feed = parse_feed(result.body)
            except FeedParseError as exc:
                report.sources_failed += 1
                entry.status = "error"
                entry.error = str(exc)
                log.warning("could not parse feed %s: %s", source.id, exc)
                self.sources.record_fetch(
                    source.id, status="error", error=f"parse: {exc}"
                )
                report.per_source.append(entry)
                continue

            entry.entries = len(feed.entries)
            report.entries_seen += len(feed.entries)

            articles = normalize.normalize_entries(
                feed.entries, source, self.config.ingest.max_articles_per_feed
            )
            for article in articles:
                article.source = source
            entry.accepted = len(articles)
            candidates.extend(articles)

            report.sources_ok += 1
            self.sources.record_fetch(
                source.id,
                status="ok",
                etag=result.etag,
                last_modified=result.last_modified,
            )
            report.per_source.append(entry)

        report.articles_normalized = len(candidates)
        if not candidates:
            log.info("ingest: no new candidate articles")
            return report

        candidates, report.duplicates_exact = dedupe.dedupe_exact(candidates)
        candidates, report.duplicates_near = dedupe.dedupe_near(candidates)

        # Skip work on articles already stored unchanged.
        known = self.articles.existing_urls([a.canonical_url for a in candidates])
        fresh = [a for a in candidates if a.canonical_url not in known]

        for article in fresh:
            article.topics = topic_analysis.classify(article)
            article.bias = bias_analysis.analyse_article(
                article, by_source.get(article.source_id)
            )

        if fresh:
            report.inserted, report.updated = self.articles.upsert_many(fresh)

        report.clusters = self._recluster()
        report.pruned = self.articles.prune(self.config.ingest.retention_days)
        log.info("ingest complete: %s", report.summary())
        return report

    def _recluster(self) -> int:
        """Rebuild story clusters over the recent window.

        Clustering is global rather than per-batch: a story that broke yesterday
        should absorb today's follow-up coverage from other outlets.
        """
        window = max(self.config.briefing.lookback_hours, dedupe.STORY_WINDOW_HOURS)
        recent = self.articles.recent(hours=window, limit=1500)
        if not recent:
            return 0
        grouped = dedupe.assign_clusters(recent)
        self.articles.set_cluster(
            {a.id: a.cluster_id for a in recent if a.id is not None}
        )
        self.clusters.upsert_many(
            [(cid, members[0].title, len(members)) for cid, members in grouped.items()]
        )
        return sum(1 for members in grouped.values() if len(members) > 1)
