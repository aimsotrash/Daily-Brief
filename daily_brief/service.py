"""Application container.

One place where the components are wired together, so the API, the CLI and the
scheduler all drive exactly the same objects. This is the whole of the
"application layer" -- a personal news reader does not need more.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from .analysis.relevance import Interest
from .briefing.generator import BriefingGenerator
from .config import Config, load_config
from .db import Database
from .llm.base import LLMProvider
from .llm.factory import build_provider
from .logging_setup import setup_logging
from .news.ingest import IngestReport, NewsIngestor
from .news.sources import RegistryError, load_registry
from .preferences import PreferencesService
from .repository import (
    ArticleRepository,
    BriefingRepository,
    ChatRepository,
    ClusterRepository,
    PreferencesRepository,
    SourceRepository,
)
from .search.answer import AnswerResult
from .search.chat import NewsSearchService
from .search.retriever import Retriever

log = logging.getLogger(__name__)


class Application:
    """Owns the database, repositories, services and the LLM provider."""

    def __init__(self, config: Config | None = None, *, db: Database | None = None) -> None:
        self.config = config or load_config()
        setup_logging(self.config)

        self.db = db or Database(self.config.storage.db_path())
        self.db.connect()

        self.article_repo = ArticleRepository(self.db)
        self.source_repo = SourceRepository(self.db)
        self.cluster_repo = ClusterRepository(self.db)
        self.preference_repo = PreferencesRepository(self.db)
        self.briefing_repo = BriefingRepository(self.db)
        self.chat_repo = ChatRepository(self.db)

        self.preferences = PreferencesService(self.preference_repo)
        self.provider: LLMProvider = build_provider(self.config.llm)
        self.retriever = Retriever(self.article_repo, self.config.search)
        self.search = NewsSearchService(
            self.config, self.retriever, self.provider, self.chat_repo
        )
        self.briefings = BriefingGenerator(
            self.config, self.article_repo, self.briefing_repo, self.provider
        )
        self.ingestor = NewsIngestor(
            self.config, self.article_repo, self.source_repo, self.cluster_repo
        )

        self.source_methodology: dict[str, Any] = {}
        self.registry_error: str = ""
        self.sync_sources()

    # ------------------------------------------------------------------ setup
    def sync_sources(self) -> int:
        """Load the source registry into the database.

        A broken registry is reported but does not prevent startup: previously
        synced sources remain usable, and the UI surfaces the error.
        """
        try:
            sources, methodology = load_registry(self.config.sources_path())
        except RegistryError as exc:
            self.registry_error = str(exc)
            log.error("source registry unusable: %s", exc)
            return 0
        self.source_methodology = methodology
        self.registry_error = ""
        return self.source_repo.upsert_many(sources)

    # --------------------------------------------------------------- pipeline
    async def refresh(self, source_ids: Sequence[str] | None = None) -> IngestReport:
        return await self.ingestor.run(source_ids)

    def reanalyze(self, *, limit: int = 20000) -> int:
        """Recompute topics, entities and bias for every stored article.

        Analysis results are cached on the row at ingest time, so changing the
        lexicon, the classifier or the bias heuristics needs a reprocessing pass.
        This does no network I/O -- it is pure recomputation over stored text.
        """
        from .analysis import bias as bias_analysis
        from .analysis import topics as topic_analysis
        from .text import extract_entities

        sources = self.source_repo.by_id_map()
        stored = self.article_repo.recent(hours=None, limit=limit)
        for article in stored:
            article.source = sources.get(article.source_id)
            article.entities = extract_entities(
                f"{article.title}. {article.best_text[:2000]}"
            )
            article.topics = topic_analysis.classify(article)
            article.bias = bias_analysis.analyse_article(article, article.source)
            self.article_repo.update_analysis(article)
        log.info("re-analysed %d articles", len(stored))
        return len(stored)

    async def generate_briefing(self, *, persist: bool = True) -> dict:
        return await self.briefings.generate(self.preferences.interests(), persist=persist)

    async def get_briefing(self, *, refresh_if_stale: bool = True) -> dict:
        payload = self.briefings.latest()
        if refresh_if_stale and self.briefings.is_stale(payload):
            log.info("briefing missing or stale; regenerating")
            return await self.generate_briefing()
        return payload or await self.generate_briefing()

    async def ask(
        self, question: str, *, session_id: str | None = None, use_history: bool = True
    ) -> AnswerResult:
        return await self.search.search(
            question,
            session_id=session_id,
            interests=self.preferences.interests(),
            use_history=use_history,
        )

    def interests(self) -> list[Interest]:
        return self.preferences.interests()

    # ----------------------------------------------------------------- status
    async def status(self) -> dict[str, Any]:
        prefs = self.preferences.load()
        stats = self.article_repo.stats()
        feed_state = self.source_repo.all_feed_state()
        sources = self.source_repo.all()
        failing = [
            {
                "id": sid,
                "error": state.get("last_error") or "",
                "failures": state.get("consecutive_failures") or 0,
            }
            for sid, state in feed_state.items()
            if (state.get("consecutive_failures") or 0) > 0
        ]
        latest = self.briefing_repo.latest()
        return {
            "onboarded": prefs.onboarded,
            "interests": prefs.interests,
            "articles": {
                "total": stats.get("total") or 0,
                "last_24h": stats.get("last_24h") or 0,
                "sources_with_articles": stats.get("sources") or 0,
                "last_ingest": stats.get("last_ingest"),
            },
            "sources": {
                "total": len(sources),
                "enabled": sum(1 for s in sources if s.enabled),
                "failing": sorted(failing, key=lambda f: -f["failures"])[:10],
                "registry_error": self.registry_error,
            },
            "llm": {
                **self.provider.describe(),
                "available": await self.provider.available(),
                "fallback_to_extractive": self.config.llm.fallback_to_extractive,
            },
            "briefing": {
                "generated_at": (latest or {}).get("generated_at"),
                "story_count": (latest or {}).get("story_count", 0),
                "stale": self.briefings.is_stale(latest),
            },
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    async def aclose(self) -> None:
        await self.provider.aclose()
        self.db.close()
