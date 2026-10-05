"""The search/chat service.

Sequences the pipeline the product specifies:

    query -> understand -> retrieve -> select -> process -> generate -> attribute

Conversation state is kept only to interpret follow-up questions ("only the
biggest ones", "how are sources covering this?"). It is scoped to news research
on purpose: prior turns influence *retrieval*, never the facts in an answer,
which always come from the articles retrieved for the current turn.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..analysis.relevance import Interest
from ..config import Config
from ..llm.base import LLMProvider
from ..models import ScoredArticle
from ..repository import ChatRepository
from .answer import EMPTY_QUERY_ANSWER, AnswerGenerator, AnswerResult
from .query import ParsedQuery, parse_query, refine_with_llm
from .retriever import Retriever

log = logging.getLogger(__name__)


class NewsSearchService:
    def __init__(
        self,
        config: Config,
        retriever: Retriever,
        provider: LLMProvider,
        chats: ChatRepository,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.provider = provider
        self.chats = chats
        self.generator = AnswerGenerator(
            provider, fallback_to_extractive=config.llm.fallback_to_extractive
        )

    def new_session_id(self) -> str:
        return uuid.uuid4().hex[:16]

    async def search(
        self,
        question: str,
        *,
        session_id: str | None = None,
        interests: Sequence[Interest] | None = None,
        use_history: bool = True,
        persist: bool = True,
        now: datetime | None = None,
    ) -> AnswerResult:
        now = now or datetime.now(timezone.utc)
        question = (question or "").strip()

        if not question:
            return AnswerResult(
                answer=EMPTY_QUERY_ANSWER,
                sources=[],
                engine="none",
                has_results=False,
                intent="empty",
                warnings=["Empty query."],
            )

        history: list[dict] = []
        if session_id and use_history:
            history = self.chats.history(session_id, limit=self.config.search.history_turns * 2)

        # 1. Understand the query (deterministic, optionally LLM-expanded).
        parsed = parse_query(question, history)
        parsed = await refine_with_llm(parsed, self.provider, history)

        if parsed.is_empty():
            log.debug("query produced no usable retrieval terms: %r", question)

        # 2. Retrieve, 3. select.
        scored: list[ScoredArticle] = self.retriever.retrieve(
            parsed,
            interests=interests,
            limit=self.config.search.context_articles,
            now=now,
        )

        # 4. Group related coverage.
        if parsed.intent == "compare_coverage":
            scored = self._expand_for_comparison(scored)
            groups = self.retriever.group_into_stories(scored, include_related=False)
        else:
            groups = self.retriever.group_into_stories(scored)

        # 5/6. Generate a grounded response with attribution.
        result = await self.generator.generate(
            question,
            scored,
            groups,
            intent=parsed.intent,
            history=history,
            query_info=parsed.to_dict(),
            now=now,
        )

        if persist and session_id:
            self._persist(session_id, question, result)
        return result

    def _expand_for_comparison(self, scored: list[ScoredArticle]) -> list[ScoredArticle]:
        """Pull in the rest of the top story's cluster so a comparison has a sample.

        A coverage question is about one story; retrieval diversity caps would
        otherwise leave only one or two articles from it.
        """
        if not scored:
            return scored
        lead = scored[0].article
        siblings = self.retriever.coverage_for(lead, [s.article for s in scored[1:]])
        by_url = {s.article.display_url: s for s in scored}
        expanded: list[ScoredArticle] = []
        for article in siblings:
            existing = by_url.get(article.display_url)
            if existing is not None:
                expanded.append(existing)
            else:
                expanded.append(ScoredArticle(article=article, score=0.0))
        # Keep the comparison to a readable size.
        return expanded[: max(self.config.search.context_articles, 10)]

    def _persist(self, session_id: str, question: str, result: AnswerResult) -> None:
        try:
            self.chats.ensure_session(session_id, title=question[:120])
            self.chats.add_message(session_id, "user", question)
            self.chats.add_message(
                session_id,
                "assistant",
                result.answer,
                sources=result.sources,
                meta={
                    "engine": result.engine,
                    "intent": result.intent,
                    "has_results": result.has_results,
                    "warnings": result.warnings,
                    "comparison": result.comparison,
                    "cited": result.cited,
                },
            )
        except Exception as exc:  # pragma: no cover - persistence must not break search
            log.warning("could not persist chat turn: %s", exc)

    def history(self, session_id: str) -> list[dict]:
        return self.chats.history(session_id, limit=200)

    def clear(self, session_id: str) -> None:
        self.chats.clear(session_id)
