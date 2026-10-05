"""Grounding and hallucination prevention.

The product's central promise is that Daily-Brief does not fabricate news. These
tests cover all three enforcement layers:

1. structural  -- with no retrieved articles the model is never invoked;
2. prompted    -- retrieved article text and metadata reach the model;
3. verified    -- citations are checked against what was actually supplied.
"""

from __future__ import annotations

import pytest

from daily_brief.llm.prompts import build_search_prompt, format_article_block
from daily_brief.search.answer import (
    NO_RESULTS_ANSWER,
    AnswerGenerator,
    extractive_answer,
    validate_citations,
)
from daily_brief.search.chat import NewsSearchService
from daily_brief.search.retriever import Retriever

from .conftest import StubProvider, make_article


@pytest.fixture
def grounded_corpus(app):
    app.article_repo.upsert_many([
        make_article(
            "NVIDIA announces the RTX 5090 graphics card",
            source_id="tech-daily", source_name="Tech Daily",
            summary="NVIDIA revealed the RTX 5090 today with 32GB of memory.",
            content="NVIDIA revealed the RTX 5090 today with 32GB of memory. "
                    "The company said it ships in September.",
            topics=["hardware", "nvidia"], entities=["NVIDIA"], age_hours=3,
        ),
        make_article(
            "NVIDIA earnings beat expectations",
            source_id="left-post", source_name="The Left Post",
            summary="NVIDIA reported record quarterly revenue of $41 billion.",
            topics=["business", "nvidia"], entities=["NVIDIA"], age_hours=6,
        ),
    ])
    return app


def service(app, provider) -> NewsSearchService:
    app.provider = provider
    return NewsSearchService(
        app.config, Retriever(app.article_repo, app.config.search), provider, app.chat_repo
    )


# --------------------------------------------------------------- citations


class TestCitationValidation:
    def test_valid_citations_survive(self):
        text, cited, warnings = validate_citations("NVIDIA shipped a GPU [1] and beat earnings [2].", 2)
        assert cited == [1, 2]
        assert warnings == []
        assert "[1]" in text and "[2]" in text

    def test_out_of_range_citation_is_stripped(self):
        text, cited, warnings = validate_citations("A fact [1]. An invented one [9].", 2)
        assert cited == [1]
        assert "[9]" not in text
        assert warnings and "cited sources it was not given" in warnings[0]

    def test_all_citations_invalid_yields_none(self):
        text, cited, warnings = validate_citations("Claim [7]. Another [8].", 3)
        assert cited == []
        assert warnings

    def test_uncited_text_reports_no_citations(self):
        _, cited, warnings = validate_citations("A confident claim with no source.", 3)
        assert cited == [] and warnings == []

    def test_repeated_citations_are_deduplicated(self):
        _, cited, _ = validate_citations("One [1]. Two [1]. Three [2].", 2)
        assert cited == [1, 2]


# ------------------------------------------------------------ prompt content


class TestModelReceivesArticles:
    def test_article_block_carries_required_metadata(self, grounded_corpus):
        articles = grounded_corpus.article_repo.recent(limit=2)
        block = format_article_block(articles)
        for article in articles:
            assert article.title in block
            assert article.source_name in block
            assert article.display_url in block
        assert "published:" in block
        assert "[1]" in block and "[2]" in block

    def test_undated_articles_are_labelled_not_guessed(self):
        article = make_article("No date story")
        article.published_at = None
        block = format_article_block([article])
        assert "publication time not provided" in block

    def test_prompt_states_the_grounding_constraint(self, grounded_corpus):
        articles = grounded_corpus.article_repo.recent(limit=2)
        system = build_search_prompt("What about NVIDIA?", articles)[0]["content"]
        assert "ONLY source of information" in system
        assert "Never invent" in system

    async def test_model_actually_receives_the_retrieved_text(self, grounded_corpus):
        provider = StubProvider(["NVIDIA announced the RTX 5090 [1]."])
        await service(grounded_corpus, provider).search("What about NVIDIA?")
        prompt = provider.last_prompt
        assert "RTX 5090" in prompt
        assert "Tech Daily" in prompt
        assert "32GB of memory" in prompt

    async def test_bias_context_reaches_the_model(self, grounded_corpus):
        provider = StubProvider(["NVIDIA news [1]."])
        await service(grounded_corpus, provider).search("What about NVIDIA?")
        assert "source/framing context" in provider.last_prompt


# ---------------------------------------------------------- no-results path


class TestNoResults:
    async def test_no_results_says_so(self, app):
        provider = StubProvider(["This should never be used."])
        result = await service(app, provider).search("completely unindexed subject matter")
        assert result.has_results is False
        assert result.answer == NO_RESULTS_ANSWER

    async def test_no_results_never_calls_the_model(self, app):
        """The strongest guarantee: with nothing retrieved, generation cannot run.

        Query expansion may still call the model -- it is only asked for search
        terms -- but no prompt containing an ARTICLES block is ever built, so the
        model is never in a position to answer from its own knowledge.
        """
        provider = StubProvider(["FABRICATED: NVIDIA acquired Intel for $200 billion."])
        result = await service(app, provider).search("zzqqx flurbulator quantum banana")
        assert provider.generation_calls == []
        assert not any(
            "ARTICLES" in message.content
            for call in provider.calls
            for message in call
        )
        assert "FABRICATED" not in result.answer
        assert result.engine == "none"

    async def test_no_results_returns_no_sources(self, app):
        result = await service(app, StubProvider([])).search("nothing at all matches this")
        assert result.sources == []
        assert result.cited == []

    async def test_empty_query_is_handled_without_generation(self, app):
        provider = StubProvider(["should not be used"])
        result = await service(app, provider).search("   ")
        assert result.has_results is False
        assert provider.calls == []

    def test_extractive_answer_refuses_with_no_groups(self):
        answer, cited = extractive_answer("anything", [])
        assert answer == NO_RESULTS_ANSWER
        assert cited == []


# ------------------------------------------------------ generation guarantees


class TestGroundedGeneration:
    async def test_response_references_retrieved_sources(self, grounded_corpus):
        provider = StubProvider(["NVIDIA revealed the RTX 5090 [1] and beat earnings [2]."])
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        assert result.cited == [1, 2]
        assert len(result.sources) >= 2
        assert result.grounded is True

    async def test_sources_carry_full_attribution(self, grounded_corpus):
        provider = StubProvider(["NVIDIA news [1]."])
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        source = result.sources[0]
        for field in ("title", "url", "source", "published_at", "retrieved_at", "bias"):
            assert field in source
        assert source["url"].startswith("http")

    async def test_every_returned_source_is_a_real_stored_article(self, grounded_corpus):
        provider = StubProvider(["NVIDIA news [1][2]."])
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        for source in result.sources:
            assert grounded_corpus.article_repo.get_by_url(source["url"]) is not None

    async def test_hallucinated_citation_is_removed(self, grounded_corpus):
        provider = StubProvider([
            "NVIDIA shipped a GPU [1]. Reuters also reported a merger [8]."
        ])
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        assert "[8]" not in result.answer
        assert 8 not in result.cited
        assert result.warnings

    async def test_uncited_model_output_falls_back_to_extraction(self, grounded_corpus):
        """An answer that cannot be audited against sources is not shown."""
        provider = StubProvider(["NVIDIA definitely acquired Intel this morning."])
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        assert result.engine == "extractive"
        assert "acquired Intel" not in result.answer
        assert any("no usable citations" in w for w in result.warnings)

    async def test_unreachable_model_falls_back_without_fabricating(self, grounded_corpus):
        provider = StubProvider([], available=False)
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        assert result.engine == "extractive"
        assert result.has_results is True
        assert result.sources
        # Everything in the answer must come from the stored articles.
        assert "RTX 5090" in result.answer

    async def test_extractive_output_only_contains_retrieved_text(self, grounded_corpus):
        provider = StubProvider([], available=False)
        result = await service(grounded_corpus, provider).search("What about NVIDIA?")
        stored = " ".join(
            f"{a.title} {a.best_text}" for a in grounded_corpus.article_repo.recent(limit=10)
        )
        for sentence in ("NVIDIA revealed the RTX 5090", "record quarterly revenue"):
            if sentence in result.answer:
                assert sentence in stored

    async def test_reasoning_blocks_are_stripped(self, grounded_corpus):
        provider = StubProvider([
            "<think>Let me consider what to say here.</think>NVIDIA shipped a GPU [1]."
        ])
        # clean_completion runs inside the provider; emulate a provider that does it.
        from daily_brief.llm.base import clean_completion

        assert "<think>" not in clean_completion(provider.responses[0])
        assert "consider what to say" not in clean_completion(provider.responses[0])


# ----------------------------------------------------------- conversation


class TestConversationContext:
    async def test_history_is_persisted(self, grounded_corpus):
        provider = StubProvider(["NVIDIA news [1].", "More NVIDIA news [1]."])
        svc = service(grounded_corpus, provider)
        await svc.search("What about NVIDIA?", session_id="s1")
        await svc.search("Only the biggest ones", session_id="s1")
        history = svc.history("s1")
        assert len(history) == 4  # two user turns, two assistant turns

    async def test_follow_up_retrieves_using_prior_context(self, grounded_corpus):
        provider = StubProvider(["NVIDIA news [1].", "Filtered NVIDIA news [1]."])
        svc = service(grounded_corpus, provider)
        await svc.search("What's happening with NVIDIA?", session_id="s2")
        result = await svc.search("Only show me the biggest stories.", session_id="s2")
        assert result.has_results is True
        assert any("nvidia" in s["title"].lower() for s in result.sources)

    async def test_clearing_history_works(self, grounded_corpus):
        provider = StubProvider(["NVIDIA news [1]."])
        svc = service(grounded_corpus, provider)
        await svc.search("What about NVIDIA?", session_id="s3")
        svc.clear("s3")
        assert svc.history("s3") == []

    async def test_sessions_are_isolated(self, grounded_corpus):
        provider = StubProvider(["A [1].", "B [1]."])
        svc = service(grounded_corpus, provider)
        await svc.search("What about NVIDIA?", session_id="alpha")
        await svc.search("What about NVIDIA?", session_id="beta")
        assert len(svc.history("alpha")) == 2
        assert len(svc.history("beta")) == 2
