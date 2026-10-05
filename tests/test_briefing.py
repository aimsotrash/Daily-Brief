"""Daily briefing: personalization, dedup, grouping, ranking, sectioning."""

from __future__ import annotations

import pytest

from daily_brief.analysis.relevance import parse_interests
from daily_brief.briefing.generator import BriefingGenerator

from .conftest import StubProvider, make_article


def generator(app, provider=None) -> BriefingGenerator:
    return BriefingGenerator(
        app.config, app.article_repo, app.briefing_repo, provider or app.provider
    )


@pytest.fixture
def mixed_corpus(app):
    """Articles across several topics, including a multi-outlet story."""
    app.article_repo.upsert_many([
        make_article(
            "Linux 7.3 released with a new scheduler",
            source_id="linux-weekly", source_name="Linux Weekly",
            summary="The Linux kernel 7.3 release improves scheduling on hybrid CPUs.",
            topics=["linux"], entities=["Linux"], age_hours=3,
        ),
        make_article(
            "GNOME 49 ships with Wayland fixes",
            source_id="linux-weekly", source_name="Linux Weekly",
            summary="The GNOME desktop released version 49 with Wayland improvements.",
            topics=["linux"], entities=["GNOME", "Wayland"], age_hours=6,
        ),
        make_article(
            "OpenAI releases a new model",
            source_id="tech-daily", source_name="Tech Daily",
            summary="OpenAI announced a new large language model for developers.",
            topics=["ai"], entities=["OpenAI"], age_hours=2,
        ),
        make_article(
            "Senate confirms the attorney general",
            source_id="left-post", source_name="The Left Post",
            summary="The Senate confirmed the nominee after debate.",
            topics=["us-politics"], entities=["Senate"], age_hours=4,
            cluster_id="cluster-senate",
        ),
        make_article(
            "Senate confirms attorney general on party lines",
            source_id="right-herald", source_name="The Right Herald",
            summary="Republicans confirmed the nominee over objections.",
            topics=["us-politics"], entities=["Senate"], age_hours=4,
            cluster_id="cluster-senate",
        ),
        make_article(
            "A football transfer story nobody asked for",
            source_id="tech-daily", source_name="Tech Daily",
            summary="A footballer moved clubs for a record fee.",
            topics=["sports"], entities=["Premier League"], age_hours=5,
        ),
    ])
    return app


class TestPersonalization:
    async def test_only_relevant_stories_appear(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("Linux"))
        titles = [
            story["title"]
            for section in payload["sections"]
            for story in section["stories"]
        ]
        assert any("Linux 7.3" in t for t in titles)
        assert not any("football" in t.lower() for t in titles)

    async def test_different_interests_give_different_briefings(self, mixed_corpus):
        gen = generator(mixed_corpus)
        linux = await gen.generate(parse_interests("Linux"), persist=False)
        politics = await gen.generate(parse_interests("US politics"), persist=False)
        assert [s["title"] for s in linux["sections"]] != [
            s["title"] for s in politics["sections"]
        ]

    async def test_sections_are_named_after_interests(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("Linux, AI"))
        titles = {section["title"] for section in payload["sections"]}
        assert titles <= {"Linux & Open Source", "AI & Machine Learning"}

    async def test_stories_land_in_their_best_matching_section(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("AI, Linux"))
        by_section = {
            section["title"]: [s["title"] for s in section["stories"]]
            for section in payload["sections"]
        }
        assert any("OpenAI" in t for t in by_section.get("AI & Machine Learning", []))
        assert any("Linux 7.3" in t for t in by_section.get("Linux & Open Source", []))

    async def test_no_interests_still_produces_a_briefing(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate([])
        assert payload["story_count"] > 0

    async def test_interests_are_recorded_in_the_payload(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("Linux"))
        assert payload["interests"] == ["Linux & Open Source"]


class TestGroupingAndDedup:
    async def test_a_multi_outlet_story_occupies_one_slot(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("US politics"))
        stories = [s for section in payload["sections"] for s in section["stories"]]
        senate = [s for s in stories if "Senate" in s["title"]]
        assert len(senate) == 1
        assert senate[0]["source_count"] == 2
        assert senate[0]["related"]

    async def test_related_coverage_is_preserved_not_discarded(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("US politics"))
        senate = next(
            s for section in payload["sections"] for s in section["stories"]
            if "Senate" in s["title"]
        )
        assert {r["source"] for r in senate["related"]} == {"The Right Herald"} or {
            r["source"] for r in senate["related"]
        } == {"The Left Post"}

    async def test_exact_duplicates_never_appear_twice(self, app):
        app.article_repo.upsert_many([
            make_article("Linux 7.3 released", source_id="linux-weekly",
                         summary="The Linux kernel 7.3 was released.", topics=["linux"]),
        ])
        # Re-upserting the same canonical url must not create a second story.
        app.article_repo.upsert_many([
            make_article("Linux 7.3 released", source_id="linux-weekly",
                         summary="The Linux kernel 7.3 was released.", topics=["linux"]),
        ])
        payload = await generator(app).generate(parse_interests("Linux"))
        titles = [s["title"] for sec in payload["sections"] for s in sec["stories"]]
        assert len(titles) == len(set(titles))

    async def test_multi_outlet_stories_rank_above_single_outlet_ones(self, app):
        app.article_repo.upsert_many([
            make_article("Senate confirms the attorney general", source_id="left-post",
                         summary="The Senate confirmed the nominee.",
                         topics=["us-politics"], age_hours=6, cluster_id="c1"),
            make_article("Senate confirms attorney general on party lines",
                         source_id="right-herald", summary="Republicans confirmed the nominee.",
                         topics=["us-politics"], age_hours=6, cluster_id="c1"),
            make_article("A minor procedural vote in the Senate", source_id="left-post",
                         summary="The Senate held a minor procedural vote.",
                         topics=["us-politics"], age_hours=6),
        ])
        payload = await generator(app).generate(parse_interests("US politics"))
        first = payload["sections"][0]["stories"][0]
        assert first["source_count"] == 2


class TestSectionLimits:
    async def test_respects_max_stories_per_section(self, app):
        app.config.briefing.max_stories_per_section = 2
        app.article_repo.upsert_many([
            make_article(f"Linux story number {i}", source_id="linux-weekly",
                         summary=f"A Linux kernel development story number {i}.",
                         topics=["linux"], age_hours=i + 1)
            for i in range(8)
        ])
        payload = await generator(app).generate(parse_interests("Linux"))
        assert all(len(s["stories"]) <= 2 for s in payload["sections"])

    async def test_respects_max_total_stories(self, app):
        app.config.briefing.max_total_stories = 3
        app.config.briefing.max_stories_per_section = 10
        app.article_repo.upsert_many([
            make_article(f"Linux story number {i}", source_id="linux-weekly",
                         summary=f"A Linux kernel story number {i}.",
                         topics=["linux"], age_hours=i + 1)
            for i in range(10)
        ])
        payload = await generator(app).generate(parse_interests("Linux"))
        assert payload["story_count"] <= 3

    async def test_one_outlet_cannot_fill_a_section(self, app):
        app.config.briefing.max_stories_per_section = 5
        app.article_repo.upsert_many(
            [
                make_article(f"Linux story from one outlet {i}", source_id="linux-weekly",
                             source_name="Linux Weekly",
                             summary=f"A Linux kernel story {i}.", topics=["linux"],
                             age_hours=i + 1)
                for i in range(8)
            ]
            + [
                make_article("Linux story from another outlet", source_id="tech-daily",
                             source_name="Tech Daily",
                             summary="A Linux kernel story from elsewhere.",
                             topics=["linux"], age_hours=2)
            ]
        )
        payload = await generator(app).generate(parse_interests("Linux"))
        section = payload["sections"][0]
        sources = [s["source"] for s in section["stories"]]
        assert len(set(sources)) >= 2


class TestEmptyAndFailureCases:
    async def test_no_articles_yields_an_explained_empty_briefing(self, app):
        payload = await generator(app).generate(parse_interests("Linux"))
        assert payload["sections"] == []
        assert payload["story_count"] == 0
        assert "no articles" in payload["empty_reason"]

    async def test_no_matching_articles_is_explained(self, app):
        app.article_repo.upsert_many([
            make_article("A football transfer story", source_id="tech-daily",
                         summary="A footballer moved clubs.", topics=["sports"])
        ])
        payload = await generator(app).generate(parse_interests("Linux"))
        assert payload["sections"] == []
        assert "matched your interests" in payload["empty_reason"]

    async def test_generation_survives_a_dead_model(self, mixed_corpus):
        provider = StubProvider([], available=False)
        payload = await generator(mixed_corpus, provider).generate(parse_interests("Linux"))
        assert payload["story_count"] > 0
        assert payload["engine"] == "extractive"

    async def test_stories_always_carry_a_summary(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("Linux, AI"))
        for section in payload["sections"]:
            for story in section["stories"]:
                assert story["summary"]

    async def test_summaries_come_from_article_text(self, mixed_corpus):
        payload = await generator(mixed_corpus).generate(parse_interests("AI"))
        story = payload["sections"][0]["stories"][0]
        article = mixed_corpus.article_repo.get_by_url(story["url"])
        assert article is not None
        # Extractive summaries are literal spans of the stored article.
        assert story["summary"][:40] in f"{article.title}. {article.best_text}"


class TestPersistenceAndStaleness:
    async def test_briefing_is_persisted_and_reloadable(self, mixed_corpus):
        gen = generator(mixed_corpus)
        await gen.generate(parse_interests("Linux"))
        latest = gen.latest()
        assert latest is not None
        assert latest["story_count"] > 0

    async def test_latest_returns_the_most_recent(self, mixed_corpus):
        gen = generator(mixed_corpus)
        await gen.generate(parse_interests("Linux"))
        await gen.generate(parse_interests("AI"))
        assert gen.latest()["interests"] == ["AI & Machine Learning"]

    def test_missing_briefing_is_stale(self, app):
        assert generator(app).is_stale(None) is True

    async def test_fresh_briefing_is_not_stale(self, mixed_corpus):
        gen = generator(mixed_corpus)
        payload = await gen.generate(parse_interests("Linux"))
        assert gen.is_stale(payload) is False

    def test_malformed_timestamp_counts_as_stale(self, app):
        assert generator(app).is_stale({"generated_at": "nonsense"}) is True


class TestLLMSummaries:
    async def test_cited_model_lines_replace_extractive_summaries(self, mixed_corpus):
        provider = StubProvider([
            "[1] The Linux 7.3 kernel landed with a reworked scheduler for hybrid CPUs.\n"
            "[2] GNOME 49 shipped a batch of Wayland fixes."
        ])
        payload = await generator(mixed_corpus, provider).generate(parse_interests("Linux"))
        summaries = [s["summary"] for s in payload["sections"][0]["stories"]]
        assert any("reworked scheduler" in s for s in summaries)

    async def test_lines_citing_unknown_articles_are_discarded(self, mixed_corpus):
        provider = StubProvider([
            "[9] An invented story about something that was never retrieved."
        ])
        payload = await generator(mixed_corpus, provider).generate(parse_interests("Linux"))
        for story in payload["sections"][0]["stories"]:
            assert "invented story" not in story["summary"]

    async def test_uncited_model_lines_are_discarded(self, mixed_corpus):
        provider = StubProvider(["Something confident but entirely uncited."])
        payload = await generator(mixed_corpus, provider).generate(parse_interests("Linux"))
        for story in payload["sections"][0]["stories"]:
            assert "entirely uncited" not in story["summary"]
