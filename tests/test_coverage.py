"""Coverage lists and the framing labels shown against each outlet.

The UI shows, per story, every outlet running it with that outlet's
classification. These tests pin the two things that are easy to get subtly
wrong: the count must match reality, and a publication-level prior must never be
presented as though this specific article had been analysed.
"""

from __future__ import annotations

import pytest

from daily_brief.analysis.bias import analyse_article, lean_badge
from daily_brief.analysis.relevance import parse_interests
from daily_brief.briefing.generator import BriefingGenerator
from daily_brief.models import Article, BiasAssessment, Confidence, Lean, Source
from daily_brief.search.answer import MAX_COVERAGE_ENTRIES, build_coverage, coverage_entry
from daily_brief.search.chat import NewsSearchService
from daily_brief.search.retriever import Retriever

from .conftest import StubProvider, make_article


def src(source_id: str, lean: str, confidence: str = "high", **kwargs) -> Source:
    return Source(
        id=source_id,
        name=kwargs.pop("name", source_id.replace("-", " ").title()),
        url=f"https://{source_id}.test/feed",
        lean=Lean.parse(lean),
        confidence=Confidence.parse(confidence),
        **kwargs,
    )


class TestLeanBadge:
    def test_article_level_classification_is_marked_as_such(self):
        article = make_article(
            "Senate passes the spending bill",
            summary="The Senate passed a spending bill. Republicans and Democrats "
                    "disagreed over the text, according to a statement.",
            topics=["us-politics"],
        )
        article.bias = analyse_article(article, src("left-post", "left"))
        badge = lean_badge(article)
        assert badge["basis"] == "article"
        assert badge["lean"] in {"left", "center-left", "center", "center-right", "right"}

    def test_non_political_article_reports_not_applicable_not_unknown(self):
        """"No political framing here" is a finding, so it is article-level."""
        article = make_article(
            "Linux 7.3 improves scheduling on hybrid CPUs",
            summary="The kernel gained cluster-aware load balancing.",
            topics=["linux"],
        )
        article.bias = analyse_article(article, src("linux-weekly", "not-applicable"))
        badge = lean_badge(article)
        assert badge["lean"] == "not-applicable"
        assert badge["basis"] == "article"

    def test_falls_back_to_the_source_prior_and_says_so(self):
        article = make_article("Something unclassified")
        article.bias = BiasAssessment(lean=Lean.UNKNOWN, source_lean=Lean.RIGHT)
        badge = lean_badge(article)
        assert badge["lean"] == "right"
        assert badge["basis"] == "source"

    def test_source_prior_is_never_labelled_article_level(self):
        """The distinction §6 requires: a prior must not masquerade as analysis."""
        article = make_article("Something unclassified")
        article.bias = BiasAssessment(lean=Lean.UNKNOWN, source_lean=Lean.LEFT)
        assert lean_badge(article)["basis"] != "article"

    def test_nothing_known_yields_unknown(self):
        article = make_article("No analysis at all")
        article.bias = None
        badge = lean_badge(article)
        assert badge["lean"] == "unknown"
        assert badge["basis"] == "none"

    def test_unknown_article_and_unknown_source_stays_unknown(self):
        article = make_article("Neither known")
        article.bias = BiasAssessment(lean=Lean.UNKNOWN, source_lean=Lean.UNKNOWN)
        assert lean_badge(article)["basis"] == "none"

    def test_source_prior_of_not_applicable_is_not_promoted(self):
        article = make_article("Tech thing")
        article.bias = BiasAssessment(lean=Lean.UNKNOWN, source_lean=Lean.NOT_APPLICABLE)
        assert lean_badge(article)["basis"] == "none"

    def test_state_affiliation_is_surfaced(self):
        article = make_article("Government statement", topics=["geopolitics"])
        article.bias = BiasAssessment(
            lean=Lean.CENTER, source_lean=Lean.STATE_AFFILIATED
        )
        assert lean_badge(article)["state_affiliated"] is True

    def test_opinion_flag_is_carried(self):
        article = make_article("Why the Senate got it wrong", topics=["us-politics"])
        article.bias = BiasAssessment(lean=Lean.LEFT, is_opinion=True)
        assert lean_badge(article)["is_opinion"] is True


class TestCoverageEntry:
    def test_entry_has_what_a_coverage_row_needs(self):
        article = make_article("A story", source_id="left-post", source_name="The Left Post")
        entry = coverage_entry(article)
        assert entry["source"] == "The Left Post"
        assert entry["url"].startswith("http")
        assert entry["title"] == "A story"
        assert "lean_badge" in entry
        assert entry["is_lead"] is False

    def test_lead_is_flagged_and_listed_first(self):
        lead = make_article("Lead story", source_id="left-post")
        other = make_article("Same story elsewhere", source_id="right-herald")
        coverage = build_coverage(lead, [other])
        assert coverage[0]["is_lead"] is True
        assert coverage[0]["source_id"] == "left-post"
        assert coverage[1]["is_lead"] is False

    def test_lead_is_not_duplicated_if_also_in_related(self):
        lead = make_article("Lead story", source_id="left-post")
        coverage = build_coverage(lead, [lead])
        assert len(coverage) == 1

    def test_one_row_per_outlet_even_when_it_ran_two_pieces(self):
        """An outlet running a live blog *and* a write-up is still one source.

        Otherwise the list disagrees with the "covered by N sources" heading.
        """
        lead = make_article("Live updates: the vote", source_id="the-hill",
                            source_name="The Hill", url="https://hill.test/live")
        second = make_article("Senate confirms the nominee", source_id="the-hill",
                              source_name="The Hill", url="https://hill.test/story")
        other = make_article("Senate vote recap", source_id="left-post")
        coverage = build_coverage(lead, [second, other])
        assert len(coverage) == 2
        assert [c["source_id"] for c in coverage] == ["the-hill", "left-post"]

    def test_row_count_equals_distinct_outlet_count(self):
        lead = make_article("Lead", source_id="a")
        related = [make_article(f"r{i}", source_id=sid)
                   for i, sid in enumerate(["a", "b", "b", "c"])]
        coverage = build_coverage(lead, related)
        assert len(coverage) == len({c["source_id"] for c in coverage}) == 3

    def test_every_entry_carries_a_badge(self):
        lead = make_article("Lead", source_id="left-post")
        related = [make_article(f"Other {i}", source_id=f"s{i}") for i in range(3)]
        for entry in build_coverage(lead, related):
            assert "lean" in entry["lean_badge"]
            assert "basis" in entry["lean_badge"]

    def test_coverage_is_capped(self):
        lead = make_article("Lead", source_id="left-post")
        related = [make_article(f"Other {i}", source_id=f"s{i}") for i in range(40)]
        assert len(build_coverage(lead, related)) == MAX_COVERAGE_ENTRIES

    def test_no_related_yields_just_the_lead(self):
        assert len(build_coverage(make_article("Solo"), [])) == 1


@pytest.fixture
def clustered(app):
    """One story run by four outlets across the lean scale, plus an unrelated one."""
    sources = {
        "left-post": src("left-post", "left", name="The Left Post"),
        "right-herald": src("right-herald", "right", name="The Right Herald"),
        "tech-daily": src("tech-daily", "center", name="Tech Daily"),
        "linux-weekly": src("linux-weekly", "not-applicable", name="Linux Weekly"),
    }
    articles = []
    for sid, title in [
        ("left-post", "Senate confirms the attorney general after long debate"),
        ("right-herald", "Senate confirms attorney general on party lines"),
        ("tech-daily", "Senate confirms the attorney general"),
        ("linux-weekly", "Senate confirms attorney general, tech policy in focus"),
    ]:
        a = make_article(
            title, source_id=sid, source_name=sources[sid].name,
            summary="The Senate confirmed the nominee after debate between "
                    "Republicans and Democrats, according to a statement.",
            topics=["us-politics"], entities=["Senate"], cluster_id="cluster-senate",
        )
        a.bias = analyse_article(a, sources[sid])
        articles.append(a)
    articles.append(
        make_article("Unrelated Linux kernel release", source_id="linux-weekly",
                     summary="The Linux kernel was released.", topics=["linux"])
    )
    app.article_repo.upsert_many(articles)
    return app


class TestBriefingCoverage:
    async def test_coverage_lists_every_outlet(self, clustered):
        gen = BriefingGenerator(
            clustered.config, clustered.article_repo,
            clustered.briefing_repo, clustered.provider,
        )
        payload = await gen.generate(parse_interests("US politics"))
        story = next(
            s for section in payload["sections"] for s in section["stories"]
            if "Senate" in s["title"]
        )
        assert story["source_count"] == 4
        assert len(story["coverage"]) == 4

    async def test_count_matches_the_listed_outlets(self, clustered):
        """"Covered by N sources" must not disagree with the list under it.

        Checks the row count, not just the distinct-outlet count: a repeated
        outlet would satisfy the latter while still looking wrong on screen.
        """
        gen = BriefingGenerator(
            clustered.config, clustered.article_repo,
            clustered.briefing_repo, clustered.provider,
        )
        payload = await gen.generate(parse_interests("US politics"))
        for section in payload["sections"]:
            for story in section["stories"]:
                assert len(story["coverage"]) == story["source_count"]
                outlets = [c["source_id"] for c in story["coverage"]]
                assert len(outlets) == len(set(outlets))

    async def test_coverage_survives_interest_filtering(self, clustered):
        """A story's outlet count reflects the cluster, not interest matching.

        Only some members of a cluster clear the interest threshold; the
        coverage list must still show everyone who ran the story.
        """
        gen = BriefingGenerator(
            clustered.config, clustered.article_repo,
            clustered.briefing_repo, clustered.provider,
        )
        payload = await gen.generate(parse_interests("US politics"))
        story = next(
            s for section in payload["sections"] for s in section["stories"]
            if "Senate" in s["title"]
        )
        assert story["source_count"] == 4

    async def test_related_excludes_the_lead(self, clustered):
        gen = BriefingGenerator(
            clustered.config, clustered.article_repo,
            clustered.briefing_repo, clustered.provider,
        )
        payload = await gen.generate(parse_interests("US politics"))
        story = next(
            s for section in payload["sections"] for s in section["stories"]
            if "Senate" in s["title"]
        )
        assert len(story["related"]) == story["source_count"] - 1
        assert story["source_id"] not in {r["source_id"] for r in story["related"]}

    async def test_single_outlet_story_has_a_one_entry_coverage(self, clustered):
        gen = BriefingGenerator(
            clustered.config, clustered.article_repo,
            clustered.briefing_repo, clustered.provider,
        )
        payload = await gen.generate(parse_interests("Linux"))
        story = payload["sections"][0]["stories"][0]
        assert story["source_count"] == 1
        assert len(story["coverage"]) == 1

    async def test_lead_carries_a_top_level_badge(self, clustered):
        gen = BriefingGenerator(
            clustered.config, clustered.article_repo,
            clustered.briefing_repo, clustered.provider,
        )
        payload = await gen.generate(parse_interests("US politics"))
        story = payload["sections"][0]["stories"][0]
        assert "lean_badge" in story
        assert story["lean_badge"]["basis"] in {"article", "source", "none"}


class TestSearchCoverage:
    def _service(self, app, provider) -> NewsSearchService:
        return NewsSearchService(
            app.config, Retriever(app.article_repo, app.config.search),
            provider, app.chat_repo,
        )

    async def test_search_sources_carry_coverage(self, clustered):
        provider = StubProvider(["The Senate confirmed the nominee [1]."])
        result = await self._service(clustered, provider).search("Senate attorney general")
        senate = next(s for s in result.sources if "Senate" in s["title"])
        assert senate["source_count"] == 4
        assert len(senate["coverage"]) == 4
        assert senate["coverage"][0]["is_lead"] is True

    async def test_every_coverage_outlet_has_a_classification(self, clustered):
        provider = StubProvider(["The Senate confirmed the nominee [1]."])
        result = await self._service(clustered, provider).search("Senate attorney general")
        for source in result.sources:
            for entry in source.get("coverage", []):
                badge = entry["lean_badge"]
                assert badge["lean"]
                assert badge["basis"] in {"article", "source", "none"}

    async def test_coverage_spans_the_lean_scale(self, clustered):
        """A left and a right outlet on one story must not collapse to one label."""
        provider = StubProvider(["The Senate confirmed the nominee [1]."])
        result = await self._service(clustered, provider).search("Senate attorney general")
        senate = next(s for s in result.sources if "Senate" in s["title"])
        leans = {c["lean_badge"]["lean"] for c in senate["coverage"]}
        assert len(leans) > 1

    async def test_no_results_has_no_coverage(self, app):
        provider = StubProvider([])
        result = await self._service(app, provider).search("nothing matches this at all")
        assert result.sources == []
