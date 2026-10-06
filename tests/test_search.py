"""Search: query understanding, retrieval, and result selection."""

from __future__ import annotations

import pytest

from daily_brief.analysis.relevance import parse_interests
from daily_brief.search.query import detect_intent, detect_time_window, parse_query
from daily_brief.search.retriever import Retriever, build_match_expression

from .conftest import make_article


@pytest.fixture
def corpus(app):
    """A small, deliberately varied article set with realistic overlap."""
    articles = [
        make_article(
            "NVIDIA announces the RTX 5090 graphics card",
            source_id="tech-daily", source_name="Tech Daily",
            summary="NVIDIA revealed the RTX 5090 today, its fastest consumer GPU yet.",
            topics=["hardware", "nvidia"], entities=["NVIDIA", "RTX"], age_hours=3,
        ),
        make_article(
            "NVIDIA earnings beat expectations on AI demand",
            source_id="left-post", source_name="The Left Post",
            summary="NVIDIA reported record revenue driven by demand for AI accelerators.",
            topics=["business", "nvidia"], entities=["NVIDIA"], age_hours=8,
        ),
        make_article(
            "Linux 7.3 improves scheduling on hybrid CPUs",
            source_id="linux-weekly", source_name="Linux Weekly",
            summary="The Linux kernel gained better cluster load balancing for hybrid CPUs.",
            topics=["linux"], entities=["Linux", "Intel"], age_hours=5,
        ),
        make_article(
            "Senate confirms the new attorney general",
            source_id="left-post", source_name="The Left Post",
            summary="The Senate voted to confirm the nominee after a long debate.",
            topics=["us-politics"], entities=["Senate"], age_hours=4,
            cluster_id="cluster-senate",
        ),
        make_article(
            "Senate confirms attorney general after party-line vote",
            source_id="right-herald", source_name="The Right Herald",
            summary="Republicans pushed the confirmation through over Democrat objections.",
            topics=["us-politics"], entities=["Senate"], age_hours=4,
            cluster_id="cluster-senate",
        ),
        make_article(
            "Old story about NVIDIA from three weeks ago",
            source_id="tech-daily", source_name="Tech Daily",
            summary="An older NVIDIA announcement that is no longer current.",
            topics=["nvidia"], entities=["NVIDIA"], age_hours=24 * 20,
        ),
    ]
    app.article_repo.upsert_many(articles)
    return app


class TestQueryUnderstanding:
    def test_extracts_entities(self):
        parsed = parse_query("What's happening with NVIDIA today?")
        assert "NVIDIA" in parsed.entities

    def test_strips_query_filler_from_keywords(self):
        parsed = parse_query("What's happening with NVIDIA today?")
        assert "today" not in parsed.keywords
        assert "happening" not in parsed.keywords

    def test_empty_query_is_empty(self):
        parsed = parse_query("")
        assert parsed.is_empty()
        assert parsed.all_terms == []

    @pytest.mark.parametrize(
        "question,hours",
        [
            ("What happened today?", 24),
            ("What's happening this week?", 168),
            ("news from the last month", 720),
            ("What's going on with tariffs?", 96),
        ],
    )
    def test_time_windows(self, question, hours):
        assert detect_time_window(question) == hours

    @pytest.mark.parametrize(
        "question,intent",
        [
            ("How are different sources covering this?", "compare_coverage"),
            ("Compare the coverage of this story", "compare_coverage"),
            ("Is the reporting biased?", "compare_coverage"),
            ("Summarise the AI news", "summarize"),
            ("What's happening with NVIDIA?", "search"),
        ],
    )
    def test_intent_detection(self, question, intent):
        assert detect_intent(question)[0] == intent

    def test_follow_up_inherits_prior_subject(self):
        history = [
            {"role": "user", "content": "What's happening with NVIDIA?"},
            {"role": "assistant", "content": "NVIDIA announced new GPUs."},
        ]
        parsed = parse_query("Only show me the biggest stories.", history)
        assert parsed.is_follow_up is True
        assert any("nvidia" in term.lower() for term in parsed.all_terms)

    def test_follow_up_without_history_still_parses(self):
        assert parse_query("Only the biggest ones").intent in {"follow_up", "search"}

    @pytest.mark.parametrize(
        "question",
        [
            "What's new with NVIDIA this week?",
            "Why did NVIDIA stock fall?",
            "Only the biggest ones",
            "How are different sources covering this?",
        ],
    )
    def test_nothing_is_a_follow_up_without_earlier_turns(self, question):
        intent, follow_up = detect_intent(question)
        assert follow_up is False
        assert intent != "follow_up"

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Only the biggest ones", True),
            ("What about Intel?", True),
            ("Why?", True),
            ("Why did they do that?", True),
            ("Tell me more about the story", True),
            ("Why did NVIDIA stock fall?", False),
            ("What's new with NVIDIA this week?", False),
            ("What happened this morning?", False),
        ],
    )
    def test_follow_up_detection_with_earlier_turns(self, question, expected):
        assert detect_intent(question, has_history=True)[1] is expected

    def test_contractions_and_possessives_are_not_keywords(self):
        assert parse_query("What's new in auto-cpufreq?").keywords == ["auto-cpufreq"]
        keywords = parse_query("Why are NVIDIA's earnings up?").keywords
        assert "earnings" in keywords
        assert not any("'" in keyword for keyword in keywords)


class TestMatchExpression:
    def test_anchors_on_entities_when_present(self):
        parsed = parse_query("What's happening with NVIDIA today?")
        parsed.topics = ["hardware"]
        expression = build_match_expression(parsed)
        assert "NVIDIA" in expression
        # Topics must never leak in as free-text terms.
        assert "hardware" not in expression

    def test_falls_back_to_keywords_without_entities(self):
        parsed = parse_query("quarterly earnings and revenue growth")
        assert build_match_expression(parsed)

    def test_broad_mode_includes_keywords(self):
        parsed = parse_query("What is NVIDIA doing about tariffs?")
        broad = build_match_expression(parsed, broad=True)
        assert "NVIDIA" in broad and "tariff" in broad

    def test_quotes_are_escaped(self):
        parsed = parse_query('the "quoted" thing')
        parsed.entities = ['a "risky" name']
        # Must not raise and must not emit an unbalanced quote.
        assert build_match_expression(parsed).count('"') % 2 == 0

    def test_empty_query_yields_no_expression(self):
        assert build_match_expression(parse_query("")) == ""


class TestRetrieval:
    def _retriever(self, app) -> Retriever:
        return Retriever(app.article_repo, app.config.search)

    def test_valid_query_returns_relevant_results(self, corpus):
        results = self._retriever(corpus).retrieve(parse_query("What about NVIDIA?"))
        assert results
        assert all("nvidia" in r.article.title.lower() for r in results)

    def test_empty_query_returns_nothing(self, corpus):
        assert self._retriever(corpus).retrieve(parse_query("")) == []

    def test_no_matching_articles_returns_nothing(self, corpus):
        results = self._retriever(corpus).retrieve(
            parse_query("zzqqx flurbulator quantum banana treaty")
        )
        assert results == []

    def test_irrelevant_single_word_overlap_is_rejected(self, corpus):
        """A query must be substantially covered, not share one incidental word."""
        results = self._retriever(corpus).retrieve(
            parse_query("underwater basket weaving championship results")
        )
        assert results == []

    def test_multiple_related_articles_are_all_retrieved(self, corpus):
        results = self._retriever(corpus).retrieve(parse_query("Senate attorney general"))
        assert len(results) >= 2
        assert {r.article.source_id for r in results} >= {"left-post", "right-herald"}

    def test_recent_articles_outrank_old_ones(self, corpus):
        results = self._retriever(corpus).retrieve(parse_query("NVIDIA"))
        titles = [r.article.title for r in results]
        assert "Old story about NVIDIA from three weeks ago" not in titles[:1]

    def test_respects_the_result_limit(self, corpus):
        assert len(self._retriever(corpus).retrieve(parse_query("NVIDIA"), limit=2)) <= 2

    def test_source_diversity_is_enforced(self, app):
        """One prolific outlet must not be able to fill the entire answer."""
        app.article_repo.upsert_many([
            make_article(f"NVIDIA story number {i}", source_id="tech-daily",
                         summary="NVIDIA did something notable.", entities=["NVIDIA"])
            for i in range(12)
        ] + [
            make_article("NVIDIA coverage from another outlet", source_id="left-post",
                         summary="NVIDIA did something notable.", entities=["NVIDIA"])
        ])
        results = Retriever(app.article_repo, app.config.search).retrieve(
            parse_query("NVIDIA"), limit=6
        )
        assert len({r.article.source_id for r in results}) >= 2

    def test_interests_influence_ranking_not_membership(self, corpus):
        retriever = self._retriever(corpus)
        parsed = parse_query("Senate attorney general")
        without = retriever.retrieve(parsed)
        with_interests = retriever.retrieve(parsed, interests=parse_interests("US politics"))
        assert {r.article.title for r in with_interests} == {r.article.title for r in without}

    def test_grouping_folds_a_cluster_into_one_story(self, corpus):
        retriever = self._retriever(corpus)
        results = retriever.retrieve(parse_query("Senate attorney general"))
        groups = retriever.group_into_stories(results)
        senate = [g for g in groups if "Senate" in g.lead.title]
        assert len(senate) == 1
        assert senate[0].related  # the sibling coverage is attached, not dropped

    def test_coverage_for_returns_the_whole_cluster(self, corpus):
        retriever = self._retriever(corpus)
        lead = corpus.article_repo.get_by_url(
            "https://left-post.test/senate-confirms-the-new-attorney-general"
        )
        assert lead is not None
        coverage = retriever.coverage_for(lead)
        assert len(coverage) == 2
        assert len({a.source_id for a in coverage}) == 2

    def test_a_precise_match_is_not_padded_with_name_only_matches(self, corpus):
        corpus.article_repo.upsert_many([
            make_article(
                "Nvidia's Groq deal faces a shareholder lawsuit",
                source_id="right-herald", source_name="The Right Herald",
                summary="Groq stockholders filed a lawsuit over the Nvidia deal.",
                entities=["Nvidia", "Groq"],
            )
        ])
        results = self._retriever(corpus).retrieve(parse_query("Groq lawsuit against Nvidia"))
        assert [r.article.title for r in results] == [
            "Nvidia's Groq deal faces a shareholder lawsuit"
        ]

    def test_an_incidental_shared_word_does_not_pad_the_answer(self, app):
        app.article_repo.upsert_many([
            make_article("Auto-CPUFreq 3.2 adds dynamic boost controls",
                         summary="The auto-cpufreq power manager gained new boost settings."),
            make_article("What's going to happen in the midterms?",
                         summary="What's at stake as voters head to the polls."),
        ])
        results = Retriever(app.article_repo, app.config.search).retrieve(
            parse_query("What's new in auto-cpufreq?")
        )
        assert [r.article.title for r in results] == [
            "Auto-CPUFreq 3.2 adds dynamic boost controls"
        ]

    def test_a_question_about_two_names_keeps_articles_about_either(self, corpus):
        titles = {r.article.title for r in self._retriever(corpus).retrieve(parse_query("NVIDIA and Linux"))}
        assert any("NVIDIA" in title for title in titles)
        assert any("Linux" in title for title in titles)

    def test_widens_the_window_rather_than_returning_nothing(self, app):
        app.article_repo.upsert_many([
            make_article("Obscure topic ferrofluid research breakthrough",
                         summary="Ferrofluid research produced a breakthrough.",
                         entities=["Ferrofluid"], age_hours=24 * 10)
        ])
        parsed = parse_query("What happened with ferrofluid today?")
        assert parsed.time_window_hours == 24
        assert Retriever(app.article_repo, app.config.search).retrieve(parsed)
