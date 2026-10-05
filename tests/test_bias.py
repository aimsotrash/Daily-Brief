"""Bias and source-context analysis.

The product requirement is not "be right about politics" -- it is to produce a
transparent, article-level, clearly-labelled classification that does not simply
inherit the publication's label, and to expose the evidence behind it.
"""

from __future__ import annotations

from daily_brief.analysis.bias import analyse_article, compare_coverage
from daily_brief.models import Confidence, Lean, Source

from .conftest import make_article


def src(source_id: str, lean: str, confidence: str = "high", **kwargs) -> Source:
    return Source(
        id=source_id,
        name=kwargs.pop("name", source_id.replace("-", " ").title()),
        url=f"https://{source_id}.test/feed",
        lean=Lean.parse(lean),
        confidence=Confidence.parse(confidence),
        **kwargs,
    )


class TestSourceModel:
    def test_lean_scores_are_ordered(self):
        assert Lean.LEFT.score < Lean.CENTER_LEFT.score < Lean.CENTER.score
        assert Lean.CENTER.score < Lean.CENTER_RIGHT.score < Lean.RIGHT.score

    def test_non_political_leans_are_neutral(self):
        assert Lean.NOT_APPLICABLE.score == 0.0
        assert Lean.NOT_APPLICABLE.is_political is False
        assert Lean.STATE_AFFILIATED.is_political is False

    def test_unknown_lean_parses_safely(self):
        assert Lean.parse("nonsense") is Lean.UNKNOWN
        assert Lean.parse(None) is Lean.UNKNOWN

    def test_source_context_is_serialisable(self):
        context = src("left-post", "left").as_context()
        assert context["lean"] == "left"
        assert "lean_confidence" in context


class TestArticleLevelAnalysis:
    def test_non_political_article_is_marked_not_applicable(self):
        article = make_article(
            "Linux 7.3 improves scheduling on hybrid CPUs",
            summary="The kernel gained cluster-aware load balancing for hybrid CPUs.",
            topics=["linux"],
        )
        bias = analyse_article(article, src("linux-weekly", "not-applicable"))
        assert bias.lean is Lean.NOT_APPLICABLE
        assert "does not appear to cover a politically contested subject" in bias.rationale

    def test_political_article_inherits_a_prior_but_is_re_evaluated(self):
        article = make_article(
            "Senate passes the spending bill",
            summary="The Senate passed a spending bill. Republicans and Democrats "
                    "disagreed over the final text, according to a statement.",
            topics=["us-politics"],
        )
        bias = analyse_article(article, src("left-post", "left"))
        assert bias.lean.is_political
        assert bias.source_lean is Lean.LEFT
        assert any("Source prior" in s for s in bias.signals)

    def test_article_level_result_can_differ_from_the_source_prior(self):
        """A publication label must not be applied blindly to every article."""
        article = make_article(
            "Border crisis: radical left agenda blamed for the invasion",
            summary="Critics say the radical left agenda and open borders caused the "
                    "border crisis, calling it an invasion driven by big-government overreach.",
            topics=["us-politics"],
        )
        bias = analyse_article(article, src("left-post", "left"))
        # Right-coded framing vocabulary must pull it away from the left prior.
        assert bias.lean_score > Lean.LEFT.score

    def test_opinion_url_is_detected(self):
        article = make_article(
            "Why the Senate got this wrong",
            url="https://left-post.test/opinion/senate-wrong",
            summary="I think the Senate should clearly have acted differently.",
            topics=["us-politics"],
        )
        bias = analyse_article(article, src("left-post", "left"))
        assert bias.is_opinion is True
        assert "opinion" in bias.framing

    def test_opinion_title_prefix_is_detected(self):
        article = make_article(
            "Opinion: the Senate got this wrong",
            summary="The Senate acted wrongly.", topics=["us-politics"],
        )
        assert analyse_article(article, src("left-post", "left")).is_opinion is True

    def test_loaded_language_raises_subjectivity(self):
        plain = make_article(
            "Senate passes spending bill",
            summary="The Senate passed the bill, according to a statement from the clerk.",
            topics=["us-politics"],
        )
        charged = make_article(
            "Senate's shocking, disgraceful betrayal slams the nation into chaos",
            summary="In an outrageous and devastating move, the reckless Senate "
                    "slammed the country into chaos and turmoil.",
            topics=["us-politics"],
        )
        source = src("left-post", "left")
        assert (
            analyse_article(charged, source).subjectivity
            > analyse_article(plain, source).subjectivity
        )

    def test_charged_headline_is_flagged(self):
        article = make_article(
            "Senator slams the shocking and outrageous bill",
            summary="A senator criticised the bill.", topics=["us-politics"],
        )
        assert "charged-headline" in analyse_article(article, src("left-post", "left")).framing

    def test_question_headline_is_flagged(self):
        article = make_article(
            "Is the Senate hiding something from voters?",
            summary="Some critics say the Senate is withholding information.",
            topics=["us-politics"],
        )
        assert "question-headline" in analyse_article(article, src("left-post", "left")).framing

    def test_anonymous_sourcing_is_flagged(self):
        article = make_article(
            "Senate deal near",
            summary="Sources say a deal is close, and people familiar with the talks "
                    "reportedly expect a vote.",
            topics=["us-politics"],
        )
        bias = analyse_article(article, src("left-post", "left"))
        assert "anonymous-sourcing" in bias.framing

    def test_vendor_source_is_flagged_as_first_party(self):
        article = make_article(
            "Our new GPU is the fastest ever",
            summary="We announced our fastest GPU today.", topics=["hardware"],
        )
        bias = analyse_article(article, src("vendor-blog", "not-applicable", source_type="vendor"))
        assert "first-party" in bias.framing

    def test_aggregator_is_flagged(self):
        article = make_article("Some link", summary="A link to elsewhere.", topics=["technology"])
        bias = analyse_article(
            article, src("aggregator", "not-applicable", source_type="aggregator")
        )
        assert "aggregator" in bias.framing

    def test_state_affiliation_is_flagged(self):
        article = make_article(
            "Government announces new policy",
            summary="The government announced a policy on foreign relations.",
            topics=["geopolitics"],
        )
        bias = analyse_article(article, src("state-tv", "state-affiliated"))
        assert "state-affiliated" in bias.framing

    def test_analysis_always_carries_evidence(self):
        article = make_article("Anything at all", summary="Some text.", topics=["technology"])
        bias = analyse_article(article, src("tech-daily", "center"))
        assert bias.signals
        assert bias.method == "heuristic"

    def test_serialisation_includes_the_disclaimer(self):
        article = make_article("Anything", summary="Text.", topics=["technology"])
        payload = analyse_article(article, src("tech-daily", "center")).to_dict()
        assert "not a statement of fact" in payload["disclaimer"]
        assert payload["method"] == "heuristic"

    def test_handles_an_article_with_no_source(self):
        bias = analyse_article(make_article("Orphan", summary="Text."), None)
        assert bias.source_lean is Lean.UNKNOWN


class TestCoverageComparison:
    def _spread(self):
        left = make_article(
            "Senate advances landmark protections for workers",
            source_id="left-post", source_name="The Left Post",
            summary="The Senate advanced protections that unions and workers rights "
                    "groups called a win against corporate greed.",
            topics=["us-politics"],
        )
        right = make_article(
            "Senate pushes big-government overreach onto job creators",
            source_id="right-herald", source_name="The Right Herald",
            summary="The Senate advanced a bill that critics call big-government "
                    "overreach harming job creators and taxpayer-funded programs.",
            topics=["us-politics"],
        )
        left.bias = analyse_article(left, src("left-post", "left"))
        right.bias = analyse_article(right, src("right-herald", "right"))
        return [left, right]

    def test_buckets_by_lean(self):
        comparison = compare_coverage(self._spread())
        assert comparison["article_count"] == 2
        assert comparison["source_count"] == 2
        assert len(comparison["buckets"]) >= 1

    def test_reports_measurable_wording_differences(self):
        comparison = compare_coverage(self._spread())
        distinct = comparison["distinct_left_terms"] + comparison["distinct_right_terms"]
        assert distinct  # the two headlines genuinely use different vocabulary

    def test_carries_the_disclaimer(self):
        assert "not statements of fact" in compare_coverage(self._spread())["disclaimer"]

    def test_single_article_does_not_crash(self):
        comparison = compare_coverage(self._spread()[:1])
        assert comparison["article_count"] == 1
        assert comparison["lean_spread"] == 0.0

    def test_empty_input(self):
        comparison = compare_coverage([])
        assert comparison["article_count"] == 0
        assert comparison["buckets"] == []

    def test_counts_opinion_pieces(self):
        articles = self._spread()
        articles[0].bias.is_opinion = True
        assert compare_coverage(articles)["opinion_count"] == 1
