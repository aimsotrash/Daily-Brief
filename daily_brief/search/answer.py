"""Grounded answer generation.

The invariant this module enforces: **an answer about current events is only ever
produced from articles that were actually retrieved.**

Three mechanisms, not one:

1. *Structural* -- if retrieval returns nothing, the model is never called at
   all. A fixed "no relevant results" response is returned instead, so there is
   no code path in which the model is asked about news with an empty context.
2. *Prompted* -- the system prompt states the constraint and requires ``[n]``
   citations (see :mod:`daily_brief.llm.prompts`).
3. *Verified* -- every citation the model emits is checked against the articles
   that were supplied. Out-of-range citations are stripped, and an answer with
   no valid citation at all is rejected in favour of the extractive engine.

The extractive engine is the floor: it only ever reproduces sentences that exist
in retrieved articles, so it cannot fabricate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from ..analysis import bias as bias_analysis
from ..analysis.summarize import summarize_article, summarize_group
from ..llm.base import ChatMessage, LLMProvider, LLMUnavailable
from ..llm.prompts import build_coverage_prompt, build_search_prompt
from ..models import Article, ScoredArticle, StoryGroup
from ..text import truncate

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")

NO_RESULTS_ANSWER = (
    "I couldn't find any articles about that in the news currently indexed, so I "
    "don't have anything to report. I only answer from articles that have actually "
    "been retrieved from your configured sources — I won't guess.\n\n"
    "Things that might help:\n"
    "- Try different or broader wording, or name the organisation or place directly.\n"
    "- Widen the time frame — the story may be older than the window I searched.\n"
    "- Refresh the feeds if it's a story that broke in the last few minutes.\n"
    "- The topic may not be covered by any of your configured sources."
)

EMPTY_QUERY_ANSWER = (
    "Ask me something about the news — for example a company, a place, a topic, or "
    "what happened today in an area you follow."
)


@dataclass
class AnswerResult:
    """A generated answer plus everything needed to audit it."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    engine: str = "extractive"
    grounded: bool = True
    has_results: bool = True
    intent: str = "search"
    query: dict = field(default_factory=dict)
    comparison: dict | None = None
    warnings: list[str] = field(default_factory=list)
    #: Article numbers actually cited by the generated text.
    cited: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "engine": self.engine,
            "grounded": self.grounded,
            "has_results": self.has_results,
            "intent": self.intent,
            "query": self.query,
            "comparison": self.comparison,
            "warnings": self.warnings,
            "cited": self.cited,
        }


#: How many outlets a coverage list will enumerate. Above this the UI reports
#: the true total and lists the most recent members.
MAX_COVERAGE_ENTRIES = 12


def coverage_entry(article: Article, *, is_lead: bool = False) -> dict:
    """One outlet's line in a story's coverage list.

    Compact on purpose: enough to render "The Guardian [CENTER-LEFT]" as a
    clickable row, without shipping a second copy of the whole article.
    """
    return {
        "title": article.title,
        "source": article.source_name,
        "source_id": article.source_id,
        "url": article.display_url,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "lean_badge": bias_analysis.lean_badge(article),
        "is_lead": is_lead,
    }


def build_coverage(lead: Article, related: Sequence[Article]) -> list[dict]:
    """The coverage list for a story: the lead outlet first, then the rest.

    One row per *outlet*, not per article. An outlet often runs two pieces on
    the same story (a live blog and a write-up, say); listing both would make
    the list disagree with the "covered by N sources" count above it. The
    outlet's first article -- the lead, else the most recent -- represents it.
    """
    entries = [coverage_entry(lead, is_lead=True)]
    seen_sources = {lead.source_id}
    for article in related:
        if article.source_id in seen_sources:
            continue
        seen_sources.add(article.source_id)
        entries.append(coverage_entry(article))
    return entries[:MAX_COVERAGE_ENTRIES]


def article_to_source(article: Article, index: int, *, score: float | None = None) -> dict:
    """Serialise an article as a citable source, with all retained metadata."""
    bias = article.bias
    return {
        "n": index,
        "title": article.title,
        "url": article.display_url,
        "source": article.source_name,
        "source_id": article.source_id,
        "author": article.author,
        "published_at": article.published_at.isoformat() if article.published_at else None,
        "retrieved_at": article.retrieved_at.isoformat() if article.retrieved_at else None,
        "summary": truncate(article.summary or article.content, 320),
        "topics": article.topics,
        "entities": article.entities[:8],
        "image_url": article.image_url,
        "cluster_id": article.cluster_id,
        "bias": bias.to_dict() if bias else None,
        "lean_badge": bias_analysis.lean_badge(article),
        "score": round(score, 4) if score is not None else None,
    }


def validate_citations(text: str, article_count: int) -> tuple[str, list[int], list[str]]:
    """Strip citations that do not refer to a supplied article.

    Returns ``(cleaned_text, valid_citation_numbers, warnings)``. A model that
    cites ``[9]`` when it was given five articles is inventing a source; that
    reference is removed rather than shown to the user.
    """
    warnings: list[str] = []
    invalid: set[int] = set()
    valid: list[int] = []

    for match in CITATION_RE.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= article_count:
            if number not in valid:
                valid.append(number)
        else:
            invalid.add(number)

    cleaned = text
    if invalid:
        warnings.append(
            "Removed "
            + ", ".join(f"[{n}]" for n in sorted(invalid))
            + " — the model cited sources it was not given."
        )
        cleaned = CITATION_RE.sub(
            lambda m: "" if int(m.group(1)) in invalid else m.group(0), cleaned
        )
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)

    return cleaned.strip(), sorted(valid), warnings


# ---------------------------------------------------------------------------
# Extractive engine -- the fallback that cannot fabricate
# ---------------------------------------------------------------------------


def extractive_answer(
    question: str, groups: Sequence[StoryGroup], *, now: datetime | None = None
) -> tuple[str, list[int]]:
    """Build an answer purely by selecting sentences from retrieved articles."""
    now = now or datetime.now(timezone.utc)
    if not groups:
        return (NO_RESULTS_ANSWER, [])

    lead = groups[0]
    header = (
        f"Here's what {len(groups)} retrieved "
        f"{'story' if len(groups) == 1 else 'stories'} say. "
        "It is assembled directly from the article text, so it reads as extracts "
        "rather than prose."
    )

    lines: list[str] = [header, ""]
    cited: list[int] = []
    number = 1

    for group in groups:
        article = group.lead
        when = (
            article.published_at.strftime("%d %b %H:%M UTC")
            if article.published_at
            else "date not supplied"
        )
        if group.related:
            body = summarize_group(
                [article, *group.related], max_sentences=2, query=question
            )
        else:
            body = summarize_article(article, max_sentences=2, query=question)

        also = ""
        if group.related:
            others = sorted({a.source_name for a in group.related})[:4]
            if others:
                also = f" Also covered by {', '.join(others)}."

        lines.append(f"**{article.title}** [{number}]")
        lines.append(f"{article.source_name} · {when}")
        lines.append(body or truncate(article.summary, 300))
        if also:
            lines.append(also.strip())
        lines.append("")
        cited.append(number)
        number += 1

    return ("\n".join(lines).strip(), cited)


def extractive_coverage_answer(
    articles: Sequence[Article], comparison: dict
) -> tuple[str, list[int]]:
    """Coverage comparison without a model: report only what is measurable."""
    if not articles:
        return (NO_RESULTS_ANSWER, [])

    lines = [
        f"{comparison['article_count']} article(s) from "
        f"{comparison['source_count']} outlet(s) cover this. Grouped by "
        "Daily-Brief's own lean classification — an analytical label, not a fact:",
        "",
    ]
    index_by_url = {a.display_url: i for i, a in enumerate(articles, start=1)}
    cited: list[int] = []

    for bucket in comparison.get("buckets", []):
        label = bucket["lean"].replace("-", " ")
        lines.append(f"**Classified {label}**")
        for entry in bucket["articles"]:
            n = index_by_url.get(entry["url"])
            marker = f" [{n}]" if n else ""
            if n:
                cited.append(n)
            flags = []
            if entry.get("is_opinion"):
                flags.append("opinion")
            flags.extend(entry.get("framing", [])[:2])
            suffix = f" ({', '.join(dict.fromkeys(flags))})" if flags else ""
            lines.append(f"- {entry['source']}: “{entry['title']}”{suffix}{marker}")
        lines.append("")

    left_terms = comparison.get("distinct_left_terms") or []
    right_terms = comparison.get("distinct_right_terms") or []
    if left_terms or right_terms:
        lines.append("**Wording that differs between the two groups**")
        if left_terms:
            lines.append(f"- Only in left-classified headlines: {', '.join(left_terms[:8])}")
        if right_terms:
            lines.append(f"- Only in right-classified headlines: {', '.join(right_terms[:8])}")
        lines.append("")

    if comparison.get("source_count", 0) < 3:
        lines.append(
            "Note: this is a small sample, so differences here may reflect the "
            "individual articles rather than the outlets."
        )
    elif comparison.get("lean_spread", 0.0) < 0.3:
        lines.append(
            "Note: the retrieved coverage sits in a narrow band of the lean scale, "
            "so this is not a broad cross-section."
        )

    return ("\n".join(lines).strip(), sorted(set(cited)))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class AnswerGenerator:
    def __init__(self, provider: LLMProvider, *, fallback_to_extractive: bool = True) -> None:
        self.provider = provider
        self.fallback_to_extractive = fallback_to_extractive

    async def generate(
        self,
        question: str,
        scored: Sequence[ScoredArticle],
        groups: Sequence[StoryGroup],
        *,
        intent: str = "search",
        history: Sequence[dict] | None = None,
        query_info: dict | None = None,
        now: datetime | None = None,
    ) -> AnswerResult:
        now = now or datetime.now(timezone.utc)

        # (1) Structural guard: no retrieval hits means no generation at all.
        if not scored:
            return AnswerResult(
                answer=NO_RESULTS_ANSWER,
                sources=[],
                engine="none",
                grounded=True,
                has_results=False,
                intent=intent,
                query=query_info or {},
            )

        if intent == "compare_coverage":
            return await self._generate_coverage(question, scored, now, query_info)

        context_articles = [group.lead for group in groups]
        sources = [
            article_to_source(group.lead, i, score=group.score)
            for i, group in enumerate(groups, start=1)
        ]
        # Attach sibling coverage so the UI can list every outlet running the
        # story, each with its own framing label.
        for source_entry, group in zip(sources, groups):
            source_entry["source_count"] = group.source_count
            source_entry["coverage"] = build_coverage(group.lead, group.related)
            source_entry["related"] = source_entry["coverage"][1:]

        warnings: list[str] = []
        if await self.provider.available():
            try:
                messages = [
                    ChatMessage(**m)
                    for m in build_search_prompt(
                        question, context_articles, history=history, now=now
                    )
                ]
                raw = await self.provider.complete(messages)
                cleaned, cited, citation_warnings = validate_citations(
                    raw, len(context_articles)
                )
                warnings.extend(citation_warnings)

                if not cited:
                    # An uncited answer cannot be audited against the sources.
                    warnings.append(
                        "The model produced no usable citations, so Daily-Brief fell "
                        "back to extracting directly from the retrieved articles."
                    )
                    log.warning("model answer had no valid citations; using extractive")
                elif cleaned:
                    return AnswerResult(
                        answer=cleaned,
                        sources=sources,
                        engine=f"llm:{self.provider.model}",
                        grounded=True,
                        intent=intent,
                        query=query_info or {},
                        warnings=warnings,
                        cited=cited,
                    )
            except LLMUnavailable as exc:
                log.info("LLM unavailable, using extractive engine: %s", exc)
                warnings.append(
                    f"The configured model was unreachable ({exc}). Showing extracts "
                    "from the retrieved articles instead."
                )
                if not self.fallback_to_extractive:
                    raise
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("answer generation failed: %s", exc)
                warnings.append("Answer generation failed; showing article extracts.")

        answer, cited = extractive_answer(question, groups, now=now)
        return AnswerResult(
            answer=answer,
            sources=sources,
            engine="extractive",
            grounded=True,
            intent=intent,
            query=query_info or {},
            warnings=warnings,
            cited=cited,
        )

    async def _generate_coverage(
        self,
        question: str,
        scored: Sequence[ScoredArticle],
        now: datetime,
        query_info: dict | None,
    ) -> AnswerResult:
        articles = [item.article for item in scored]
        comparison = bias_analysis.compare_coverage(articles)
        sources = [
            article_to_source(a, i, score=s.score)
            for i, (a, s) in enumerate(zip(articles, scored), start=1)
        ]
        warnings: list[str] = []

        if await self.provider.available():
            try:
                messages = [
                    ChatMessage(**m)
                    for m in build_coverage_prompt(question, articles, comparison, now=now)
                ]
                raw = await self.provider.complete(messages)
                cleaned, cited, citation_warnings = validate_citations(raw, len(articles))
                warnings.extend(citation_warnings)
                if cited and cleaned:
                    return AnswerResult(
                        answer=cleaned,
                        sources=sources,
                        engine=f"llm:{self.provider.model}",
                        grounded=True,
                        intent="compare_coverage",
                        query=query_info or {},
                        comparison=comparison,
                        warnings=warnings,
                        cited=cited,
                    )
                warnings.append(
                    "The model produced no usable citations; showing the measured "
                    "comparison instead."
                )
            except LLMUnavailable as exc:
                log.info("LLM unavailable for coverage comparison: %s", exc)
                warnings.append(f"The configured model was unreachable ({exc}).")
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("coverage generation failed: %s", exc)

        answer, cited = extractive_coverage_answer(articles, comparison)
        return AnswerResult(
            answer=answer,
            sources=sources,
            engine="extractive",
            grounded=True,
            intent="compare_coverage",
            query=query_info or {},
            comparison=comparison,
            warnings=warnings,
            cited=cited,
        )
