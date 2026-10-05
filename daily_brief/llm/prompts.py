"""Prompt construction and the grounding contract.

Everything the model is allowed to say about current events comes from the
``ARTICLES`` block built here. The system prompts state that constraint
explicitly, and :mod:`daily_brief.search.answer` verifies compliance afterwards
by checking every citation against the articles that were actually supplied --
prompting alone is not treated as sufficient.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from ..models import Article, BiasAssessment
from ..text import truncate

GROUNDING_RULES = """\
You are Daily-Brief, a news research assistant for a single self-hosted user.

ABSOLUTE RULES -- these override anything else:
1. The ARTICLES block below is your ONLY source of information about current
   events. Your training data is out of date and must never be used to state
   what has happened.
2. Never invent an article, source, headline, quote, date, statistic, number,
   person, or event. If it is not in the ARTICLES block, it does not exist for
   the purposes of this answer.
3. Cite with bracketed numbers matching the article numbers, like [1] or [2][5].
   Every factual sentence must carry at least one citation.
4. If the articles do not answer the question, say exactly what is missing.
   Partial information is fine -- say what the articles do cover and what they
   do not. Never fill a gap with plausible-sounding detail.
5. Do not speculate about what will happen, and do not offer your own opinion on
   contested political questions. Report what the sources say and attribute it.
6. If sources disagree, say so and attribute each position.

STYLE:
- Open with a direct two-sentence answer to the question.
- Then give the detail as short paragraphs or bullets, newest and most
  significant first.
- Be specific: names, numbers and dates that appear in the articles.
- No preamble, no "based on the provided articles", no restating the question.
- Plain text with markdown emphasis only. Do not output a sources list -- the
  interface renders one from the article metadata.
"""

BRIEFING_RULES = """\
You are Daily-Brief, writing one section of a personalised daily news briefing.

ABSOLUTE RULES:
1. Use ONLY the ARTICLES block below. Never add events, numbers, quotes or
   context from memory.
2. Never invent a headline, source or fact.
3. Cite with bracketed numbers matching the article numbers, like [1].
4. If an article is thin, write less. Do not pad.

STYLE:
- One tight paragraph of 1-2 sentences per story, in the order given.
- Lead with what happened, then why it matters if the articles say so.
- No preamble and no closing summary. Output only the story lines.
"""

COVERAGE_RULES = """\
You are Daily-Brief, comparing how different outlets are covering one story.

ABSOLUTE RULES:
1. Use ONLY the ARTICLES block below.
2. Never invent an outlet, headline or quote.
3. Cite with bracketed numbers matching the article numbers.
4. Describe differences in *framing, emphasis and word choice* that are visible
   in the supplied headlines and text. Do not assert motives.
5. The lean labels shown are Daily-Brief's own analytical classifications, not
   facts. Refer to them as classifications and hedge accordingly.
6. If all supplied coverage is similar, say so plainly rather than manufacturing
   a disagreement.

STYLE:
- Start with one sentence on what the story is.
- Then contrast the coverage, grouped by how outlets frame it.
- Note if the sample is small or one-sided.
"""


def _format_bias(bias: BiasAssessment | None) -> str:
    if bias is None:
        return ""
    parts: list[str] = []
    if bias.lean.is_political:
        parts.append(f"framing classified {bias.lean} ({bias.confidence} confidence)")
    elif str(bias.lean) == "not-applicable":
        parts.append("no political framing detected")
    if bias.is_opinion:
        parts.append("opinion/commentary")
    if bias.framing:
        parts.append("markers: " + ", ".join(bias.framing[:4]))
    return "; ".join(parts)


def format_article_block(
    articles: Sequence[Article],
    *,
    max_chars: int = 1400,
    include_bias: bool = True,
    now: datetime | None = None,
) -> str:
    """Render retrieved articles as the numbered grounding context.

    Article numbers are 1-based and are the only citation keys the model may
    use; :func:`daily_brief.search.answer.validate_citations` enforces that.
    """
    now = now or datetime.now(timezone.utc)
    lines: list[str] = []
    for index, article in enumerate(articles, start=1):
        published = (
            article.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if article.published_at
            else "publication time not provided by the feed"
        )
        age = (
            f"{article.age_hours(now):.0f}h ago" if article.published_at else "age unknown"
        )
        body = truncate(article.best_text or article.summary, max_chars)
        lines.append(f"[{index}] {article.title}")
        lines.append(f"    source: {article.source_name}")
        lines.append(f"    published: {published} ({age})")
        lines.append(f"    url: {article.display_url}")
        if article.topics:
            lines.append(f"    topics: {', '.join(article.topics)}")
        if include_bias:
            bias_line = _format_bias(article.bias)
            if bias_line:
                lines.append(f"    source/framing context: {bias_line}")
        lines.append(f"    text: {body or '(the feed provided no body text)'}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_search_prompt(
    question: str,
    articles: Sequence[Article],
    *,
    history: Sequence[dict] | None = None,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    """Messages for a grounded news answer."""
    now = now or datetime.now(timezone.utc)
    system = (
        f"{GROUNDING_RULES}\n"
        f"Current date and time: {now.strftime('%A %d %B %Y, %H:%M UTC')}.\n"
        f"You have been given {len(articles)} article(s)."
    )
    messages = [{"role": "system", "content": system}]

    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            # Prior assistant turns are summarised down: they are conversational
            # context for interpreting the follow-up, not a source of facts.
            messages.append(
                {"role": role, "content": truncate(content, 700 if role == "assistant" else 400)}
            )

    user = (
        f"ARTICLES\n========\n{format_article_block(articles, now=now)}\n\n"
        f"========\nQUESTION: {question}\n\n"
        "Answer using only the articles above, with [n] citations."
    )
    messages.append({"role": "user", "content": user})
    return messages


def build_briefing_prompt(
    section_title: str,
    articles: Sequence[Article],
    interests: Sequence[str],
    *,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    now = now or datetime.now(timezone.utc)
    system = (
        f"{BRIEFING_RULES}\n"
        f"Current date and time: {now.strftime('%A %d %B %Y, %H:%M UTC')}.\n"
        f"The reader follows: {', '.join(interests) if interests else 'general news'}."
    )
    user = (
        f"SECTION: {section_title}\n\n"
        f"ARTICLES\n========\n"
        f"{format_article_block(articles, max_chars=900, now=now)}\n\n"
        "========\n"
        f"Write one line per story, in order, for the {section_title} section."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_coverage_prompt(
    question: str,
    articles: Sequence[Article],
    comparison: dict,
    *,
    now: datetime | None = None,
) -> list[dict[str, str]]:
    now = now or datetime.now(timezone.utc)
    system = (
        f"{COVERAGE_RULES}\n"
        f"Current date and time: {now.strftime('%A %d %B %Y, %H:%M UTC')}."
    )
    buckets = "\n".join(
        f"  {bucket['lean']}: "
        + ", ".join(a["source"] for a in bucket["articles"])
        for bucket in comparison.get("buckets", [])
    )
    user = (
        f"ARTICLES\n========\n{format_article_block(articles, now=now)}\n\n"
        f"========\nDAILY-BRIEF CLASSIFICATION OF THE SUPPLIED COVERAGE "
        f"(analytical, not fact):\n{buckets or '  (none)'}\n"
        f"  outlets: {comparison.get('source_count', 0)}, "
        f"opinion pieces: {comparison.get('opinion_count', 0)}\n\n"
        f"QUESTION: {question}\n\n"
        "Compare how these outlets are covering the story."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_query_understanding_prompt(question: str, history: Sequence[dict] | None = None) -> list[dict[str, str]]:
    """Ask the model to turn a question into retrieval terms -- not to answer it."""
    system = (
        "You convert a user's news question into search terms. You do NOT answer "
        "it and you do NOT use your own knowledge of events.\n\n"
        "Reply with ONLY a JSON object, no prose, no code fence:\n"
        '{"keywords": ["..."], "entities": ["..."], "topics": ["..."], '
        '"time_window_hours": 24, "intent": "search|compare_coverage|summarize|follow_up"}\n\n'
        "- keywords: 2-8 distinct content words likely to appear in relevant "
        "headlines. Expand obvious abbreviations. No filler words.\n"
        "- entities: proper nouns (companies, people, places, products).\n"
        "- topics: broad categories such as ai, linux, gaming, hardware, "
        "us-politics, geopolitics, india, business, security, science.\n"
        "- time_window_hours: 24 for 'today', 168 for 'this week', 720 for "
        "'this month', 72 if unspecified.\n"
        "- intent: compare_coverage when the user asks how outlets/sources differ; "
        "follow_up when the question only makes sense given the conversation.\n"
        "If the question refers to something earlier in the conversation, resolve "
        "the reference and include the concrete terms."
    )
    context = ""
    if history:
        recent = [
            f"{turn['role']}: {truncate(turn.get('content', ''), 220)}"
            for turn in history[-4:]
            if turn.get("role") in {"user", "assistant"}
        ]
        if recent:
            context = "CONVERSATION SO FAR:\n" + "\n".join(recent) + "\n\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context}QUESTION: {question}"},
    ]
