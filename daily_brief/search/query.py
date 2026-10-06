"""Query understanding.

Turns a natural-language question into structured retrieval terms. A
deterministic parser always runs; when an LLM is configured it refines the
result. The LLM is asked only to *expand search terms* -- never to supply facts
-- and its output is merged with, not substituted for, the deterministic parse,
so a bad completion degrades recall rather than corrupting the answer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from ..analysis.topics import LEXICON, score_topics
from ..llm.base import ChatMessage, LLMProvider, LLMUnavailable
from ..llm.prompts import build_query_understanding_prompt
from ..text import QUERY_NOISE, STOPWORDS, extract_entities, fold, tokenize

log = logging.getLogger(__name__)

Intent = str  # "search" | "compare_coverage" | "summarize" | "follow_up"

_COMPARE_RE = re.compile(
    r"\b(different sources|other sources|各|how (?:are|is|do|does)\s+\w*\s*"
    r"(?:sources?|outlets?|media|publications?|papers?)\b|compare (?:the )?coverage|"
    r"coverage compare|both sides|left and right|bias(?:ed)?|spin|framing|"
    r"how .{0,30}covering|who(?:'s| is) reporting)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(summar(?:ise|ize|y)|tl;?dr|recap|catch me up|what did i miss|brief me)\b",
    re.IGNORECASE,
)
#: Openings that only make sense as a continuation of the conversation.
_CONTINUATION_RE = re.compile(
    r"^\s*(?:and |but |so |ok(?:ay)? )?(?:what about|how about|only|just|now|"
    r"more|tell me more|expand|go deeper|and (?:the )?(?:rest|others)|"
    r"show me (?:the )?(?:biggest|most|top|other)|which one)\b",
    re.IGNORECASE,
)
#: A bare "why/who/when/where" continues the conversation only when the
#: question names no subject of its own ("why?" vs "why did NVIDIA fall?").
_BARE_QUESTION_RE = re.compile(r"^\s*(?:and |but |so )?(?:why|who|when|where)\b", re.IGNORECASE)
#: Phrases that point back at an earlier answer whatever else the question says.
_BACK_REFERENCE_RE = re.compile(
    r"\b(the story|this story|that story|these stories|those stories|the same|above|"
    r"the (?:first|second|third|last) one)\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(
    r"\b(this|that|those|these|it|its|they|them|their|he|she|his|her)\b",
    re.IGNORECASE,
)
#: "this week", "that morning": demonstratives that set a time, not refer back.
_TIME_DEMONSTRATIVE_RE = re.compile(
    r"\b(?:this|that|these|those)\s+(?:morning|afternoon|evening|night|week(?:end)?|"
    r"month|year|quarter|season|days?|time)\b",
    re.IGNORECASE,
)
#: Possessive or contraction suffix: "nvidia's" -> "nvidia", "what's" -> "what".
_POSSESSIVE_RE = re.compile(r"['’]s$")

_TIME_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\b(right now|breaking|last hour|past hour)\b", re.I), 6),
    (re.compile(r"\b(today|todays|today's|so far today|this morning|tonight)\b", re.I), 24),
    (re.compile(r"\byesterday\b", re.I), 48),
    (re.compile(r"\b(last|past|this)\s+(?:24|48)\s*hours?\b", re.I), 48),
    (re.compile(r"\b(this week|past week|last week|last 7 days|recently)\b", re.I), 168),
    (re.compile(r"\b(this month|past month|last month|last 30 days)\b", re.I), 720),
]

DEFAULT_WINDOW_HOURS = 96
MAX_WINDOW_HOURS = 24 * 30


@dataclass(slots=True)
class ParsedQuery:
    raw: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    time_window_hours: int = DEFAULT_WINDOW_HOURS
    intent: Intent = "search"
    #: True when the wording depends on earlier turns.
    is_follow_up: bool = False
    method: str = "heuristic"

    @property
    def all_terms(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for term in [*self.entities, *self.keywords]:
            key = fold(term)
            if key and key not in seen:
                seen.add(key)
                out.append(term)
        return out

    def is_empty(self) -> bool:
        return not self.all_terms and not self.topics

    def to_dict(self) -> dict:
        return {
            "keywords": self.keywords,
            "entities": self.entities,
            "topics": self.topics,
            "time_window_hours": self.time_window_hours,
            "intent": self.intent,
            "is_follow_up": self.is_follow_up,
            "method": self.method,
        }


def detect_intent(question: str, *, has_history: bool = False) -> tuple[Intent, bool]:
    """Classify the question, and say whether it depends on earlier turns.

    Without earlier turns nothing can be a follow-up, however it is phrased.
    """
    follow_up = has_history and _is_follow_up(question)
    if _COMPARE_RE.search(question):
        return ("compare_coverage", follow_up)
    if _SUMMARY_RE.search(question):
        return ("summarize", follow_up)
    if follow_up:
        return ("follow_up", True)
    return ("search", False)


def _is_follow_up(question: str) -> bool:
    if _CONTINUATION_RE.match(question) or _BACK_REFERENCE_RE.search(question):
        return True
    # A question that names its own subject stands on its own.
    if extract_entities(question, limit=1):
        return False
    if _BARE_QUESTION_RE.match(question):
        return True
    unanchored = _TIME_DEMONSTRATIVE_RE.sub(" ", question)
    return bool(_PRONOUN_RE.search(unanchored)) and len(question.split()) <= 12


def _content_keywords(text: str, exclude: set[str] | None = None) -> list[str]:
    """Search terms from free text: function words, query filler and contractions dropped."""
    out: list[str] = []
    for token in tokenize(text, drop_stopwords=False):
        token = _POSSESSIVE_RE.sub("", token)
        if "'" in token or "’" in token:
            continue  # "don't", "they're": never a search term
        if token in STOPWORDS or token in QUERY_NOISE or token in (exclude or set()):
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        if token not in out:
            out.append(token)
    return out


def detect_time_window(question: str) -> int:
    for pattern, hours in _TIME_PATTERNS:
        if pattern.search(question):
            return hours
    return DEFAULT_WINDOW_HOURS


def parse_query(question: str, history: Sequence[dict] | None = None) -> ParsedQuery:
    """Deterministic query parse. Always runs; never fails."""
    question = (question or "").strip()
    parsed = ParsedQuery(raw=question)
    if not question:
        return parsed

    intent, follow_up = detect_intent(question, has_history=bool(history))
    parsed.intent = intent
    parsed.is_follow_up = follow_up
    parsed.time_window_hours = detect_time_window(question)

    parsed.entities = extract_entities(question, limit=8)
    entity_tokens = {t for e in parsed.entities for t in tokenize(e)}

    parsed.keywords = _content_keywords(question, exclude=entity_tokens)[:10]

    scores = score_topics(question)
    parsed.topics = [
        topic
        for topic, value in sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        if value >= 1.0
    ]

    # A follow-up inherits the subject of the previous user turn.
    if follow_up and history:
        parsed = _inherit_context(parsed, history)

    return parsed


def _inherit_context(parsed: ParsedQuery, history: Sequence[dict]) -> ParsedQuery:
    """Carry entities/topics forward so 'only the biggest ones' still retrieves."""
    for turn in reversed(list(history)):
        if turn.get("role") != "user":
            continue
        prior = turn.get("content") or ""
        prior_entities = extract_entities(prior, limit=6)
        prior_keywords = _content_keywords(prior)[:8]
        if not prior_entities and not prior_keywords:
            continue
        for entity in prior_entities:
            if fold(entity) not in {fold(e) for e in parsed.entities}:
                parsed.entities.append(entity)
        for keyword in prior_keywords:
            if keyword not in parsed.keywords:
                parsed.keywords.append(keyword)
        prior_topics = [
            topic
            for topic, value in sorted(score_topics(prior).items(), key=lambda kv: -kv[1])[:2]
            if value >= 1.0
        ]
        for topic in prior_topics:
            if topic not in parsed.topics:
                parsed.topics.append(topic)
        break
    parsed.entities = parsed.entities[:10]
    parsed.keywords = parsed.keywords[:14]
    return parsed


async def refine_with_llm(
    parsed: ParsedQuery,
    provider: LLMProvider,
    history: Sequence[dict] | None = None,
) -> ParsedQuery:
    """Merge LLM-suggested search terms into a deterministic parse.

    Additive only: the heuristic terms are always kept. If the model is
    unavailable or returns junk, the parse is returned unchanged.
    """
    if not parsed.raw:
        return parsed
    try:
        messages = [
            ChatMessage(**m) for m in build_query_understanding_prompt(parsed.raw, history)
        ]
        raw = await provider.complete(messages, temperature=0.0, max_tokens=300)
    except LLMUnavailable as exc:
        log.debug("query refinement unavailable: %s", exc)
        return parsed
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("query refinement failed: %s", exc)
        return parsed

    data = _extract_json(raw)
    if not isinstance(data, dict):
        return parsed

    def _strings(key: str, limit: int) -> list[str]:
        values = data.get(key)
        if not isinstance(values, list):
            return []
        out: list[str] = []
        for value in values:
            if isinstance(value, str) and 1 < len(value) <= 60:
                cleaned = value.strip()
                if cleaned:
                    out.append(cleaned)
        return out[:limit]

    for entity in _strings("entities", 8):
        if fold(entity) not in {fold(e) for e in parsed.entities}:
            parsed.entities.append(entity)
    for keyword in _strings("keywords", 10):
        folded = fold(keyword)
        if folded not in {fold(k) for k in parsed.keywords} and folded not in QUERY_NOISE:
            parsed.keywords.append(keyword)
    for topic in _strings("topics", 4):
        slug = fold(topic).replace(" ", "-")
        if slug in LEXICON and slug not in parsed.topics:
            parsed.topics.append(slug)

    window = data.get("time_window_hours")
    if isinstance(window, (int, float)) and 1 <= window <= MAX_WINDOW_HOURS:
        # Only widen: an over-narrow window silently hides relevant results.
        parsed.time_window_hours = max(parsed.time_window_hours, int(window))

    intent = data.get("intent")
    if isinstance(intent, str) and intent in {
        "search",
        "compare_coverage",
        "summarize",
        "follow_up",
    }:
        # Trust the heuristic when it positively detected a comparison request,
        # and never call a question a follow-up when nothing came before it.
        if parsed.intent == "search" and (intent != "follow_up" or history):
            parsed.intent = intent

    parsed.entities = parsed.entities[:12]
    parsed.keywords = parsed.keywords[:16]
    parsed.method = "heuristic+llm"
    return parsed


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None
