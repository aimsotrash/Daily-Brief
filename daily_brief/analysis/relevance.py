"""Interpreting free-text interests and scoring articles against them.

The user types whatever they like ("AI, Linux, gaming and interesting things
happening in India"). This module turns that into structured
:class:`Interest` objects and scores articles against them. Nothing here is
specific to any user or any set of interests -- the lexicon in
:mod:`.topics` provides the expansions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Article
from ..text import contains_term, fold, tokenize
from .topics import LEXICON, label_for

#: Filler that appears when interests are typed as a sentence.
_LEADING_FILLER = re.compile(
    r"^(?:and|also|plus|as well as|interesting|the|some|any|all|things|stuff|"
    r"news|updates?|anything|everything|happening|going on|about|in|on|with|"
    r"related to|latest)\b\s*",
    re.IGNORECASE,
)
_TRAILING_FILLER = re.compile(
    r"\s*\b(?:news|updates?|stories|stuff|things|topics?|in general)\b$", re.IGNORECASE
)
_SPLIT = re.compile(r"[,;\n\r•·|/]+|\s+&\s+|\s+\band\b\s+|\s+\bplus\b\s+", re.IGNORECASE)


@dataclass(slots=True)
class Interest:
    """One parsed user interest."""

    label: str
    #: Canonical topic key when the interest maps onto one, else "".
    topic: str = ""
    #: Every matchable term: the label, its tokens, and lexicon expansions.
    terms: list[str] = field(default_factory=list)
    #: Terms drawn verbatim from what the user typed.
    literal_terms: list[str] = field(default_factory=list)
    #: Distinctive terms -- multi-word phrases and long unambiguous words. Only
    #: these can qualify an article on their own; generic single words like
    #: "game" or "chip" appear incidentally in unrelated body text far too often.
    strong_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "topic": self.topic, "terms": self.terms[:40]}


def _canonical_topic(phrase: str) -> str:
    """Map a phrase onto a canonical topic key when it clearly names one."""
    folded = fold(phrase).strip()
    slug = folded.replace(" ", "-")
    if slug in LEXICON:
        return slug
    aliases = {
        "artificial intelligence": "ai",
        "machine learning": "ai",
        "llms": "ai",
        "llm": "ai",
        "us politics": "us-politics",
        "american politics": "us-politics",
        "us-politics": "us-politics",
        "politics": "us-politics",
        "geopolitics": "geopolitics",
        "world news": "world",
        "international": "world",
        "foreign policy": "geopolitics",
        "tech": "technology",
        "computing": "technology",
        "video games": "gaming",
        "videogames": "gaming",
        "games": "gaming",
        "pc gaming": "gaming",
        "open source": "open-source",
        "foss": "open-source",
        "gpus": "hardware",
        "gpu": "hardware",
        "chips": "hardware",
        "semiconductors": "hardware",
        "cybersecurity": "security",
        "infosec": "security",
        "privacy": "security",
        "economy": "markets",
        "finance": "markets",
        "stock market": "markets",
        "space exploration": "space",
        "climate change": "climate",
        "energy": "climate",
        "indian": "india",
        "bharat": "india",
        "movies": "culture",
        "entertainment": "culture",
        "sport": "sports",
    }
    if folded in aliases:
        return aliases[folded]
    # A phrase that is itself a lexicon term (e.g. "nvidia") maps to that topic.
    for topic, terms in LEXICON.items():
        if folded in terms:
            return topic
    return ""


def parse_interests(raw: str | list[str]) -> list[Interest]:
    """Parse free-text interests into structured :class:`Interest` objects."""
    if isinstance(raw, list):
        phrases = [str(part) for part in raw]
    else:
        phrases = _SPLIT.split(raw or "")

    interests: list[Interest] = []
    seen: set[str] = set()
    by_topic: dict[str, Interest] = {}

    for phrase in phrases:
        cleaned = " ".join(phrase.strip().split())
        # Strip conversational filler repeatedly ("and interesting things in India").
        for _ in range(4):
            new = _LEADING_FILLER.sub("", cleaned).strip()
            new = _TRAILING_FILLER.sub("", new).strip()
            if new == cleaned:
                break
            cleaned = new
        cleaned = cleaned.strip(" .!?-–—\"'")
        if len(cleaned) < 2 or len(cleaned) > 60:
            continue

        key = fold(cleaned)
        if key in seen:
            continue
        seen.add(key)

        topic = _canonical_topic(cleaned)

        # "AI", "A.I." and "artificial intelligence" are one interest, not three.
        # Merge into the existing entry, keeping the extra phrasing for matching.
        if topic and topic in by_topic:
            existing = by_topic[topic]
            if cleaned not in existing.literal_terms:
                existing.literal_terms.append(cleaned)
            for token in {fold(cleaned), *tokenize(cleaned)}:
                if len(token) >= 2 and token not in existing.terms:
                    existing.terms.append(token)
            existing.terms.sort()
            continue

        literal = [cleaned]
        terms = {fold(cleaned)}
        terms.update(tokenize(cleaned))
        strong = {fold(cleaned)}
        strong.update(t for t in tokenize(cleaned) if len(t) >= 5)
        if topic:
            for term in LEXICON.get(topic, []):
                folded = fold(term)
                terms.add(folded)
                # Phrases are unambiguous; so are long single words. Short ones
                # ("ai", "gpt", "chip", "game") are not, and stay weak-only.
                if " " in folded or len(folded) >= 7:
                    strong.add(folded)

        label = label_for(topic) if topic else cleaned.title()
        interest = Interest(
            label=label,
            topic=topic,
            terms=sorted(t for t in terms if len(t) >= 2),
            literal_terms=literal,
            strong_terms=sorted(strong),
        )
        interests.append(interest)
        if topic:
            by_topic[topic] = interest

    return interests


def interest_labels(interests: list[Interest]) -> list[str]:
    return [i.label for i in interests]


#: An interest must reach this to count as a match.
MATCH_THRESHOLD = 0.4


def score_interests(article: Article, interests: list[Interest]) -> dict[str, float]:
    """Score an article against each interest individually.

    The dominant signal is the topic classifier's output, not raw keyword
    overlap: the classifier already weighs phrases, demotes generic categories
    and considers feed metadata. Keyword overlap only *adds* evidence, and
    generic single words can never qualify an article on their own -- that was
    the difference between "this is an AI story" and "this article happens to
    contain the word 'inference' somewhere in paragraph nine".
    """
    if not interests:
        return {}

    title_folded = fold(article.title)
    body_folded = fold(f"{article.summary} {article.content[:2000]}")
    title_tokens = set(tokenize(article.title))
    body_tokens = set(tokenize(f"{article.summary} {article.content[:2000]}"))
    entity_blob = fold(" ".join(article.entities))
    lead_topic = article.topics[0] if article.topics else ""
    article_topics = set(article.topics)

    scores: dict[str, float] = {}
    for interest in interests:
        score = 0.0

        # 1. Topic agreement -- the strongest and best-calibrated signal.
        if interest.topic:
            if interest.topic == lead_topic:
                score = max(score, 0.9)
            elif interest.topic in article_topics:
                score = max(score, 0.7)

        # 2. The literal phrase the user typed.
        for literal in interest.literal_terms:
            needle = fold(literal)
            if len(needle) < 3:
                continue
            if contains_term(title_folded, needle):
                score = max(score, 0.9)
            elif contains_term(entity_blob, needle):
                score = max(score, 0.8)
            elif len(needle) >= 5 and contains_term(body_folded, needle):
                score = max(score, 0.55)

        # 3. Distinctive expansion terms.
        strong = set(interest.strong_terms)
        strong_title = len(title_tokens & strong) + sum(
            1 for term in strong if " " in term and contains_term(title_folded, term)
        )
        strong_body = len(body_tokens & strong) + sum(
            1 for term in strong if " " in term and contains_term(body_folded, term)
        )
        if strong_title:
            score = max(score, min(0.85, 0.55 + 0.1 * strong_title))
        elif strong_body >= 2:
            score = max(score, min(0.6, 0.35 + 0.08 * strong_body))

        # 4. Generic overlap: corroborating only. Capped below the match
        #    threshold so it can never qualify an article by itself.
        weak_title = len(title_tokens & set(interest.terms))
        weak_body = len(body_tokens & set(interest.terms))
        if weak_title >= 2:
            score = max(score, 0.38)
        elif weak_body >= 4:
            score = max(score, 0.3)

        # A weak signal that agrees with the topic classifier is promoted.
        if 0.3 <= score < MATCH_THRESHOLD and interest.topic in article_topics:
            score = 0.5

        scores[interest.label] = min(1.0, score)

    return scores


def score_article(article: Article, interests: list[Interest]) -> tuple[float, list[str]]:
    """Score an article against the user's interests.

    Returns ``(best score in 0..1, matched interest labels ordered by score)``.
    With no interests configured everything scores neutrally, so an
    un-personalised briefing still works.
    """
    if not interests:
        return (0.5, [])
    scores = score_interests(article, interests)
    if not scores:
        return (0.0, [])
    matched = sorted(
        (label for label, value in scores.items() if value >= MATCH_THRESHOLD),
        key=lambda label: -scores[label],
    )
    return (max(scores.values()), matched)
