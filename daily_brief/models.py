"""Domain model.

Plain dataclasses rather than an ORM: the schema is small, the queries are hand
written for FTS5, and keeping the model free of persistence concerns makes the
processing pipeline easy to test without a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Lean(StrEnum):
    """Coarse political-framing prior. See config/sources.yaml for methodology."""

    LEFT = "left"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    RIGHT = "right"
    STATE_AFFILIATED = "state-affiliated"
    NOT_APPLICABLE = "not-applicable"
    UNKNOWN = "unknown"

    @property
    def score(self) -> float:
        """Position on a -1 (left) .. +1 (right) axis. Non-political leans are 0."""
        return {
            Lean.LEFT: -1.0,
            Lean.CENTER_LEFT: -0.5,
            Lean.CENTER: 0.0,
            Lean.CENTER_RIGHT: 0.5,
            Lean.RIGHT: 1.0,
        }.get(self, 0.0)

    @property
    def is_political(self) -> bool:
        return self in {
            Lean.LEFT,
            Lean.CENTER_LEFT,
            Lean.CENTER,
            Lean.CENTER_RIGHT,
            Lean.RIGHT,
        }

    @classmethod
    def parse(cls, value: str | None) -> "Lean":
        if not value:
            return cls.UNKNOWN
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.UNKNOWN

    @classmethod
    def from_score(cls, score: float) -> "Lean":
        if score <= -0.66:
            return cls.LEFT
        if score <= -0.2:
            return cls.CENTER_LEFT
        if score < 0.2:
            return cls.CENTER
        if score < 0.66:
            return cls.CENTER_RIGHT
        return cls.RIGHT


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def parse(cls, value: str | None) -> "Confidence":
        try:
            return cls((value or "low").strip().lower())
        except ValueError:
            return cls.LOW

    @property
    def weight(self) -> float:
        return {Confidence.LOW: 0.35, Confidence.MEDIUM: 0.65, Confidence.HIGH: 1.0}[
            self
        ]


@dataclass(slots=True)
class Source:
    """A news source as declared in the registry."""

    id: str
    name: str
    url: str
    site: str = ""
    categories: list[str] = field(default_factory=list)
    lean: Lean = Lean.UNKNOWN
    confidence: Confidence = Confidence.LOW
    source_type: str = "digital-native"
    country: str = ""
    ownership: str = ""
    notes: str = ""
    enabled: bool = True

    def as_context(self) -> dict[str, Any]:
        """The source metadata shown alongside an article."""
        return {
            "id": self.id,
            "name": self.name,
            "site": self.site,
            "lean": str(self.lean),
            "lean_confidence": str(self.confidence),
            "source_type": self.source_type,
            "country": self.country,
            "ownership": self.ownership,
            "notes": self.notes,
        }


@dataclass(slots=True)
class BiasAssessment:
    """Article-level framing analysis.

    Always an *analytical classification*, never presented as fact. ``signals``
    carries the evidence so the UI can show why the label was assigned, and
    ``method`` records whether a heuristic or an LLM produced it.
    """

    lean: Lean = Lean.UNKNOWN
    lean_score: float = 0.0
    confidence: Confidence = Confidence.LOW
    subjectivity: float = 0.0
    is_opinion: bool = False
    framing: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    method: str = "heuristic"
    rationale: str = ""
    source_lean: Lean = Lean.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "lean": str(self.lean),
            "lean_score": round(self.lean_score, 3),
            "confidence": str(self.confidence),
            "subjectivity": round(self.subjectivity, 3),
            "is_opinion": self.is_opinion,
            "framing": list(self.framing),
            "signals": list(self.signals),
            "method": self.method,
            "rationale": self.rationale,
            "source_lean": str(self.source_lean),
            "disclaimer": (
                "Analytical classification produced by Daily-Brief, not a "
                "statement of fact."
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BiasAssessment":
        if not data:
            return cls()
        return cls(
            lean=Lean.parse(data.get("lean")),
            lean_score=float(data.get("lean_score") or 0.0),
            confidence=Confidence.parse(data.get("confidence")),
            subjectivity=float(data.get("subjectivity") or 0.0),
            is_opinion=bool(data.get("is_opinion")),
            framing=list(data.get("framing") or []),
            signals=list(data.get("signals") or []),
            method=data.get("method") or "heuristic",
            rationale=data.get("rationale") or "",
            source_lean=Lean.parse(data.get("source_lean")),
        )


@dataclass(slots=True)
class Article:
    """A normalized news article.

    Fields cover the metadata the product requires: title, source, url,
    published/retrieved timestamps, summary, topics and bias context. New fields
    can be added without touching the pipeline -- storage serialises the
    open-ended parts (topics, entities, bias) as JSON.
    """

    url: str
    title: str
    source_id: str
    source_name: str = ""
    canonical_url: str = ""
    summary: str = ""
    content: str = ""
    author: str = ""
    published_at: datetime | None = None
    retrieved_at: datetime = field(default_factory=utcnow)
    language: str = ""
    image_url: str = ""
    feed_categories: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    bias: BiasAssessment | None = None
    content_hash: str = ""
    simhash: int = 0
    cluster_id: str = ""
    id: int | None = None
    source: Source | None = None

    @property
    def best_text(self) -> str:
        """Longest available body text, for summarisation and grounding."""
        return self.content if len(self.content) > len(self.summary) else self.summary

    @property
    def display_url(self) -> str:
        return self.canonical_url or self.url

    def age_hours(self, now: datetime | None = None) -> float:
        if self.published_at is None:
            return 1e6
        reference = now or utcnow()
        return max(0.0, (reference - self.published_at).total_seconds() / 3600.0)


@dataclass(slots=True)
class Cluster:
    """A group of articles covering the same underlying story."""

    id: str
    key_title: str
    article_ids: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class ScoredArticle:
    """An article plus the retrieval scores that selected it."""

    article: Article
    score: float
    lexical: float = 0.0
    recency: float = 0.0
    interest: float = 0.0
    entity: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StoryGroup:
    """A retrieved story: one lead article plus related coverage."""

    lead: Article
    related: list[Article] = field(default_factory=list)
    score: float = 0.0

    @property
    def all_articles(self) -> list[Article]:
        return [self.lead, *self.related]

    @property
    def source_count(self) -> int:
        return len({a.source_id for a in self.all_articles})


@dataclass(slots=True)
class Preferences:
    """User personalization. Distinct from system configuration."""

    interests: list[str] = field(default_factory=list)
    raw_interests_text: str = ""
    onboarded: bool = False
    #: Reserved for future expansion without a schema migration.
    extras: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "interests": list(self.interests),
            "raw_interests_text": self.raw_interests_text,
            "onboarded": self.onboarded,
            "extras": dict(self.extras),
            "updated_at": self.updated_at.isoformat(),
        }
