"""Daily briefing generation.

    recent articles
        -> score against the user's interests
        -> drop what isn't relevant
        -> group related coverage into stories
        -> de-duplicate
        -> rank by importance
        -> summarize
        -> organise into sections

The briefing deliberately does not dump every retrieved article: an article only
appears if it clears an interest threshold, one story appears once no matter how
many outlets ran it, and each section is capped.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Sequence

from ..analysis.relevance import MATCH_THRESHOLD, Interest, score_interests
from ..analysis.summarize import summarize_article, summarize_group
from ..analysis.topics import label_for
from ..config import Config
from ..llm.base import ChatMessage, LLMProvider, LLMUnavailable
from ..llm.prompts import build_briefing_prompt
from ..models import Article, StoryGroup
from ..repository import ArticleRepository, BriefingRepository
from ..search.answer import CITATION_RE, article_to_source, build_coverage
from ..text import tokenize, truncate

log = logging.getLogger(__name__)

#: An article must reach this interest score to enter a personalised briefing.
INTEREST_THRESHOLD = 0.4

#: A headline word used by at most this many stories in the pool is
#: distinctive: a name like "OpenRadioss", not a word like "AI".
DISTINCTIVE_MAX_STORIES = 6


class BriefingGenerator:
    def __init__(
        self,
        config: Config,
        articles: ArticleRepository,
        briefings: BriefingRepository,
        provider: LLMProvider,
    ) -> None:
        self.config = config
        self.articles = articles
        self.briefings = briefings
        self.provider = provider

    # ------------------------------------------------------------------ public
    async def generate(
        self,
        interests: Sequence[Interest],
        *,
        now: datetime | None = None,
        persist: bool = True,
    ) -> dict:
        now = now or datetime.now(timezone.utc)
        cfg = self.config.briefing

        pool = self.articles.recent(hours=cfg.lookback_hours, limit=2000)
        if not pool:
            payload = self._empty_payload(interests, now, "no articles have been ingested yet")
            if persist:
                self.briefings.save(payload)
            return payload

        ranked = self._rank(pool, interests, now)
        if not ranked:
            payload = self._empty_payload(
                interests,
                now,
                "none of the recently ingested articles matched your interests",
            )
            payload["article_pool"] = len(pool)
            if persist:
                self.briefings.save(payload)
            return payload

        groups = self._group(ranked)
        sections = self._sectionize(groups, interests)
        sections = await self._summarize_sections(sections, interests, now)

        story_count = sum(len(s["stories"]) for s in sections)
        payload = {
            "date": now.date().isoformat(),
            "generated_at": now.isoformat(),
            "engine": (
                f"llm:{self.provider.model}"
                if await self.provider.available()
                else "extractive"
            ),
            "interests": [i.label for i in interests],
            "sections": sections,
            "story_count": story_count,
            "article_pool": len(pool),
            "lookback_hours": cfg.lookback_hours,
            "empty_reason": "",
        }
        if persist:
            payload["id"] = self.briefings.save(payload)
            self.briefings.prune(keep=30)
        log.info(
            "briefing generated: %d stories across %d sections from %d articles",
            story_count,
            len(sections),
            len(pool),
        )
        return payload

    def latest(self) -> dict | None:
        return self.briefings.latest()

    def is_stale(self, payload: dict | None, now: datetime | None = None) -> bool:
        if not payload:
            return True
        now = now or datetime.now(timezone.utc)
        try:
            generated = datetime.fromisoformat(payload["generated_at"])
        except (KeyError, ValueError):
            return True
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        age_minutes = (now - generated).total_seconds() / 60
        return age_minutes > self.config.briefing.max_age_minutes

    # ----------------------------------------------------------------- ranking
    def _rank(
        self, pool: Sequence[Article], interests: Sequence[Interest], now: datetime
    ) -> list[tuple[Article, float, dict[str, float]]]:
        """Score articles by interest match, recency and source signal.

        The per-interest scores are carried forward, not just the best one, so
        sectioning can place each story under the interest it matches *most*
        strongly rather than the first one that happened to match.
        """
        out: list[tuple[Article, float, dict[str, float]]] = []
        for article in pool:
            matched = score_interests(article, list(interests))
            interest_score = max(matched.values()) if matched else 0.5
            if interests and interest_score < INTEREST_THRESHOLD:
                continue

            age = article.age_hours(now)
            recency = max(0.0, 1.0 - min(1.0, age / max(1.0, self.config.briefing.lookback_hours)))

            substance = min(1.0, len(article.best_text) / 900.0)
            bias = article.bias
            # Straight reporting is preferred over commentary in a briefing.
            opinion_penalty = 0.12 if bias and bias.is_opinion else 0.0
            promo_penalty = 0.08 if bias and "first-party" in bias.framing else 0.0

            score = (
                0.50 * interest_score
                + 0.28 * recency
                + 0.12 * substance
                + 0.10 * (1.0 if article.published_at else 0.4)
                - opinion_penalty
                - promo_penalty
            )
            out.append((article, score, matched))

        out.sort(key=lambda item: -item[1])
        return out

    def _group(
        self, ranked: Sequence[tuple[Article, float, dict[str, float]]]
    ) -> list[tuple[StoryGroup, float, dict[str, float]]]:
        """Collapse ranked articles into stories using stored cluster ids.

        A story covered by several outlets is boosted -- several independent
        outlets running it is the best available proxy for importance -- but it
        still occupies exactly one slot in the briefing.

        Interest scores come from the *lead* article only. Unioning them across
        the cluster would let one loosely-matching member drag the whole story
        into an unrelated section.
        """
        by_cluster: dict[str, list[tuple[Article, float, dict[str, float]]]] = defaultdict(list)
        order: list[str] = []
        for article, score, matched in ranked:
            key = article.cluster_id or f"solo:{article.id}"
            if key not in by_cluster:
                order.append(key)
            by_cluster[key].append((article, score, matched))

        groups: list[tuple[StoryGroup, float, dict[str, float]]] = []
        for key in order:
            members = by_cluster[key]
            lead_article, lead_score, lead_matched = members[0]
            related = [a for a, _, _ in members[1:]]

            distinct_sources = len({a.source_id for a, _, _ in members})
            corroboration = min(0.22, 0.08 * (distinct_sources - 1))
            total = lead_score + corroboration

            groups.append(
                (
                    StoryGroup(lead=lead_article, related=related, score=total),
                    total,
                    lead_matched,
                )
            )

        groups.sort(key=lambda item: -item[1])
        return groups

    def _sectionize(
        self,
        groups: Sequence[tuple[StoryGroup, float, dict[str, float]]],
        interests: Sequence[Interest],
    ) -> list[dict]:
        """Organise stories into sections, one per interest where possible.

        Each story goes to the interest it matches *best*. If that section is
        already full it falls through to its next-best interest, so a strong
        story is not dropped just because one section filled up first.
        """
        cfg = self.config.briefing
        interest_order = [i.label for i in interests]
        buckets: dict[str, list[tuple[StoryGroup, float, list[str]]]] = defaultdict(list)
        # One outlet should not fill a whole section just because it publishes a lot.
        per_source_cap = max(2, cfg.max_stories_per_section // 2)
        source_counts: dict[tuple[str, str], int] = defaultdict(int)
        used: set[str] = set()
        placed_headlines: list[set[str]] = []
        distinctive = _distinctive_words(group.lead.title for group, _, _ in groups)
        total = 0

        def has_room(label: str, source_id: str) -> bool:
            return (
                len(buckets[label]) < cfg.max_stories_per_section
                and source_counts[(label, source_id)] < per_source_cap
            )

        for group, score, matched in groups:
            if total >= cfg.max_total_stories:
                break
            key = group.lead.canonical_url or group.lead.url
            if key in used:
                continue
            # Outlets word one event differently enough that clustering keeps
            # them apart; the briefing still gives the event one slot.
            headline = set(tokenize(group.lead.title))
            if any(_same_event(headline, other, distinctive) for other in placed_headlines):
                continue
            source_id = group.lead.source_id

            ranked_labels = sorted(
                (label for label, value in matched.items() if value >= MATCH_THRESHOLD),
                key=lambda label: -matched[label],
            )
            section = next(
                (label for label in ranked_labels if has_room(label, source_id)), ""
            )
            if not section:
                if interests and ranked_labels:
                    # Every interest it matches is full; leave it for tomorrow.
                    continue
                if interests:
                    continue
                topic = group.lead.topics[0] if group.lead.topics else "general"
                section = label_for(topic)
                if not has_room(section, source_id):
                    continue

            buckets[section].append((group, score, ranked_labels))
            source_counts[(section, source_id)] += 1
            used.add(key)
            placed_headlines.append(headline)
            total += 1

        # Order sections by the user's interest order, then by aggregate score.
        def section_sort_key(item: tuple[str, list]) -> tuple:
            name, entries = item
            try:
                rank = interest_order.index(name)
            except ValueError:
                rank = len(interest_order) + 1
            return (rank, -sum(score for _, score, _ in entries))

        ordered = sorted(buckets.items(), key=section_sort_key)[: cfg.max_sections]

        sections: list[dict] = []
        for name, entries in ordered:
            if not entries:
                continue
            sections.append(
                {
                    "title": name,
                    "stories": [
                        self._story_payload(group, score, matched, index)
                        for index, (group, score, matched) in enumerate(entries, start=1)
                    ],
                }
            )
        return sections

    def _story_payload(
        self, group: StoryGroup, score: float, matched: list[str], index: int
    ) -> dict:
        lead = group.lead
        payload = article_to_source(lead, index, score=score)

        # Coverage is reported over the *whole* story cluster, not just the
        # members that happened to clear the interest threshold. Otherwise a
        # story run by seven outlets could announce "covered by 3 sources"
        # purely because the other four did not match the user's interests.
        related = group.related
        if lead.cluster_id:
            members = self.articles.cluster_members(lead.cluster_id, exclude_id=lead.id)
            if len(members) > len(related):
                related = members

        payload.update(
            {
                "summary": summarize_group([lead, *group.related], max_sentences=2)
                if group.related
                else summarize_article(lead, max_sentences=2),
                "matched_interests": matched,
                "source_count": len({lead.source_id, *(a.source_id for a in related)}),
                "coverage": build_coverage(lead, related),
                # Retained for callers that only want the sibling outlets.
                "related": build_coverage(lead, related)[1:],
            }
        )
        return payload

    # ------------------------------------------------------------ summarising
    async def _summarize_sections(
        self, sections: list[dict], interests: Sequence[Interest], now: datetime
    ) -> list[dict]:
        """Optionally rewrite each section's story lines with the LLM.

        Every story already carries a deterministic extractive summary; this only
        replaces it when the model produces a properly cited alternative.
        """
        if not sections or not await self.provider.available():
            return sections

        labels = [i.label for i in interests]
        for section in sections:
            articles = [
                self.articles.get_by_url(story["url"]) for story in section["stories"]
            ]
            articles = [a for a in articles if a is not None]
            if not articles:
                continue
            try:
                messages = [
                    ChatMessage(**m)
                    for m in build_briefing_prompt(section["title"], articles, labels, now=now)
                ]
                raw = await self.provider.complete(messages, max_tokens=700)
            except LLMUnavailable as exc:
                log.info("briefing summarisation unavailable: %s", exc)
                break
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("briefing summarisation failed for %r: %s", section["title"], exc)
                continue

            lines = self._split_cited_lines(raw, len(articles))
            for number, text in lines.items():
                if 1 <= number <= len(section["stories"]) and text:
                    section["stories"][number - 1]["summary"] = truncate(text, 420)
                    section["stories"][number - 1]["summary_engine"] = "llm"
        return sections

    @staticmethod
    def _split_cited_lines(raw: str, article_count: int) -> dict[int, str]:
        """Map ``[n]``-cited lines back onto their article.

        Lines citing an article that was not supplied are discarded -- the same
        citation check the search path applies.
        """
        out: dict[int, str] = {}
        for line in raw.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if not line:
                continue
            numbers = [int(m.group(1)) for m in CITATION_RE.finditer(line)]
            valid = [n for n in numbers if 1 <= n <= article_count]
            if not valid:
                continue
            text = CITATION_RE.sub("", line).strip()
            text = text.lstrip("0123456789.).: ").strip()
            if len(text) < 20:
                continue
            key = valid[0]
            out[key] = f"{out[key]} {text}".strip() if key in out else text
        return out

    # ---------------------------------------------------------------- helpers
    def _empty_payload(
        self, interests: Sequence[Interest], now: datetime, reason: str
    ) -> dict:
        return {
            "date": now.date().isoformat(),
            "generated_at": now.isoformat(),
            "engine": "none",
            "interests": [i.label for i in interests],
            "sections": [],
            "story_count": 0,
            "article_pool": 0,
            "lookback_hours": self.config.briefing.lookback_hours,
            "empty_reason": reason,
        }


def _distinctive_words(headlines: Iterable[str]) -> set[str]:
    """Headline words rare enough in this pool to identify a particular event."""
    counts: dict[str, int] = defaultdict(int)
    for headline in headlines:
        for word in set(tokenize(headline)):
            counts[word] += 1
    return {
        word
        for word, count in counts.items()
        if count <= DISTINCTIVE_MAX_STORIES and len(word) >= 4 and not word.isdigit()
    }


def _same_event(left: set[str], right: set[str], distinctive: set[str]) -> bool:
    """Two headlines describe one event when they share a distinctive word and more."""
    shared = left & right
    return len(shared) >= 2 and bool(shared & distinctive)
