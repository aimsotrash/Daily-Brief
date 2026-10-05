"""Shared fixtures.

Every test runs against an isolated temporary database and a fake source
registry. Nothing here touches the network: feed fetching is injected, and the
LLM provider is a stub. That keeps the suite fast and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daily_brief.config import Config, load_config
from daily_brief.db import Database
from daily_brief.llm.base import ChatMessage, LLMProvider, LLMUnavailable
from daily_brief.models import Article, Source, utcnow
from daily_brief.news.feeds import FetchResult
from daily_brief.service import Application

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# Feeds
# --------------------------------------------------------------------------

RSS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{title}</title>
    <link>https://example.test</link>
    <language>en</language>
    {items}
  </channel>
</rss>"""

RSS_ITEM = """
    <item>
      <title>{title}</title>
      <link>{link}</link>
      <description>{description}</description>
      <pubDate>{pubdate}</pubDate>
      <guid>{link}</guid>
      {categories}
    </item>"""

ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Example</title>
  <link href="https://atom.test/"/>
  <entry>
    <title>Atom entry about Linux kernel scheduling</title>
    <link rel="alternate" href="https://atom.test/linux-kernel"/>
    <id>tag:atom.test,2026:1</id>
    <updated>2026-08-08T10:00:00Z</updated>
    <summary>The Linux kernel gained new scheduler improvements for hybrid CPUs.</summary>
    <author><name>A. Writer</name></author>
    <category term="linux"/>
  </entry>
  <entry>
    <title>Atom entry with no link</title>
    <id>not-a-url</id>
    <updated>2026-08-08T09:00:00Z</updated>
    <summary>This entry has no usable link and must be skipped.</summary>
  </entry>
</feed>"""

RDF_FEED = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel rdf:about="https://rdf.test/">
    <title>RDF Example</title>
    <link>https://rdf.test/</link>
  </channel>
  <item rdf:about="https://rdf.test/story-1">
    <title>RDF story about NVIDIA GPUs</title>
    <link>https://rdf.test/story-1</link>
    <description>NVIDIA announced new GPUs today.</description>
    <dc:date>2026-08-08T08:00:00Z</dc:date>
  </item>
</rdf:RDF>"""

MALFORMED_FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Broken</title>
<item><title>Unclosed item<link>https://broken.test/a</link>
</channel>"""

TRUNCATED_XML = b"<?xml version='1.0'?><rss><chan"

NOT_A_FEED = b"<html><body><h1>This is a web page, not a feed</h1></body></html>"

EMPTY_FEED = RSS_TEMPLATE.format(title="Empty", items="").encode()


def build_rss(items: list[dict], title: str = "Test Feed") -> bytes:
    rendered = "".join(
        RSS_ITEM.format(
            title=item["title"],
            link=item["link"],
            description=item.get("description", ""),
            pubdate=item.get(
                "pubdate", format_rfc822(utcnow() - timedelta(hours=item.get("age", 1)))
            ),
            categories="".join(
                f"<category>{c}</category>" for c in item.get("categories", [])
            ),
        )
        for item in items
    )
    return RSS_TEMPLATE.format(title=title, items=rendered).encode()


def format_rfc822(moment: datetime) -> str:
    return moment.strftime("%a, %d %b %Y %H:%M:%S +0000")


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeFetcher:
    """Stands in for :class:`FeedFetcher`; returns canned bodies per source id."""

    def __init__(self, responses: dict[str, bytes | Exception | str]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def fetch_many(self, sources, state=None):
        results = []
        for source in sources:
            self.calls.append(source.id)
            payload = self.responses.get(source.id)
            if payload is None:
                results.append(
                    FetchResult(source=source, status="error", error="no canned response")
                )
            elif isinstance(payload, Exception):
                results.append(
                    FetchResult(source=source, status="error", error=str(payload))
                )
            elif payload == "not-modified":
                results.append(FetchResult(source=source, status="not-modified"))
            else:
                body = payload if isinstance(payload, bytes) else payload.encode()
                results.append(
                    FetchResult(source=source, status="ok", body=body, etag='"abc"')
                )
        return results


class StubProvider(LLMProvider):
    """Scripted LLM. Records the prompts it was given so tests can assert on them.

    Query-understanding calls are answered separately from the scripted list, so
    a test's script maps one-to-one onto *generation* calls. That keeps tests
    readable and makes ``generation_calls`` a precise assertion target.
    """

    name = "stub"
    model = "stub-model"

    def __init__(self, responses: list[str] | None = None, available: bool = True) -> None:
        self.responses = list(responses or [])
        self._available = available
        self.calls: list[list[ChatMessage]] = []
        self.query_expansion = '{"keywords": [], "entities": [], "topics": []}'

    async def available(self) -> bool:
        return self._available

    @staticmethod
    def _is_query_understanding(messages) -> bool:
        return bool(messages) and "into search terms" in messages[0].content

    async def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        self.calls.append(messages)
        if not self._available:
            raise LLMUnavailable("stub is offline")
        if self._is_query_understanding(messages):
            return self.query_expansion
        if not self.responses:
            raise LLMUnavailable("stub ran out of scripted responses")
        return self.responses.pop(0)

    @property
    def generation_calls(self) -> list[list[ChatMessage]]:
        """Calls that asked the model to produce an answer, not to expand a query."""
        return [c for c in self.calls if not self._is_query_understanding(c)]

    @property
    def last_prompt(self) -> str:
        calls = self.generation_calls or self.calls
        return "\n".join(m.content for m in calls[-1]) if calls else ""


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(
        """
methodology:
  version: 1
  disclaimer: Test methodology.
defaults:
  enabled: true
  lean: unknown
  confidence: low
  source_type: digital-native
  country: US
sources:
  - id: tech-daily
    name: Tech Daily
    url: https://tech.test/feed
    categories: [technology, ai]
    lean: center
    confidence: medium
    source_type: trade
  - id: left-post
    name: The Left Post
    url: https://left.test/feed
    categories: [us-politics]
    lean: left
    confidence: high
    source_type: newspaper
  - id: right-herald
    name: The Right Herald
    url: https://right.test/feed
    categories: [us-politics]
    lean: right
    confidence: high
    source_type: newspaper
  - id: linux-weekly
    name: Linux Weekly
    url: https://linux.test/feed
    categories: [linux, open-source]
    lean: not-applicable
    confidence: high
    source_type: trade
  - id: broken-feed
    name: Broken Feed
    url: https://broken.test/feed
    categories: [technology]
  - id: disabled-feed
    name: Disabled Feed
    url: https://disabled.test/feed
    enabled: false
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config(tmp_path: Path, registry_path: Path) -> Config:
    cfg = load_config(None)
    cfg.storage.data_dir = str(tmp_path / "data")
    cfg.config_dir = registry_path.parent
    cfg.sources.registry = registry_path.name
    cfg.llm.provider = "none"
    cfg.logging.file = ""
    cfg.briefing.lookback_hours = 72
    cfg.search.recent_hours = 24
    return cfg


@pytest.fixture
def db(config: Config) -> Database:
    database = Database(config.storage.db_path())
    database.connect()
    yield database
    database.close()


@pytest.fixture
def app(config: Config, db: Database) -> Application:
    application = Application(config, db=db)
    yield application


@pytest.fixture
def source() -> Source:
    from daily_brief.models import Confidence, Lean

    return Source(
        id="tech-daily",
        name="Tech Daily",
        url="https://tech.test/feed",
        categories=["technology", "ai"],
        lean=Lean.CENTER,
        confidence=Confidence.MEDIUM,
        source_type="trade",
    )


def make_article(
    title: str,
    *,
    source_id: str = "tech-daily",
    source_name: str = "Tech Daily",
    url: str | None = None,
    summary: str = "",
    content: str = "",
    age_hours: float = 2.0,
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    cluster_id: str = "",
) -> Article:
    from daily_brief.text import content_hash, simhash, tokenize

    slug = "".join(c if c.isalnum() else "-" for c in title.lower())[:60]
    resolved = url or f"https://{source_id}.test/{slug}"
    article = Article(
        url=resolved,
        canonical_url=resolved,
        title=title,
        source_id=source_id,
        source_name=source_name,
        summary=summary or f"{title}.",
        content=content,
        published_at=utcnow() - timedelta(hours=age_hours),
        retrieved_at=utcnow(),
        topics=topics or [],
        entities=entities or [],
        cluster_id=cluster_id,
    )
    article.content_hash = content_hash(article.title, article.canonical_url)
    article.simhash = simhash(tokenize(title))
    # Stored articles always carry an analysis, so fixtures should too.
    from daily_brief.analysis.bias import analyse_article

    article.bias = analyse_article(article)
    return article
