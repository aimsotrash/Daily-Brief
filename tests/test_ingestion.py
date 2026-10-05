"""News ingestion: feed parsing, normalization, dedup, and failure handling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from daily_brief.models import utcnow
from daily_brief.news import dedupe
from daily_brief.news.ingest import NewsIngestor
from daily_brief.news.normalize import canonical_url, normalize_entries, normalize_entry
from daily_brief.news.parser import FeedParseError, RawEntry, parse_date, parse_feed
from daily_brief.news.sources import RegistryError, parse_registry
from daily_brief.repository import ArticleRepository, ClusterRepository, SourceRepository

from .conftest import (
    ATOM_FEED,
    EMPTY_FEED,
    MALFORMED_FEED,
    NOT_A_FEED,
    RDF_FEED,
    TRUNCATED_XML,
    FakeFetcher,
    build_rss,
    make_article,
)


# ---------------------------------------------------------------- registry


class TestSourceRegistry:
    def test_loads_valid_registry(self, registry_path):
        from daily_brief.news.sources import load_registry

        sources, methodology = load_registry(registry_path)
        assert len(sources) == 6
        assert methodology["version"] == 1
        assert {s.id for s in sources} >= {"tech-daily", "left-post", "right-herald"}

    def test_enabled_flag_is_respected(self, registry_path):
        from daily_brief.news.sources import load_registry

        sources, _ = load_registry(registry_path)
        assert next(s for s in sources if s.id == "disabled-feed").enabled is False

    def test_defaults_are_applied(self, registry_path):
        from daily_brief.news.sources import load_registry

        sources, _ = load_registry(registry_path)
        broken = next(s for s in sources if s.id == "broken-feed")
        assert str(broken.lean) == "unknown"
        assert broken.country == "US"

    def test_entry_missing_url_is_skipped_not_fatal(self):
        sources, _ = parse_registry(
            {"sources": [{"id": "ok", "url": "https://a.test/f"}, {"id": "bad"}]}
        )
        assert [s.id for s in sources] == ["ok"]

    def test_non_http_url_is_skipped(self):
        sources, _ = parse_registry(
            {"sources": [{"id": "ok", "url": "https://a.test/f"},
                         {"id": "evil", "url": "file:///etc/passwd"}]}
        )
        assert [s.id for s in sources] == ["ok"]

    def test_duplicate_ids_keep_first(self):
        sources, _ = parse_registry(
            {"sources": [{"id": "x", "url": "https://a.test/1", "name": "First"},
                         {"id": "x", "url": "https://a.test/2", "name": "Second"}]}
        )
        assert len(sources) == 1 and sources[0].name == "First"

    def test_registry_with_no_usable_sources_raises(self):
        with pytest.raises(RegistryError):
            parse_registry({"sources": []})

    def test_registry_without_sources_key_raises(self):
        with pytest.raises(RegistryError):
            parse_registry({"defaults": {}})


# ------------------------------------------------------------------ parsing


class TestFeedParsing:
    def test_parses_rss(self):
        feed = parse_feed(build_rss([
            {"title": "First story", "link": "https://tech.test/1", "description": "Body one."},
            {"title": "Second story", "link": "https://tech.test/2", "description": "Body two."},
        ]))
        assert len(feed.entries) == 2
        assert feed.entries[0].title == "First story"
        assert feed.entries[0].link == "https://tech.test/1"
        assert feed.entries[0].published is not None

    def test_parses_atom(self):
        feed = parse_feed(ATOM_FEED)
        assert len(feed.entries) == 1  # the entry with no link is dropped
        assert feed.skipped == 1
        entry = feed.entries[0]
        assert entry.link == "https://atom.test/linux-kernel"
        assert entry.author == "A. Writer"
        assert "linux" in entry.categories

    def test_parses_rdf_rss1(self):
        feed = parse_feed(RDF_FEED)
        assert len(feed.entries) == 1
        assert feed.entries[0].link == "https://rdf.test/story-1"
        assert feed.entries[0].published is not None

    def test_strips_html_from_descriptions(self):
        feed = parse_feed(build_rss([{
            "title": "HTML story",
            "link": "https://tech.test/html",
            "description": "&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;&lt;script&gt;bad()&lt;/script&gt;",
        }]))
        summary = feed.entries[0].summary
        assert "Hello world" in summary
        assert "<" not in summary and "bad()" not in summary

    def test_malformed_xml_is_salvaged_or_rejected_cleanly(self):
        # Either it salvages entries or raises FeedParseError -- never a raw XML error.
        try:
            feed = parse_feed(MALFORMED_FEED)
            assert isinstance(feed.entries, list)
        except FeedParseError:
            pass

    def test_truncated_xml_raises_feed_parse_error(self):
        with pytest.raises(FeedParseError):
            parse_feed(TRUNCATED_XML)

    def test_html_page_raises_feed_parse_error(self):
        with pytest.raises(FeedParseError):
            parse_feed(NOT_A_FEED)

    def test_empty_document_raises_feed_parse_error(self):
        with pytest.raises(FeedParseError):
            parse_feed(b"")

    def test_empty_feed_parses_to_no_entries(self):
        assert parse_feed(EMPTY_FEED).entries == []

    def test_illegal_control_characters_are_survivable(self):
        payload = build_rss([{"title": "Ctrl story", "link": "https://tech.test/c"}])
        payload = payload.replace(b"Ctrl story", b"Ctrl \x0b story")
        assert len(parse_feed(payload).entries) == 1

    @pytest.mark.parametrize(
        "value",
        [
            "Fri, 08 Aug 2026 10:00:00 GMT",
            "2026-08-08T10:00:00Z",
            "2026-08-08T10:00:00+02:00",
            "2026-08-08 10:00:00",
        ],
    )
    def test_parses_common_date_formats(self, value):
        parsed = parse_date(value)
        assert parsed is not None and parsed.tzinfo is not None

    @pytest.mark.parametrize("value", [None, "", "not a date", "yesterday-ish??"])
    def test_bad_dates_become_none(self, value):
        assert parse_date(value) is None

    def test_absurd_dates_are_rejected(self):
        assert parse_date("Mon, 01 Jan 1400 00:00:00 GMT") is None


# ------------------------------------------------------------- normalization


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://www.Example.com/story/?utm_source=x&utm_medium=y",
             "https://example.com/story"),
            ("http://example.com/story/", "https://example.com/story"),
            ("https://example.com/story?id=7&utm_campaign=z", "https://example.com/story?id=7"),
            ("https://example.com/story/amp/", "https://example.com/story"),
            ("https://example.com:443/story", "https://example.com/story"),
            ("https://example.com/a?b=1&fbclid=zzz&a=2", "https://example.com/a?a=2&b=1"),
        ],
    )
    def test_canonical_url(self, raw, expected):
        assert canonical_url(raw) == expected

    def test_canonical_url_handles_junk(self):
        assert canonical_url("") == ""
        assert canonical_url("not a url") == "not a url"

    def test_normalizes_a_full_entry(self, source):
        entry = RawEntry(
            title="  NVIDIA announces new GPUs  ",
            link="https://tech.test/nvidia?utm_source=rss",
            summary="NVIDIA revealed its next generation of GPUs today.",
            published=utcnow(),
            categories=["hardware"],
        )
        article = normalize_entry(entry, source)
        assert article is not None
        assert article.title == "NVIDIA announces new GPUs"
        assert article.canonical_url == "https://tech.test/nvidia"
        assert article.source_id == "tech-daily"
        assert article.source_name == "Tech Daily"
        assert article.content_hash and article.simhash
        assert "NVIDIA" in article.entities

    def test_title_entities_and_tags_are_decoded(self, source):
        article = normalize_entry(
            RawEntry(title="Altman says &#8216;some bad things&#8217; will happen &amp; <em>soon</em>",
                     link="https://tech.test/entities"),
            source,
        )
        assert article.title == "Altman says \u2018some bad things\u2019 will happen & soon"

    def test_entry_without_title_is_rejected(self, source):
        assert normalize_entry(RawEntry(title="", link="https://a.test/x"), source) is None

    def test_entry_without_link_is_rejected(self, source):
        assert normalize_entry(RawEntry(title="Something", link=""), source) is None

    def test_non_http_link_is_rejected(self, source):
        assert normalize_entry(
            RawEntry(title="Bad", link="javascript:alert(1)"), source
        ) is None

    def test_missing_metadata_is_tolerated(self, source):
        """A title and a link are the only hard requirements."""
        article = normalize_entry(
            RawEntry(title="Bare minimum story", link="https://tech.test/bare"), source
        )
        assert article is not None
        assert article.published_at is None
        assert article.author == ""
        assert article.summary == ""

    def test_long_description_becomes_content(self, source):
        body = "Sentence number one is fairly long. " * 30
        article = normalize_entry(
            RawEntry(title="Long", link="https://tech.test/long", summary=body), source
        )
        assert len(article.content) > len(article.summary)

    def test_normalize_entries_sorts_newest_first_and_caps(self, source):
        entries = [
            RawEntry(title=f"Story {i}", link=f"https://tech.test/{i}",
                     published=utcnow() - timedelta(hours=i))
            for i in range(10)
        ]
        articles = normalize_entries(entries, source, limit=3)
        assert len(articles) == 3
        assert articles[0].title == "Story 0"

    def test_undated_entries_are_kept(self, source):
        entries = [
            RawEntry(title="Dated", link="https://tech.test/d", published=utcnow()),
            RawEntry(title="Undated", link="https://tech.test/u"),
        ]
        assert len(normalize_entries(entries, source, limit=10)) == 2


# ---------------------------------------------------------------- dedup


class TestDeduplication:
    def test_exact_duplicate_urls_collapse(self):
        a = make_article("Same story", url="https://x.test/a")
        b = make_article("Same story", url="https://x.test/a")
        kept, removed = dedupe.dedupe_exact([a, b])
        assert len(kept) == 1 and removed == 1

    def test_tracking_params_do_not_defeat_dedup(self, source):
        first = normalize_entry(
            RawEntry(title="One story", link="https://tech.test/s?utm_source=a"), source
        )
        second = normalize_entry(
            RawEntry(title="One story", link="https://tech.test/s?utm_source=b"), source
        )
        kept, removed = dedupe.dedupe_exact([first, second])
        assert len(kept) == 1 and removed == 1

    def test_richer_duplicate_wins(self):
        thin = make_article("Story", url="https://x.test/a")
        rich = make_article("Story", url="https://x.test/a", content="Much longer body text.")
        kept, _ = dedupe.dedupe_exact([thin, rich])
        assert kept[0].content == "Much longer body text."

    def test_near_duplicate_from_same_source_removed(self):
        a = make_article("NVIDIA announces the new RTX 5090 graphics card")
        b = make_article(
            "NVIDIA announces the new RTX 5090 graphics card!",
            url="https://tech-daily.test/other",
        )
        kept, removed = dedupe.dedupe_near([a, b])
        assert len(kept) == 1 and removed == 1

    def test_similar_headlines_across_sources_are_kept(self):
        """Independent coverage is not duplication -- it is the signal we want."""
        a = make_article("NVIDIA announces new RTX 5090", source_id="tech-daily")
        b = make_article("NVIDIA announces new RTX 5090", source_id="left-post")
        kept, removed = dedupe.dedupe_near([a, b])
        assert len(kept) == 2 and removed == 0

    def test_unrelated_articles_are_not_deduped(self):
        articles = [
            make_article("Linux kernel 7.3 released with new scheduler"),
            make_article("Senate passes a spending bill before recess"),
        ]
        kept, removed = dedupe.dedupe_exact(articles)
        assert len(kept) == 2 and removed == 0

    def test_clusters_same_story_across_sources(self):
        articles = [
            make_article("Senate confirms Todd Blanche as attorney general",
                         source_id="left-post", entities=["Senate", "Todd Blanche"]),
            make_article("Senate confirms Blanche as attorney general",
                         source_id="right-herald", entities=["Senate", "Blanche"]),
            make_article("Linux 7.3 improves hybrid CPU scheduling",
                         source_id="linux-weekly", entities=["Linux"]),
        ]
        clusters = dedupe.cluster_stories(articles)
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 2]

    def test_cluster_ids_are_deterministic(self):
        articles = [
            make_article("Senate confirms Todd Blanche", source_id="left-post"),
            make_article("Senate confirms Todd Blanche", source_id="right-herald"),
        ]
        first = dedupe.assign_clusters(list(articles))
        second = dedupe.assign_clusters(list(articles))
        assert set(first) == set(second)

    def test_articles_far_apart_in_time_are_not_clustered(self):
        articles = [
            make_article("Senate confirms Todd Blanche", source_id="left-post", age_hours=1),
            make_article("Senate confirms Todd Blanche", source_id="right-herald", age_hours=400),
        ]
        assert len(dedupe.cluster_stories(articles)) == 2

    def test_single_article_forms_its_own_cluster(self):
        assert len(dedupe.cluster_stories([make_article("Solo")])) == 1

    def test_empty_input(self):
        assert dedupe.cluster_stories([]) == []
        assert dedupe.dedupe_exact([]) == ([], 0)


# ---------------------------------------------------------- full pipeline


class TestIngestPipeline:
    def _ingestor(self, app, responses):
        return NewsIngestor(
            app.config,
            app.article_repo,
            app.source_repo,
            app.cluster_repo,
            fetcher=FakeFetcher(responses),
        )

    async def test_happy_path(self, app):
        ingestor = self._ingestor(app, {
            "tech-daily": build_rss([
                {"title": "NVIDIA ships new GPUs", "link": "https://tech.test/1"},
                {"title": "OpenAI releases a model", "link": "https://tech.test/2"},
            ]),
            "linux-weekly": build_rss([
                {"title": "Linux 7.3 released", "link": "https://linux.test/1"},
            ]),
        })
        report = await ingestor.run()
        assert report.inserted == 3
        assert report.sources_ok == 2
        assert app.article_repo.count() == 3

    async def test_disabled_sources_are_not_fetched(self, app):
        fetcher = FakeFetcher({})
        ingestor = NewsIngestor(
            app.config, app.article_repo, app.source_repo, app.cluster_repo, fetcher=fetcher
        )
        await ingestor.run()
        assert "disabled-feed" not in fetcher.calls

    async def test_unavailable_source_does_not_fail_the_run(self, app):
        ingestor = self._ingestor(app, {
            "tech-daily": build_rss([{"title": "Good story", "link": "https://tech.test/1"}]),
            "left-post": ConnectionError("host unreachable"),
        })
        report = await ingestor.run()
        assert report.inserted == 1
        assert report.sources_failed >= 1
        assert app.article_repo.count() == 1

    async def test_malformed_feed_does_not_fail_the_run(self, app):
        ingestor = self._ingestor(app, {
            "tech-daily": build_rss([{"title": "Good story", "link": "https://tech.test/1"}]),
            "broken-feed": NOT_A_FEED,
        })
        report = await ingestor.run()
        assert report.inserted == 1
        assert any(r.error for r in report.per_source)

    async def test_failures_are_recorded_for_backoff(self, app):
        ingestor = self._ingestor(app, {"broken-feed": ConnectionError("nope")})
        await ingestor.run()
        state = app.source_repo.get_feed_state("broken-feed")
        assert state["consecutive_failures"] == 1
        assert state["last_error"]

    async def test_repeated_failures_pause_a_source(self, app):
        app.config.ingest.max_consecutive_failures = 2
        ingestor = self._ingestor(app, {"broken-feed": ConnectionError("nope")})
        await ingestor.run()
        await ingestor.run()
        fetcher = FakeFetcher({})
        paused = NewsIngestor(
            app.config, app.article_repo, app.source_repo, app.cluster_repo, fetcher=fetcher
        )
        await paused.run()
        assert "broken-feed" not in fetcher.calls

    async def test_not_modified_is_not_a_failure(self, app):
        ingestor = self._ingestor(app, {"tech-daily": "not-modified"})
        report = await ingestor.run(["tech-daily"])
        assert report.sources_not_modified == 1
        assert report.sources_failed == 0
        assert report.inserted == 0

    async def test_reingesting_the_same_feed_inserts_nothing_new(self, app):
        feed = build_rss([{"title": "Repeat story", "link": "https://tech.test/1"}])
        ingestor = self._ingestor(app, {"tech-daily": feed})
        await ingestor.run()
        second = await ingestor.run()
        assert second.inserted == 0
        assert app.article_repo.count() == 1

    async def test_duplicates_across_sources_are_clustered_not_dropped(self, app):
        ingestor = self._ingestor(app, {
            "left-post": build_rss([{
                "title": "Senate confirms Todd Blanche as attorney general",
                "link": "https://left.test/1"}]),
            "right-herald": build_rss([{
                "title": "Senate confirms Todd Blanche as attorney general",
                "link": "https://right.test/1"}]),
        })
        await ingestor.run()
        assert app.article_repo.count() == 2
        stored = app.article_repo.recent(limit=10)
        assert stored[0].cluster_id == stored[1].cluster_id

    async def test_articles_are_classified_and_bias_analysed(self, app):
        ingestor = self._ingestor(app, {
            "linux-weekly": build_rss([{
                "title": "Linux kernel 7.3 improves the scheduler",
                "link": "https://linux.test/1",
                "description": "The Linux kernel gained scheduler improvements for hybrid CPUs.",
            }]),
        })
        await ingestor.run()
        article = app.article_repo.recent(limit=1)[0]
        assert "linux" in article.topics
        assert article.bias is not None
        assert article.bias.method == "heuristic"

    async def test_retention_prunes_old_articles(self, app):
        app.config.ingest.retention_days = 1
        old = make_article("Ancient story", age_hours=24 * 30)
        app.article_repo.upsert_many([old])
        ingestor = self._ingestor(app, {
            "tech-daily": build_rss([{"title": "New story", "link": "https://tech.test/n"}])
        })
        await ingestor.run()
        titles = [a.title for a in app.article_repo.recent(limit=50)]
        assert "Ancient story" not in titles

    async def test_empty_feed_is_handled(self, app):
        report = await self._ingestor(app, {"tech-daily": EMPTY_FEED}).run()
        assert report.inserted == 0
        assert report.sources_ok == 1

    async def test_entries_missing_metadata_still_ingest(self, app):
        feed = (
            b'<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
            b"<item><title>No date and no description</title>"
            b"<link>https://tech.test/nometa</link></item></channel></rss>"
        )
        report = await self._ingestor(app, {"tech-daily": feed}).run()
        assert report.inserted == 1
        assert app.article_repo.recent(limit=1)[0].published_at is None
