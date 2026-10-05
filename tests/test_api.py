"""HTTP API: first-run flow, onboarding, preferences, search, briefing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from daily_brief.api.app import create_app
from daily_brief.news.ingest import NewsIngestor

from .conftest import FakeFetcher, StubProvider, build_rss, make_article


@pytest.fixture
def client(app):
    """A TestClient with the scheduler disabled -- no background jobs in tests."""
    api = create_app(
        application=app, enable_scheduler=False, run_initial_ingest=False
    )
    with TestClient(api) as test_client:
        yield test_client


@pytest.fixture
def seeded_client(app, client):
    app.article_repo.upsert_many([
        make_article(
            "NVIDIA announces the RTX 5090",
            source_id="tech-daily", source_name="Tech Daily",
            summary="NVIDIA revealed the RTX 5090 today with 32GB of memory.",
            topics=["hardware", "nvidia"], entities=["NVIDIA"], age_hours=2,
        ),
        make_article(
            "Linux 7.3 released with a new scheduler",
            source_id="linux-weekly", source_name="Linux Weekly",
            summary="The Linux kernel 7.3 improves scheduling on hybrid CPUs.",
            topics=["linux"], entities=["Linux"], age_hours=4,
        ),
    ])
    return client


class TestFirstRun:
    def test_reports_not_onboarded_on_a_fresh_install(self, client):
        payload = client.get("/api/preferences").json()
        assert payload["onboarded"] is False
        assert payload["interests"] == []

    def test_frontend_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "DAILY" in response.text
        assert "onboarding" in response.text

    def test_static_assets_are_served(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/styles.css").status_code == 200

    def test_health(self, client):
        assert client.get("/api/health").json() == {"ok": True}


class TestOnboarding:
    def test_completes_onboarding(self, client):
        response = client.post(
            "/api/preferences",
            json={"interests": "AI, Linux, gaming and things happening in India"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["onboarded"] is True
        assert "India" in payload["interests"]
        assert payload["parsed"]

    def test_onboarding_persists(self, client):
        client.post("/api/preferences", json={"interests": "AI, Linux"})
        assert client.get("/api/preferences").json()["onboarded"] is True

    def test_raw_text_is_returned_for_editing(self, client):
        raw = "AI, Linux and interesting things in India"
        client.post("/api/preferences", json={"interests": raw})
        assert client.get("/api/preferences").json()["raw_interests_text"] == raw

    def test_empty_interests_are_rejected(self, client):
        assert client.post("/api/preferences", json={"interests": "   "}).status_code == 400

    def test_empty_list_is_rejected(self, client):
        assert client.post("/api/preferences", json={"interests": []}).status_code == 400

    def test_uninterpretable_interests_are_rejected(self, client):
        assert client.post("/api/preferences", json={"interests": "!!! ???"}).status_code == 400

    def test_missing_field_is_a_validation_error(self, client):
        assert client.post("/api/preferences", json={}).status_code == 422

    def test_accepts_a_list(self, client):
        response = client.post("/api/preferences", json={"interests": ["AI", "Linux"]})
        assert response.status_code == 200
        assert len(response.json()["interests"]) == 2


class TestPreferenceUpdates:
    def test_updating_replaces_interests(self, client):
        client.post("/api/preferences", json={"interests": "AI, Linux"})
        payload = client.post("/api/preferences", json={"interests": "gaming"}).json()
        assert payload["interests"] == ["Gaming"]

    def test_reset_returns_to_first_run(self, client):
        client.post("/api/preferences", json={"interests": "AI, Linux"})
        assert client.post("/api/preferences/reset").json()["onboarded"] is False
        assert client.get("/api/preferences").json()["interests"] == []


class TestSearchEndpoint:
    def test_valid_query_returns_grounded_results(self, app, seeded_client):
        app.search.generator.provider = StubProvider(["NVIDIA shipped the RTX 5090 [1]."])
        app.search.provider = app.search.generator.provider
        payload = seeded_client.post("/api/search", json={"query": "What about NVIDIA?"}).json()
        assert payload["has_results"] is True
        assert payload["sources"]
        assert payload["session_id"]
        assert payload["grounded"] is True

    def test_sources_carry_attribution(self, seeded_client):
        payload = seeded_client.post("/api/search", json={"query": "NVIDIA"}).json()
        source = payload["sources"][0]
        assert source["url"].startswith("http")
        assert source["source"]
        assert "bias" in source

    def test_no_results_is_reported_honestly(self, seeded_client):
        payload = seeded_client.post(
            "/api/search", json={"query": "zzqqx flurbulator quantum banana treaty"}
        ).json()
        assert payload["has_results"] is False
        assert payload["sources"] == []
        assert "couldn't find" in payload["answer"]

    def test_empty_query_is_handled(self, seeded_client):
        payload = seeded_client.post("/api/search", json={"query": "   "}).json()
        assert payload["has_results"] is False

    def test_missing_query_is_a_validation_error(self, seeded_client):
        assert seeded_client.post("/api/search", json={}).status_code == 422

    def test_session_is_reused_for_follow_ups(self, seeded_client):
        first = seeded_client.post("/api/search", json={"query": "NVIDIA"}).json()
        second = seeded_client.post(
            "/api/search",
            json={"query": "Only the biggest ones", "session_id": first["session_id"]},
        ).json()
        assert second["session_id"] == first["session_id"]

    def test_history_is_retrievable_and_clearable(self, seeded_client):
        session = seeded_client.post("/api/search", json={"query": "NVIDIA"}).json()["session_id"]
        history = seeded_client.get("/api/search/history", params={"session_id": session}).json()
        assert len(history["messages"]) == 2
        seeded_client.delete("/api/search/history", params={"session_id": session})
        after = seeded_client.get("/api/search/history", params={"session_id": session}).json()
        assert after["messages"] == []


class TestBriefingEndpoint:
    def test_returns_a_briefing(self, app, seeded_client):
        app.preferences.save_interests("Linux, NVIDIA")
        payload = seeded_client.get("/api/briefing").json()
        assert payload["story_count"] > 0
        assert payload["sections"]

    def test_briefing_reflects_current_interests(self, app, seeded_client):
        app.preferences.save_interests("Linux")
        payload = seeded_client.post("/api/briefing/generate").json()
        titles = [s["title"] for sec in payload["sections"] for s in sec["stories"]]
        assert any("Linux" in t for t in titles)
        assert not any("NVIDIA" in t for t in titles)

    def test_regenerating_after_a_preference_change(self, app, seeded_client):
        app.preferences.save_interests("Linux")
        seeded_client.post("/api/briefing/generate")
        seeded_client.post("/api/preferences", json={"interests": "NVIDIA"})
        payload = seeded_client.post("/api/briefing/generate").json()
        titles = [s["title"] for sec in payload["sections"] for s in sec["stories"]]
        assert any("NVIDIA" in t for t in titles)

    def test_empty_briefing_explains_itself(self, client):
        client.post("/api/preferences", json={"interests": "Linux"})
        payload = client.get("/api/briefing").json()
        assert payload["sections"] == []
        assert payload["empty_reason"]


class TestRefreshEndpoint:
    def test_refresh_ingests(self, app, client):
        app.ingestor = NewsIngestor(
            app.config, app.article_repo, app.source_repo, app.cluster_repo,
            fetcher=FakeFetcher({
                "tech-daily": build_rss([
                    {"title": "A fresh story", "link": "https://tech.test/fresh"}
                ])
            }),
        )
        payload = client.post("/api/refresh", json={}).json()
        assert payload["inserted"] == 1
        assert "1 new" in payload["summary"]

    def test_refresh_reports_failures_without_erroring(self, app, client):
        app.ingestor = NewsIngestor(
            app.config, app.article_repo, app.source_repo, app.cluster_repo,
            fetcher=FakeFetcher({"tech-daily": ConnectionError("unreachable")}),
        )
        response = client.post("/api/refresh", json={})
        assert response.status_code == 200
        assert response.json()["sources_failed"] >= 1


class TestSourcesAndStatus:
    def test_lists_sources_with_bias_metadata(self, client):
        payload = client.get("/api/sources").json()
        assert len(payload["sources"]) == 6
        source = next(s for s in payload["sources"] if s["id"] == "left-post")
        assert source["lean"] == "left"
        assert source["lean_confidence"] == "high"

    def test_exposes_the_methodology(self, client):
        assert "disclaimer" in client.get("/api/sources").json()["methodology"]

    def test_status_reports_system_state(self, seeded_client):
        payload = seeded_client.get("/api/status").json()
        assert payload["articles"]["total"] == 2
        assert payload["sources"]["enabled"] == 5
        assert "available" in payload["llm"]
        assert payload["scheduler"] == []  # disabled in tests

    def test_status_reports_the_extractive_fallback(self, seeded_client):
        payload = seeded_client.get("/api/status").json()
        assert payload["llm"]["provider"] == "none"
        assert payload["llm"]["available"] is False
