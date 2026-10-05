"""Background scheduling.

These tests actually start the scheduler. Startup failures here are easy to miss
otherwise, because every other test disables it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from daily_brief.api.app import create_app
from daily_brief.news.ingest import NewsIngestor
from daily_brief.scheduler import BriefingScheduler

from .conftest import FakeFetcher, build_rss


class TestScheduler:
    async def test_starts_and_registers_both_jobs(self, app):
        # AsyncIOScheduler binds to the running loop, so this must be async.
        scheduler = BriefingScheduler(app)
        try:
            scheduler.start(run_initial=False)
            job_ids = {job["id"] for job in scheduler.jobs()}
            assert job_ids == {"refresh-feeds", "daily-briefing"}
        finally:
            scheduler.shutdown()

    async def test_jobs_have_a_next_run_time(self, app):
        scheduler = BriefingScheduler(app)
        try:
            scheduler.start(run_initial=False)
            assert all(job["next_run"] for job in scheduler.jobs())
        finally:
            scheduler.shutdown()

    def test_timezone_resolves_to_something_usable(self, app):
        """Regression: APScheduler rejects the literal string "local"."""
        scheduler = BriefingScheduler(app)
        assert scheduler.timezone is not None
        assert str(scheduler.timezone) != "local"

    def test_shutdown_is_safe_when_not_started(self, app):
        BriefingScheduler(app).shutdown()

    async def test_refresh_interval_is_floored(self, app):
        app.config.ingest.refresh_interval_minutes = 0
        scheduler = BriefingScheduler(app)
        try:
            scheduler.start(run_initial=False)
            assert scheduler.jobs()  # a zero interval must not crash the scheduler
        finally:
            scheduler.shutdown()


class TestAppStartsWithScheduler:
    def test_full_startup_with_the_scheduler_enabled(self, app):
        """The whole lifespan must run -- this is what a real `daily-brief run` does."""
        app.ingestor = NewsIngestor(
            app.config, app.article_repo, app.source_repo, app.cluster_repo,
            fetcher=FakeFetcher({
                "tech-daily": build_rss([
                    {"title": "Startup ingest story", "link": "https://tech.test/s"}
                ])
            }),
        )
        api = create_app(application=app, enable_scheduler=True, run_initial_ingest=False)
        with TestClient(api) as client:
            assert client.get("/api/health").json() == {"ok": True}
            status = client.get("/api/status").json()
            assert len(status["scheduler"]) == 2
