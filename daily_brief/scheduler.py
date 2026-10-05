"""Background scheduling.

APScheduler running in-process on the server's event loop. For a single-user
self-hosted application that is the right size: no broker, no worker process, no
extra daemon to supervise. Two jobs:

* feed refresh on a fixed interval,
* briefing generation at a configured local time.

Both are also exposed through the CLI and the API, so anyone who would rather
drive Daily-Brief from systemd timers or cron can disable the scheduler and call
``daily-brief refresh`` / ``daily-brief generate`` instead.
"""

from __future__ import annotations

import logging
from datetime import timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .service import Application

log = logging.getLogger(__name__)


def _local_timezone():
    """Resolve the machine's timezone, falling back to UTC.

    APScheduler will not accept the literal string ``"local"``, and
    ``get_localzone()`` raises on hosts with no usable zone configuration
    (minimal containers, for example). The briefing schedule is a convenience,
    so a missing timezone should degrade to UTC rather than block startup.
    """
    try:
        from tzlocal import get_localzone

        return get_localzone()
    except Exception:  # pragma: no cover - depends on host tz configuration
        log.warning("could not determine the local timezone; scheduling in UTC")
        return timezone.utc


class BriefingScheduler:
    def __init__(self, app: Application) -> None:
        self.app = app
        self.timezone = _local_timezone()
        self.scheduler = AsyncIOScheduler(timezone=self.timezone)

    async def _refresh_job(self) -> None:
        try:
            report = await self.app.refresh()
            log.info("scheduled refresh: %s", report.summary())
        except Exception:  # pragma: no cover - a job must never kill the scheduler
            log.exception("scheduled refresh failed")

    async def _briefing_job(self) -> None:
        try:
            payload = await self.app.generate_briefing()
            log.info(
                "scheduled briefing: %d stories in %d sections",
                payload.get("story_count", 0),
                len(payload.get("sections", [])),
            )
        except Exception:  # pragma: no cover
            log.exception("scheduled briefing generation failed")

    async def _startup_job(self) -> None:
        """Fetch immediately on a cold start so the UI is not empty."""
        if self.app.article_repo.count() == 0:
            log.info("no articles stored; running an initial ingest")
            await self._refresh_job()
            await self._briefing_job()

    def start(self, *, run_initial: bool = True) -> None:
        ingest = self.app.config.ingest
        briefing = self.app.config.briefing

        self.scheduler.add_job(
            self._refresh_job,
            IntervalTrigger(minutes=max(5, ingest.refresh_interval_minutes)),
            id="refresh-feeds",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._briefing_job,
            CronTrigger(hour=briefing.schedule_hour, minute=briefing.schedule_minute),
            id="daily-briefing",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        if run_initial:
            self.scheduler.add_job(self._startup_job, id="startup-ingest")

        self.scheduler.start()
        log.info(
            "scheduler started: refresh every %d min, briefing at %02d:%02d %s",
            ingest.refresh_interval_minutes,
            briefing.schedule_hour,
            briefing.schedule_minute,
            self.timezone,
        )

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def jobs(self) -> list[dict]:
        return [
            {
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in self.scheduler.get_jobs()
        ]
