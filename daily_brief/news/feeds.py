"""Feed fetching over HTTP.

Uses conditional requests (ETag / If-Modified-Since) so repeated polling is cheap
and polite, bounds concurrency, and never raises into the ingestion loop -- an
unavailable source produces a failed :class:`FetchResult`, not an exception.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

import httpx

from ..config import IngestConfig
from ..models import Source

log = logging.getLogger(__name__)

ACCEPT = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, "
    "text/xml;q=0.9, */*;q=0.5"
)


@dataclass(slots=True)
class FetchResult:
    source: Source
    status: str  # "ok" | "not-modified" | "error"
    body: bytes = b""
    etag: str | None = None
    last_modified: str | None = None
    http_status: int | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class FeedFetcher:
    """Fetches feed documents. Injectable so ingestion can be tested offline."""

    def __init__(self, config: IngestConfig) -> None:
        self.config = config

    async def fetch_many(
        self,
        sources: Sequence[Source],
        state: dict[str, dict] | None = None,
    ) -> list[FetchResult]:
        if not sources:
            return []
        state = state or {}
        semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        limits = httpx.Limits(
            max_connections=max(1, self.config.concurrency),
            max_keepalive_connections=max(1, self.config.concurrency),
        )
        headers = {"User-Agent": self.config.user_agent, "Accept": ACCEPT}

        async with httpx.AsyncClient(
            timeout=self.config.fetch_timeout_seconds,
            follow_redirects=True,
            headers=headers,
            limits=limits,
        ) as client:

            async def run(source: Source) -> FetchResult:
                async with semaphore:
                    return await self._fetch_one(client, source, state.get(source.id, {}))

            return list(await asyncio.gather(*(run(s) for s in sources)))

    async def _fetch_one(
        self, client: httpx.AsyncClient, source: Source, state: dict
    ) -> FetchResult:
        conditional: dict[str, str] = {}
        if state.get("etag"):
            conditional["If-None-Match"] = state["etag"]
        if state.get("last_modified"):
            conditional["If-Modified-Since"] = state["last_modified"]

        try:
            response = await client.get(source.url, headers=conditional)
        except httpx.HTTPError as exc:
            log.warning("fetch failed for %s: %s", source.id, exc)
            return FetchResult(
                source=source, status="error", error=f"{type(exc).__name__}: {exc}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("unexpected fetch error for %s: %s", source.id, exc)
            return FetchResult(source=source, status="error", error=str(exc))

        etag = response.headers.get("etag")
        last_modified = response.headers.get("last-modified")

        if response.status_code == 304:
            return FetchResult(
                source=source,
                status="not-modified",
                etag=etag,
                last_modified=last_modified,
                http_status=304,
            )
        if response.status_code >= 400:
            return FetchResult(
                source=source,
                status="error",
                http_status=response.status_code,
                error=f"HTTP {response.status_code}",
            )
        if not response.content:
            return FetchResult(
                source=source,
                status="error",
                http_status=response.status_code,
                error="empty response body",
            )

        return FetchResult(
            source=source,
            status="ok",
            body=response.content,
            etag=etag,
            last_modified=last_modified,
            http_status=response.status_code,
        )
