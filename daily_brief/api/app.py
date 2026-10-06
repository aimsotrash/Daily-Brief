"""FastAPI application: JSON API plus the static frontend.

One process serves everything. The API is small and entirely CRUD-plus-two-verbs;
the interesting logic lives in the services this layer calls.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..analysis.relevance import parse_interests
from ..config import Config
from ..scheduler import BriefingScheduler
from ..service import Application
from .schemas import (
    OnboardingRequest,
    PreferencesResponse,
    RefreshRequest,
    RefreshResponse,
    SearchRequest,
    SearchResponse,
)

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def create_app(
    config: Config | None = None,
    *,
    application: Application | None = None,
    enable_scheduler: bool = True,
    run_initial_ingest: bool = True,
) -> FastAPI:
    app_state: dict = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service = application or Application(config)
        app_state["service"] = service
        scheduler: BriefingScheduler | None = None
        if enable_scheduler:
            scheduler = BriefingScheduler(service)
            scheduler.start(run_initial=run_initial_ingest)
            app_state["scheduler"] = scheduler
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown()
            if application is None:
                await service.aclose()

    api = FastAPI(
        title="Daily-Brief",
        version="0.1.0",
        summary="Self-hosted, bias-aware personal news briefing and research",
        lifespan=lifespan,
    )

    def service() -> Application:
        svc = app_state.get("service")
        if svc is None:  # pragma: no cover - only if called outside lifespan
            raise HTTPException(503, "application is still starting")
        return svc

    # ------------------------------------------------------------ preferences
    @api.get("/api/preferences", response_model=PreferencesResponse)
    def get_preferences() -> PreferencesResponse:
        svc = service()
        prefs = svc.preferences.load()
        return PreferencesResponse(
            onboarded=prefs.onboarded,
            interests=prefs.interests,
            raw_interests_text=prefs.raw_interests_text,
            parsed=[i.to_dict() for i in svc.preferences.interests()],
            updated_at=prefs.updated_at.isoformat(),
        )

    @api.post("/api/preferences", response_model=PreferencesResponse)
    def set_preferences(payload: OnboardingRequest) -> PreferencesResponse:
        svc = service()
        raw = payload.interests
        if isinstance(raw, str) and not raw.strip():
            raise HTTPException(400, "interests cannot be empty")
        if isinstance(raw, list) and not any(str(x).strip() for x in raw):
            raise HTTPException(400, "interests cannot be empty")
        prefs = svc.preferences.save_interests(raw)
        if not prefs.interests:
            raise HTTPException(
                400, "could not interpret any interests from that input"
            )
        return PreferencesResponse(
            onboarded=prefs.onboarded,
            interests=prefs.interests,
            raw_interests_text=prefs.raw_interests_text,
            parsed=[i.to_dict() for i in svc.preferences.interests()],
            updated_at=prefs.updated_at.isoformat(),
        )

    @api.post("/api/preferences/preview")
    def preview_preferences(payload: OnboardingRequest) -> dict:
        """Interpret interests without saving them, so Settings can show the
        result while the user is still typing."""
        return {"parsed": [i.to_dict() for i in parse_interests(payload.interests)]}

    @api.post("/api/preferences/reset", response_model=PreferencesResponse)
    def reset_preferences() -> PreferencesResponse:
        svc = service()
        prefs = svc.preferences.reset()
        return PreferencesResponse(
            onboarded=prefs.onboarded,
            interests=prefs.interests,
            raw_interests_text=prefs.raw_interests_text,
            parsed=[],
            updated_at=prefs.updated_at.isoformat(),
        )

    # ---------------------------------------------------------------- search
    @api.post("/api/search", response_model=SearchResponse)
    async def search(payload: SearchRequest) -> SearchResponse:
        svc = service()
        session_id = payload.session_id or svc.search.new_session_id()
        result = await svc.search.search(
            payload.query,
            session_id=session_id,
            interests=svc.interests(),
            use_history=payload.use_history,
        )
        return SearchResponse(session_id=session_id, **result.to_dict())

    @api.get("/api/search/history")
    def search_history(session_id: str = Query(...)) -> dict:
        return {"session_id": session_id, "messages": service().search.history(session_id)}

    @api.delete("/api/search/history")
    def clear_history(session_id: str = Query(...)) -> dict:
        service().search.clear(session_id)
        return {"ok": True}

    # -------------------------------------------------------------- briefing
    @api.get("/api/briefing")
    async def get_briefing(refresh: bool = Query(False)) -> dict:
        svc = service()
        if refresh:
            return await svc.generate_briefing()
        return await svc.get_briefing()

    @api.post("/api/briefing/generate")
    async def generate_briefing() -> dict:
        return await service().generate_briefing()

    # ---------------------------------------------------------------- ingest
    @api.post("/api/refresh", response_model=RefreshResponse)
    async def refresh(payload: RefreshRequest = Body(default=RefreshRequest())) -> RefreshResponse:
        report = await service().refresh(payload.source_ids)
        return RefreshResponse(
            summary=report.summary(),
            inserted=report.inserted,
            updated=report.updated,
            duplicates=report.duplicates_exact + report.duplicates_near,
            clusters=report.clusters,
            sources_ok=report.sources_ok,
            sources_failed=report.sources_failed,
            sources_total=report.sources_total,
            failures=[
                {"source": r.source_name, "error": r.error}
                for r in report.per_source
                if r.error
            ][:20],
        )

    # --------------------------------------------------------------- sources
    @api.get("/api/sources")
    def list_sources() -> dict:
        svc = service()
        state = svc.source_repo.all_feed_state()
        return {
            "methodology": svc.source_methodology,
            "sources": [
                {
                    **source.as_context(),
                    "url": source.url,
                    "categories": source.categories,
                    "enabled": source.enabled,
                    "last_success_at": state.get(source.id, {}).get("last_success_at"),
                    "last_error": state.get(source.id, {}).get("last_error"),
                    "consecutive_failures": state.get(source.id, {}).get(
                        "consecutive_failures", 0
                    ),
                }
                for source in svc.source_repo.all()
            ],
        }

    @api.get("/api/status")
    async def status() -> dict:
        svc = service()
        payload = await svc.status()
        scheduler = app_state.get("scheduler")
        payload["scheduler"] = scheduler.jobs() if scheduler else []
        return payload

    @api.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    # -------------------------------------------------------------- frontend
    if WEB_DIR.is_dir():
        api.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @api.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    else:  # pragma: no cover - only if the package is installed without web/

        @api.get("/", include_in_schema=False)
        def index() -> JSONResponse:
            return JSONResponse(
                {"error": "frontend assets not found", "expected": str(WEB_DIR)}, 500
            )

    return api
