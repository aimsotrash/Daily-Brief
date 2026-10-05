"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OnboardingRequest(BaseModel):
    interests: str | list[str] = Field(
        ...,
        description="Free text, e.g. 'AI, Linux, gaming and what's happening in India'.",
    )


class PreferencesResponse(BaseModel):
    onboarded: bool
    interests: list[str]
    raw_interests_text: str
    parsed: list[dict[str, Any]] = []
    updated_at: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="A natural-language question about the news.")
    session_id: str | None = Field(
        None, description="Opaque conversation id; omit to start a new one."
    )
    use_history: bool = True


class SearchResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict[str, Any]]
    engine: str
    grounded: bool
    has_results: bool
    intent: str
    query: dict[str, Any] = {}
    comparison: dict[str, Any] | None = None
    warnings: list[str] = []
    cited: list[int] = []


class RefreshRequest(BaseModel):
    source_ids: list[str] | None = None


class RefreshResponse(BaseModel):
    summary: str
    inserted: int
    updated: int
    duplicates: int
    clusters: int
    sources_ok: int
    sources_failed: int
    sources_total: int
    failures: list[dict[str, Any]] = []
