"""Loading the source registry.

The registry is data, not code: no module in this application references a
specific publication. Adding or removing a source is a YAML edit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from ..models import Confidence, Lean, Source

log = logging.getLogger(__name__)


class RegistryError(Exception):
    """Raised when the source registry cannot be used at all."""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def parse_registry(data: dict[str, Any]) -> tuple[list[Source], dict[str, Any]]:
    """Turn parsed YAML into :class:`Source` objects plus the methodology block.

    Individual malformed entries are skipped with a warning rather than failing
    the whole load -- one bad hand-edited entry should not take the app down.
    """
    if not isinstance(data, dict):
        raise RegistryError("source registry must be a YAML mapping")

    defaults: dict[str, Any] = data.get("defaults") or {}
    methodology: dict[str, Any] = data.get("methodology") or {}
    entries = data.get("sources")
    if not isinstance(entries, list):
        raise RegistryError("source registry must contain a 'sources' list")

    sources: list[Source] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            log.warning("registry entry #%d is not a mapping; skipped", index)
            continue
        merged = {**defaults, **raw}
        source_id = str(merged.get("id") or "").strip()
        url = str(merged.get("url") or "").strip()
        name = str(merged.get("name") or source_id).strip()
        if not source_id or not url:
            log.warning("registry entry #%d is missing 'id' or 'url'; skipped", index)
            continue
        if not url.lower().startswith(("http://", "https://")):
            log.warning("source %r has a non-HTTP url %r; skipped", source_id, url)
            continue
        if source_id in seen:
            log.warning("duplicate source id %r; keeping the first", source_id)
            continue
        seen.add(source_id)
        sources.append(
            Source(
                id=source_id,
                name=name,
                url=url,
                site=str(merged.get("site") or "").strip(),
                categories=_as_list(merged.get("categories")),
                lean=Lean.parse(merged.get("lean")),
                confidence=Confidence.parse(merged.get("confidence")),
                source_type=str(merged.get("source_type") or "").strip(),
                country=str(merged.get("country") or "").strip(),
                ownership=str(merged.get("ownership") or "").strip(),
                notes=" ".join(str(merged.get("notes") or "").split()),
                enabled=bool(merged.get("enabled", True)),
            )
        )

    if not sources:
        raise RegistryError("source registry contains no usable sources")
    return sources, methodology


def load_registry(path: str | Path) -> tuple[list[Source], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"source registry not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"could not parse {path}: {exc}") from exc
    sources, methodology = parse_registry(data)
    log.info(
        "loaded %d sources (%d enabled) from %s",
        len(sources),
        sum(1 for s in sources if s.enabled),
        path,
    )
    return sources, methodology
