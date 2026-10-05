"""Feed parsing: RSS 2.0, Atom 1.0 and RDF/RSS 1.0 into a common entry shape.

Hand-rolled on ``xml.etree`` rather than pulling in a feed library. The formats
are simple, the standard library covers them, and doing it here means malformed
input is handled the way this application wants -- salvage what parses, drop what
does not, never raise into the ingestion loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree as ET

from dateutil import parser as date_parser

from ..text import normalize_whitespace, strip_html

log = logging.getLogger(__name__)

ATOM = "http://www.w3.org/2005/Atom"
RSS10 = "http://purl.org/rss/1.0/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DC = "http://purl.org/dc/elements/1.1/"
CONTENT = "http://purl.org/rss/1.0/modules/content/"
MEDIA = "http://search.yahoo.com/mrss/"

_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>")
# Control characters that are illegal in XML 1.0 but that feeds emit anyway.
_BAD_CHARS_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f﷐-﷯￾￿]"
)


class FeedParseError(Exception):
    """Raised when a document cannot be parsed as any supported feed format."""


@dataclass(slots=True)
class RawEntry:
    """One item as it appeared in a feed, before normalization."""

    title: str = ""
    link: str = ""
    summary: str = ""
    content: str = ""
    author: str = ""
    published: datetime | None = None
    guid: str = ""
    categories: list[str] = field(default_factory=list)
    image_url: str = ""
    language: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedFeed:
    title: str = ""
    link: str = ""
    language: str = ""
    entries: list[RawEntry] = field(default_factory=list)
    #: Entries that were present but unusable (no title or no link).
    skipped: int = 0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    # itertext() catches feeds that wrap text in inline markup.
    return normalize_whitespace("".join(element.itertext()))


def _find(parent: ET.Element, *names: str) -> ET.Element | None:
    """Namespace-insensitive child lookup by local name."""
    wanted = {n.lower() for n in names}
    for child in parent:
        if _local(child.tag).lower() in wanted:
            return child
    return None


def _find_all(parent: ET.Element, *names: str) -> list[ET.Element]:
    wanted = {n.lower() for n in names}
    return [c for c in parent if _local(c.tag).lower() in wanted]


def parse_date(value: str | None) -> datetime | None:
    """Parse any of the date formats feeds use in practice.

    Returns ``None`` rather than raising: a missing or unparseable date is
    ordinary, and downstream code treats it as "publication time unknown".
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        parsed = date_parser.parse(value, fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    result = parsed.astimezone(timezone.utc)
    # Feeds occasionally carry absurd dates; treat those as unknown.
    year = result.year
    if year < 1990 or year > datetime.now(timezone.utc).year + 2:
        return None
    return result


def _clean_bytes(payload: bytes) -> bytes:
    """Remove a leading BOM and illegal control characters."""
    if payload.startswith(b"\xef\xbb\xbf"):
        payload = payload[3:]
    payload = payload.lstrip()
    return payload


def _decode(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _to_root(document: bytes | str) -> ET.Element:
    if isinstance(document, str):
        payload = document.encode("utf-8", errors="replace")
    else:
        payload = document
    payload = _clean_bytes(payload)
    if not payload:
        raise FeedParseError("empty document")

    try:
        return ET.fromstring(payload)
    except ET.ParseError as first_error:
        # Second chance: drop illegal control characters and the encoding
        # declaration (which would now be wrong), then re-encode as UTF-8.
        text = _BAD_CHARS_RE.sub("", _decode(payload))
        text = _XML_DECL_RE.sub("", text).lstrip()
        try:
            return ET.fromstring(text.encode("utf-8"))
        except ET.ParseError:
            raise FeedParseError(f"malformed XML: {first_error}") from first_error


def _entry_categories(item: ET.Element) -> list[str]:
    out: list[str] = []
    for node in _find_all(item, "category", "subject"):
        # Atom puts it in @term, RSS in the element text.
        value = (node.get("term") or node.get("label") or _text(node)).strip()
        if value and value.lower() not in {c.lower() for c in out}:
            out.append(value[:60])
    return out[:12]


def _entry_image(item: ET.Element) -> str:
    for node in _find_all(item, "thumbnail", "content"):
        if node.tag.startswith(f"{{{MEDIA}}}"):
            url = node.get("url", "")
            if url:
                return url
    enclosure = _find(item, "enclosure")
    if enclosure is not None:
        mime = enclosure.get("type", "")
        if mime.startswith("image/"):
            return enclosure.get("url", "")
    for node in _find_all(item, "link"):
        if node.get("rel") == "enclosure" and node.get("type", "").startswith("image/"):
            return node.get("href", "")
    return ""


def _atom_link(entry: ET.Element) -> str:
    """Pick the best link from an Atom entry: rel=alternate, else first href."""
    fallback = ""
    for node in _find_all(entry, "link"):
        href = node.get("href", "").strip()
        if not href:
            continue
        rel = node.get("rel", "alternate")
        if rel == "alternate":
            return href
        if not fallback and rel not in {"self", "edit", "replies"}:
            fallback = href
    return fallback


def _parse_entry(item: ET.Element, *, atom: bool) -> RawEntry | None:
    title = _text(_find(item, "title"))

    link = ""
    if atom:
        link = _atom_link(item)
    if not link:
        link_node = _find(item, "link")
        if link_node is not None:
            link = (link_node.get("href") or _text(link_node)).strip()
    guid_node = _find(item, "guid", "id")
    guid = _text(guid_node) if guid_node is not None else ""
    if not link and guid.lower().startswith(("http://", "https://")):
        link = guid

    if not title or not link:
        return None

    summary_node = _find(item, "description", "summary", "subtitle")
    content_node = _find(item, "encoded", "content", "content:encoded")
    summary = strip_html(_text(summary_node))
    content = strip_html(_text(content_node))
    # Atom <content> and RSS <description> are sometimes the same string.
    if content and summary and content.startswith(summary[: max(40, len(summary) // 2)]):
        summary = summary if len(summary) < len(content) else ""

    author = ""
    author_node = _find(item, "author", "creator", "dc:creator")
    if author_node is not None:
        name_node = _find(author_node, "name")
        author = _text(name_node) if name_node is not None else _text(author_node)

    published = None
    for tag in ("published", "pubdate", "updated", "date", "modified", "created"):
        node = _find(item, tag)
        if node is not None:
            published = parse_date(_text(node))
            if published:
                break

    return RawEntry(
        title=title,
        link=link,
        summary=summary,
        content=content,
        author=author[:160],
        published=published,
        guid=guid or link,
        categories=_entry_categories(item),
        image_url=_entry_image(item),
    )


def parse_feed(document: bytes | str) -> ParsedFeed:
    """Parse a feed document. Raises :class:`FeedParseError` if nothing parses."""
    root = _to_root(document)
    root_name = _local(root.tag).lower()

    if root_name == "rss":
        channel = _find(root, "channel")
        if channel is None:
            raise FeedParseError("<rss> element has no <channel>")
        items = _find_all(channel, "item")
        container, atom = channel, False
    elif root_name == "feed":
        container, items, atom = root, _find_all(root, "entry"), True
    elif root_name == "rdf":
        channel = _find(root, "channel")
        # RSS 1.0 puts <item> as siblings of <channel>, under the RDF root.
        items = _find_all(root, "item")
        container, atom = channel if channel is not None else root, False
    elif root_name == "channel":
        # Some feeds omit the <rss> wrapper entirely.
        container, items, atom = root, _find_all(root, "item"), False
    else:
        raise FeedParseError(f"unsupported feed root element <{root_name}>")

    feed = ParsedFeed(
        title=_text(_find(container, "title")),
        language=_text(_find(container, "language")) or root.get(
            "{http://www.w3.org/XML/1998/namespace}lang", ""
        ),
    )
    link_node = _find(container, "link")
    if link_node is not None:
        feed.link = (link_node.get("href") or _text(link_node)).strip()

    for item in items:
        try:
            entry = _parse_entry(item, atom=atom)
        except Exception as exc:  # a single bad item must not kill the feed
            log.debug("skipping unparseable feed item: %s", exc)
            feed.skipped += 1
            continue
        if entry is None:
            feed.skipped += 1
            continue
        entry.language = entry.language or feed.language
        feed.entries.append(entry)

    return feed
