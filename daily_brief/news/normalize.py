"""Normalization: raw feed entries -> :class:`Article` objects.

Also owns URL canonicalization, which is the single most effective
de-duplication measure: the same story is routinely syndicated with different
tracking parameters.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..models import Article, Source, utcnow
from ..text import content_hash, extract_entities, normalize_whitespace, simhash, tokenize, truncate
from .parser import RawEntry

#: Query parameters that never identify a document.
TRACKING_PARAMS = re.compile(
    r"^(utm_\w+|fbclid|gclid|gbraid|wbraid|msclkid|mc_[ce]id|igshid|ref|ref_src|"
    r"referrer|source|cmpid|CMP|ncid|smid|partner|at_medium|at_campaign|"
    r"guccounter|guce_\w+|__twitter_impression|amp|spm|share|sh)$",
    re.IGNORECASE,
)

_AMP_SUFFIX = re.compile(r"/amp/?$|\.amp$|/amp\.html$", re.IGNORECASE)


def canonical_url(url: str) -> str:
    """Strip tracking noise so the same article resolves to one key."""
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parts = urlparse(url)
    except ValueError:
        return url
    if not parts.scheme or not parts.netloc:
        return url

    scheme = "https" if parts.scheme in {"http", "https"} else parts.scheme
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Drop default ports.
    netloc = re.sub(r":(80|443)$", "", netloc)

    path = _AMP_SUFFIX.sub("", parts.path) or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if not TRACKING_PARAMS.match(k)]
    query = urlencode(sorted(kept))

    return urlunparse((scheme, netloc, path, "", query, ""))


def normalize_entry(entry: RawEntry, source: Source) -> Article | None:
    """Convert one raw feed entry into an :class:`Article`.

    Returns ``None`` when the entry lacks the minimum viable metadata (a title
    and a resolvable link). Everything else -- missing dates, missing summaries,
    missing authors -- is tolerated and left empty.
    """
    title = normalize_whitespace(entry.title)
    url = (entry.link or "").strip()
    if not title or not url.lower().startswith(("http://", "https://")):
        return None

    canonical = canonical_url(url)
    summary = normalize_whitespace(entry.summary)
    content = normalize_whitespace(entry.content)

    # Some feeds put the whole body in <description> and nothing in <content>.
    if not content and len(summary) > 400:
        content, summary = summary, truncate(summary, 320)
    if not summary and content:
        summary = truncate(content, 320)

    body_for_analysis = f"{title}. {content or summary}"
    article = Article(
        url=url,
        canonical_url=canonical,
        title=truncate(title, 300, suffix=""),
        source_id=source.id,
        source_name=source.name,
        summary=summary[:2000],
        content=content[:12000],
        author=normalize_whitespace(entry.author)[:160],
        published_at=entry.published,
        retrieved_at=utcnow(),
        language=(entry.language or "").split("-")[0].lower()[:8],
        image_url=entry.image_url[:600],
        feed_categories=list(entry.categories),
        entities=extract_entities(body_for_analysis),
    )
    article.content_hash = content_hash(article.title, canonical)
    article.simhash = simhash(tokenize(f"{title} {summary}"))
    return article


def normalize_entries(entries: list[RawEntry], source: Source, limit: int) -> list[Article]:
    """Normalize a feed's entries, newest first, capped at ``limit``."""
    articles: list[Article] = []
    for entry in entries:
        article = normalize_entry(entry, source)
        if article is not None:
            articles.append(article)

    # Newest first; entries without a date sort last but are kept.
    articles.sort(
        key=lambda a: a.published_at.timestamp() if a.published_at else 0.0,
        reverse=True,
    )
    return articles[:limit] if limit > 0 else articles
