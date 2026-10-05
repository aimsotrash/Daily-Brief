"""Small, dependency-free text utilities shared by ingestion, analysis and search.

Deliberately plain: regex + set operations, no NLP model. Everything here has to
run over a few thousand short articles in well under a second.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'’+.#-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\s ]+(?=[\"'“(\[]?[A-Z0-9])")

#: Common English function words plus newswire filler. Kept short on purpose --
#: an aggressive stop list hurts entity-heavy news queries ("who is on the list").
STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he her here hers herself him himself his how i if in
into is isn't it its itself just me more most my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she should
shouldn't so some such than that the their theirs them themselves then there
these they this those through to too under until up very was wasn't we were
weren't what when where which while who whom why will with won't would wouldn't
you your yours yourself yourselves
said says say told according reported report reports news update latest via
""".split()
)

#: Words that carry query intent rather than topic content.
QUERY_NOISE: frozenset[str] = frozenset(
    """
what whats happening happened happens tell show give find get latest news story
stories article articles today todays yesterday week weeks month recent recently
current currently going on around biggest bigger big top main major important
key please me my i want know about anything something new update updates
briefing brief summary summarise summarize headline headlines
""".split()
)


def strip_html(value: str | None) -> str:
    """Turn an HTML fragment into readable plain text."""
    if not value:
        return ""
    text = _SCRIPT_RE.sub(" ", value)
    text = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return _WS_RE.sub(" ", value).strip()


def fold(value: str) -> str:
    """Lowercase and strip accents so 'Türkiye' and 'Turkiye' match."""
    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(value: str | None, *, drop_stopwords: bool = True) -> list[str]:
    """Split text into comparable lowercase tokens."""
    if not value:
        return []
    tokens = _WORD_RE.findall(fold(value))
    cleaned: list[str] = []
    for token in tokens:
        token = token.strip("'’.-")
        if len(token) < 2:
            continue
        if drop_stopwords and token in STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def token_set(value: str | None, *, drop_stopwords: bool = True) -> set[str]:
    return set(tokenize(value, drop_stopwords=drop_stopwords))


def sentences(value: str | None, *, min_length: int = 25) -> list[str]:
    """Split prose into sentences. Approximate but adequate for extraction."""
    if not value:
        return []
    text = normalize_whitespace(value.replace("\n", " "))
    if not text:
        return []
    raw = _SENTENCE_RE.split(text)
    out: list[str] = []
    for candidate in raw:
        candidate = candidate.strip()
        # Merge fragments that are too short to stand alone (abbreviations etc).
        if not candidate:
            continue
        if out and len(candidate) < min_length:
            out[-1] = f"{out[-1]} {candidate}"
        else:
            out.append(candidate)
    return out


def truncate(value: str, limit: int, suffix: str = "…") -> str:
    """Truncate on a word boundary."""
    value = value.strip()
    if len(value) <= limit:
        return value
    cut = value[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.-")
    return f"{cut}{suffix}"


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern[str] | None:
    """Compile a word-boundary matcher for ``term`` (already folded).

    Naive substring matching is wrong for this job: "india" occurs inside
    "Indiana", "ai" inside "said", "eu" inside "Europe". A leading boundary is
    always required; a trailing "s"/"es" is allowed so plurals still match.
    """
    term = term.strip()
    if not term:
        return None
    return re.compile(
        r"(?<![a-z0-9])" + re.escape(term) + r"(?:es|s)?(?![a-z0-9])"
    )


def contains_term(haystack_folded: str, term: str) -> bool:
    """Whether ``term`` occurs in already-folded text as a whole word/phrase."""
    pattern = _term_pattern(fold(term))
    return bool(pattern and pattern.search(haystack_folded))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(left: Iterable[str], right: Iterable[str]) -> float:
    """Overlap normalised by the *smaller* set.

    Better than Jaccard for headline matching, where one headline is often a
    near-substring of a longer one.
    """
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def simhash(tokens: Sequence[str], bits: int = 64) -> int:
    """64-bit SimHash over token hashes, for cheap near-duplicate detection.

    Returned as a *signed* 64-bit value so it round-trips through SQLite, whose
    INTEGER type is signed. :func:`hamming` masks before comparing, so the sign
    representation is invisible to callers.
    """
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(bits):
            vector[bit] += 1 if digest >> bit & 1 else -1
    value = 0
    for bit in range(bits):
        if vector[bit] > 0:
            value |= 1 << bit
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


def hamming(left: int, right: int) -> int:
    return ((left ^ right) & ((1 << 64) - 1)).bit_count()


def title_key(title: str) -> str:
    """Stable key for exact-ish title matching across feeds."""
    tokens = sorted(set(tokenize(title)))
    return " ".join(tokens)


def content_hash(*parts: str | None) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(fold(normalize_whitespace(part)).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


_ACRONYM_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}\b")


def extract_entities(text: str, limit: int = 12) -> list[str]:
    """Cheap proper-noun/acronym extraction.

    No NER model. Catches 'NVIDIA', 'OpenAI', 'Narendra Modi', 'European Union'.
    Good enough to boost entity-centric retrieval; never used as ground truth.
    """
    if not text:
        return []
    found: dict[str, int] = {}
    for match in _ACRONYM_RE.finditer(text):
        token = match.group(0)
        if 2 <= len(token) <= 24:
            found[token] = found.get(token, 0) + 2
    # Skip the first word of each sentence: sentence-initial capitals are noise.
    for sentence in sentences(text, min_length=1):
        body = sentence.split(" ", 1)[1] if " " in sentence else ""
        for match in _PROPER_RE.finditer(body):
            token = match.group(0).strip()
            if token.lower() in STOPWORDS:
                continue
            found[token] = found.get(token, 0) + 1
    ranked = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked[:limit]]
