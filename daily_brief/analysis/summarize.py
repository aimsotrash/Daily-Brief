"""Extractive summarization.

This is the deterministic engine that runs when no LLM is configured or when the
configured one is unreachable. It only ever *selects* sentences that exist in the
retrieved article, so it cannot invent a claim -- which makes it a safe default
for a product whose main risk is fabricated news.

Scoring is a small TextRank-flavoured centroid method: sentences that share
vocabulary with the rest of the document (and with the query, when there is one)
score highest, with position and length priors.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ..models import Article
from ..text import sentences, tokenize, truncate


def _sentence_scores(
    sents: Sequence[str], query_tokens: set[str] | None = None
) -> list[float]:
    if not sents:
        return []
    token_lists = [tokenize(s) for s in sents]
    document = Counter()
    for tokens in token_lists:
        document.update(set(tokens))
    total_docs = max(1, len(token_lists))

    # Inverse sentence frequency keeps boilerplate from dominating.
    idf = {
        token: math.log(1 + total_docs / (1 + count))
        for token, count in document.items()
    }

    scores: list[float] = []
    for index, tokens in enumerate(token_lists):
        if not tokens:
            scores.append(0.0)
            continue
        unique = set(tokens)
        centroid = sum(idf.get(t, 0.0) * document[t] for t in unique) / math.sqrt(
            len(unique)
        )
        position = 1.0 / (1.0 + 0.35 * index)  # leads carry the news
        length = len(tokens)
        length_prior = 1.0 if 8 <= length <= 45 else (0.55 if length < 8 else 0.8)
        query_bonus = 0.0
        if query_tokens:
            overlap = len(unique & query_tokens)
            query_bonus = min(1.5, 0.4 * overlap)
        scores.append(centroid * position * length_prior + query_bonus * centroid * 0.5)
    return scores


def summarize_article(
    article: Article, *, max_sentences: int = 2, query: str | None = None, max_chars: int = 400
) -> str:
    """A short extractive summary of one article.

    Falls back to the feed-provided summary when the body is too thin to
    extract from -- never to invented text.
    """
    body = article.best_text
    sents = sentences(body)
    if not sents:
        return truncate(article.summary or article.title, max_chars)
    if len(sents) <= max_sentences:
        return truncate(" ".join(sents), max_chars)

    query_tokens = set(tokenize(query)) if query else None
    scores = _sentence_scores(sents, query_tokens)
    ranked = sorted(range(len(sents)), key=lambda i: -scores[i])[:max_sentences]
    ranked.sort()  # restore document order for readability
    return truncate(" ".join(sents[i] for i in ranked), max_chars)


def summarize_group(
    articles: Sequence[Article], *, max_sentences: int = 3, query: str | None = None
) -> str:
    """Summarize a cluster of articles covering one story.

    Sentences are pooled across the cluster, then de-duplicated so that
    syndicated identical paragraphs do not fill the summary.
    """
    if not articles:
        return ""
    if len(articles) == 1:
        return summarize_article(articles[0], max_sentences=max_sentences, query=query)

    pool: list[tuple[str, str]] = []  # (sentence, source_name)
    for article in articles[:6]:
        for sentence in sentences(article.best_text)[:6]:
            pool.append((sentence, article.source_name))
    if not pool:
        return summarize_article(articles[0], max_sentences=max_sentences, query=query)

    query_tokens = set(tokenize(query)) if query else None
    scores = _sentence_scores([s for s, _ in pool], query_tokens)
    order = sorted(range(len(pool)), key=lambda i: -scores[i])

    chosen: list[str] = []
    chosen_tokens: list[set[str]] = []
    for index in order:
        sentence = pool[index][0]
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        if any(
            len(tokens & prior) / max(1, min(len(tokens), len(prior))) > 0.6
            for prior in chosen_tokens
        ):
            continue
        chosen.append(sentence)
        chosen_tokens.append(tokens)
        if len(chosen) >= max_sentences:
            break

    return truncate(" ".join(chosen), 600)
