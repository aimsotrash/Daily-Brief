"""Bias and framing analysis.

Two layers, deliberately kept apart:

**Source prior** -- the coarse, publication-level classification declared in
``config/sources.yaml``. Useful context, but it is wrong to assume every article
from a publication carries the same framing.

**Article-level analysis** -- this module. It starts from the source prior and
then moves away from it based on signals measured in the article itself:
loaded/emotive vocabulary, opinion markers, attribution quality, headline
construction, and the balance of partisan references. The result carries the
evidence (``signals``) that produced it, so the interface can show *why* a label
was assigned instead of asserting it.

Everything here is an analytical classification, never a fact, and the model
records that in :meth:`BiasAssessment.to_dict`. Where a left/right axis does not
apply -- a Linux kernel release, a GPU review -- the analyser says so rather than
inventing a position.
"""

from __future__ import annotations

import re

from ..models import Article, BiasAssessment, Confidence, Lean, Source
from ..text import fold, sentences, tokenize

# --------------------------------------------------------------------------
# Signal vocabularies. Small and inspectable on purpose.
# --------------------------------------------------------------------------

#: Emotive / evaluative words that indicate the writer is characterising rather
#: than reporting. Politically neutral by construction -- these are used to
#: measure *subjectivity*, not direction.
LOADED_TERMS = frozenset(
    """
slam slammed slams blast blasted blasts bombshell shocking shocked stunning
outrageous disgraceful scandal scandalous devastating catastrophic disaster
brutal savage destroy destroyed demolish eviscerate humiliating debacle
desperate reckless dangerous alarming chilling terrifying horrific unbelievable
absurd ridiculous laughable insane crazy massive huge enormous unprecedented
historic radical extreme extremist fringe hardline far-left far-right
so-called claims alleged allegedly purported apparent supposedly
crackdown chaos turmoil meltdown firestorm backlash fury outrage furious
hoax witch-hunt smear plot conspiracy elites establishment mainstream-media
woke agenda narrative propaganda regime puppet
""".split()
)

#: Markers that the piece is commentary rather than a news report.
OPINION_MARKERS = frozenset(
    """
opinion editorial commentary analysis column columnist viewpoint perspective
essay op-ed letter argues argue believe think should must ought sadly frankly
clearly obviously undoubtedly surely lets us we our my i
""".split()
)

_OPINION_URL_RE = re.compile(
    r"/(opinion|opinions|editorial|editorials|commentary|comment|columns?|"
    r"columnists?|analysis|voices|blogs?|perspectives?|viewpoint|op-ed|oped)(/|$)",
    re.IGNORECASE,
)
_OPINION_TITLE_RE = re.compile(
    r"^\s*(opinion|editorial|commentary|analysis|column|op-ed|review|explainer)\s*[:|—-]",
    re.IGNORECASE,
)

#: Terms whose *use* tends to track a political direction, because each side of a
#: debate prefers different vocabulary for the same referent. Presence indicates
#: framing choice; it is weak evidence individually and is only used in
#: aggregate.
LEFT_FRAMING = frozenset(
    """
undocumented reproductive-rights abortion-rights gun-violence gun-safety
climate-crisis climate-emergency systemic-racism marginalised marginalized
equity inclusive lgbtq transgender-rights healthcare-access living-wage
corporate-greed billionaires oligarch price-gouging union workers-rights
far-right white-nationalist insurrection disinformation misinformation
voter-suppression social-justice progressive
""".replace("-", " ").split()
)
RIGHT_FRAMING = frozenset(
    """
illegal-aliens illegals amnesty pro-life unborn second-amendment gun-rights
climate-alarmism radical-left socialist marxist woke cancel-culture
indoctrination groomer patriot law-and-order border-crisis invasion
big-government overreach deep-state censorship free-speech taxpayer-funded
job-creators traditional-values religious-liberty election-integrity
mainstream-media legacy-media leftist
""".replace("-", " ").split()
)

#: Party / bloc references, used to detect one-sided sourcing.
LEFT_ACTORS = frozenset(
    "democrat democrats democratic biden harris pelosi schumer aoc sanders "
    "labour progressive liberals leftwing".split()
)
RIGHT_ACTORS = frozenset(
    "republican republicans gop trump desantis mcconnell johnson vance conservative "
    "tory tories rightwing maga".split()
)

#: Sourcing-quality markers.
ATTRIBUTION_STRONG = re.compile(
    r"\b(according to|said in a statement|told reporters|court filing|"
    r"data (?:from|published)|the (?:report|study|filing|ruling) (?:said|found|shows)|"
    r"in a (?:statement|filing|press release)|confirmed (?:to|that))\b",
    re.IGNORECASE,
)
ATTRIBUTION_WEAK = re.compile(
    r"\b(sources say|sources said|people familiar|insiders? (?:say|said)|"
    r"it is understood|reportedly|rumou?red|speculation|some say|critics say|"
    r"observers say|many believe)\b",
    re.IGNORECASE,
)

#: Topics on which a left/right axis is not a meaningful description.
NON_POLITICAL_TOPICS = frozenset(
    {"linux", "open-source", "gaming", "hardware", "nvidia", "space", "science", "health"}
)


def _density(hits: int, total_tokens: int, per: int = 100) -> float:
    if total_tokens <= 0:
        return 0.0
    return hits * per / total_tokens


def _looks_political(article: Article, tokens: set[str]) -> bool:
    political_topics = {"us-politics", "geopolitics", "world", "india", "europe", "asia"}
    if political_topics & set(article.topics):
        return True
    if (LEFT_ACTORS | RIGHT_ACTORS) & tokens:
        return True
    if set(article.topics) & NON_POLITICAL_TOPICS:
        return False
    return False


def analyse_article(article: Article, source: Source | None = None) -> BiasAssessment:
    """Produce an article-level framing assessment.

    Deterministic, fast, and explainable. When an LLM is configured,
    :mod:`daily_brief.analysis.llm_bias` can refine this -- but this baseline
    always runs first so the application behaves identically without one.
    """
    source_lean = source.lean if source else Lean.UNKNOWN
    source_conf = source.confidence if source else Confidence.LOW

    title = article.title or ""
    body = article.best_text or ""
    blob = f"{title}. {body}"
    tokens = tokenize(blob, drop_stopwords=False)
    token_set = set(tokens)
    folded = fold(blob)
    total = max(1, len(tokens))

    signals: list[str] = []
    framing: list[str] = []

    # ---------------------------------------------------------------- opinion
    url = article.display_url
    is_opinion = bool(_OPINION_URL_RE.search(url)) or bool(_OPINION_TITLE_RE.match(title))
    opinion_hits = len(token_set & OPINION_MARKERS)
    if is_opinion:
        signals.append("Published in an opinion/commentary section")
        framing.append("opinion")
    elif opinion_hits >= 4:
        is_opinion = True
        framing.append("commentary-style")
        signals.append(f"First-person or argumentative language ({opinion_hits} markers)")

    if source and source.source_type in {"vendor"}:
        framing.append("first-party")
        signals.append(
            f"Published by {source.name} about its own products -- promotional interest"
        )
    if source and source.source_type == "aggregator":
        framing.append("aggregator")
        signals.append(
            "Aggregator entry -- the underlying article is published elsewhere, so "
            "source-level context does not describe it"
        )
    if source and source.lean is Lean.STATE_AFFILIATED:
        framing.append("state-affiliated")
        signals.append(f"{source.name} is state-funded media")

    # ----------------------------------------------------------- subjectivity
    loaded_hits = len(token_set & LOADED_TERMS)
    loaded_density = _density(loaded_hits, total)
    if loaded_hits:
        examples = sorted(token_set & LOADED_TERMS)[:4]
        signals.append(
            f"Emotive or evaluative wording ({loaded_hits}): {', '.join(examples)}"
        )
    if loaded_density >= 1.2:
        framing.append("emotive-language")

    title_tokens = set(tokenize(title, drop_stopwords=False))
    if title_tokens & LOADED_TERMS:
        signals.append("Headline uses charged wording")
        framing.append("charged-headline")
    if title.rstrip().endswith("?"):
        framing.append("question-headline")
        signals.append("Question headline -- implies a claim without asserting it")
    if re.search(r"[“\"'].+[”\"']", title) and len(title) > 30:
        framing.append("quote-headline")

    # -------------------------------------------------------------- sourcing
    strong = len(ATTRIBUTION_STRONG.findall(blob))
    weak = len(ATTRIBUTION_WEAK.findall(blob))
    if strong:
        signals.append(f"Named or documentary attribution ({strong})")
    if weak:
        signals.append(f"Unnamed or hedged attribution ({weak})")
        framing.append("anonymous-sourcing")

    subjectivity = min(
        1.0,
        0.30 * min(1.0, loaded_density / 2.0)
        + 0.30 * min(1.0, opinion_hits / 6.0)
        + (0.30 if is_opinion else 0.0)
        + 0.10 * min(1.0, weak / 3.0),
    )

    # ------------------------------------------------------------- direction
    political = _looks_political(article, token_set)
    if not political:
        assessment = BiasAssessment(
            lean=Lean.NOT_APPLICABLE,
            lean_score=0.0,
            confidence=Confidence.MEDIUM if article.topics else Confidence.LOW,
            subjectivity=round(subjectivity, 3),
            is_opinion=is_opinion,
            framing=sorted(set(framing)),
            signals=signals
            or ["No political framing signals found in this article"],
            method="heuristic",
            rationale=(
                "No left/right position assigned: this article does not appear to "
                "cover a politically contested subject."
            ),
            source_lean=source_lean,
        )
        return assessment

    left_frame = len(token_set & LEFT_FRAMING)
    right_frame = len(token_set & RIGHT_FRAMING)
    left_actor = len(token_set & LEFT_ACTORS)
    right_actor = len(token_set & RIGHT_ACTORS)

    if left_frame or right_frame:
        picked_l = sorted(token_set & LEFT_FRAMING)[:3]
        picked_r = sorted(token_set & RIGHT_FRAMING)[:3]
        if picked_l:
            signals.append(f"Vocabulary associated with left framing: {', '.join(picked_l)}")
        if picked_r:
            signals.append(f"Vocabulary associated with right framing: {', '.join(picked_r)}")

    frame_total = left_frame + right_frame
    frame_score = ((right_frame - left_frame) / frame_total) if frame_total else 0.0

    actor_total = left_actor + right_actor
    actor_imbalance = (
        abs(right_actor - left_actor) / actor_total if actor_total >= 3 else 0.0
    )
    if actor_total >= 3 and actor_imbalance >= 0.6:
        dominant = "right-of-centre" if right_actor > left_actor else "left-of-centre"
        signals.append(
            f"Coverage references {dominant} figures far more than the other side "
            f"({right_actor} vs {left_actor})"
        )
        framing.append("one-sided-attribution")

    # Blend: the source prior anchors, article evidence moves it.
    prior = source_lean.score if source_lean.is_political else 0.0
    prior_weight = source_conf.weight * (0.55 if source_lean.is_political else 0.0)
    evidence_weight = min(0.8, 0.25 * frame_total + 0.3 * actor_imbalance)

    if evidence_weight <= 0.01 and prior_weight <= 0.01:
        lean_score = 0.0
        confidence = Confidence.LOW
    else:
        lean_score = (prior * prior_weight + frame_score * evidence_weight) / max(
            0.01, prior_weight + evidence_weight
        )
        strength = prior_weight + evidence_weight
        confidence = (
            Confidence.HIGH
            if strength >= 1.0 and frame_total >= 2
            else Confidence.MEDIUM
            if strength >= 0.5
            else Confidence.LOW
        )

    lean = Lean.from_score(lean_score)

    if source_lean.is_political:
        signals.insert(
            0,
            f"Source prior: {source.name if source else 'source'} is classified "
            f"{source_lean} ({source_conf} confidence)",
        )
    if source_lean.is_political and lean is not source_lean:
        signals.append(
            f"Article-level signals place this piece at {lean}, away from its "
            f"source's usual {source_lean} framing"
        )

    rationale = (
        f"Blend of the source prior ({source_lean}, weight {prior_weight:.2f}) and "
        f"framing evidence measured in the text (weight {evidence_weight:.2f}). "
        "Treat as a rough indicator, not a verdict."
    )

    return BiasAssessment(
        lean=lean,
        lean_score=round(lean_score, 3),
        confidence=confidence,
        subjectivity=round(subjectivity, 3),
        is_opinion=is_opinion,
        framing=sorted(set(framing)),
        signals=signals or ["No strong framing signals detected"],
        method="heuristic",
        rationale=rationale,
        source_lean=source_lean,
    )


# --------------------------------------------------------------------------
# Coverage comparison -- "how are different sources covering this?"
# --------------------------------------------------------------------------

_BUCKET_ORDER = [
    Lean.LEFT,
    Lean.CENTER_LEFT,
    Lean.CENTER,
    Lean.CENTER_RIGHT,
    Lean.RIGHT,
    Lean.STATE_AFFILIATED,
    Lean.NOT_APPLICABLE,
    Lean.UNKNOWN,
]


def compare_coverage(articles: list[Article]) -> dict:
    """Bucket articles by framing so differences in coverage can be shown.

    Returns a structure the API and the prompt builder both consume. It reports
    only what is measurable: which outlets covered it, how they are classified,
    the actual headlines, and where framing markers differ.
    """
    buckets: dict[str, list[dict]] = {}
    for article in articles:
        bias = article.bias or BiasAssessment()
        key = str(bias.lean if bias.lean is not Lean.UNKNOWN else bias.source_lean)
        buckets.setdefault(key, []).append(
            {
                "title": article.title,
                "source": article.source_name,
                "source_id": article.source_id,
                "url": article.display_url,
                "published_at": (
                    article.published_at.isoformat() if article.published_at else None
                ),
                "framing": bias.framing,
                "is_opinion": bias.is_opinion,
                "subjectivity": bias.subjectivity,
            }
        )

    ordered = [
        {"lean": str(lean), "articles": buckets[str(lean)]}
        for lean in _BUCKET_ORDER
        if str(lean) in buckets
    ]

    political = [
        a
        for a in articles
        if (a.bias or BiasAssessment()).lean.is_political
    ]
    spread = 0.0
    if len(political) >= 2:
        scores = [(a.bias.lean_score if a.bias else 0.0) for a in political]
        spread = max(scores) - min(scores)

    # Words that appear in one camp's headlines but not the other's.
    def _headline_terms(subset: list[Article]) -> set[str]:
        terms: set[str] = set()
        for article in subset:
            terms.update(tokenize(article.title))
        return terms

    left_side = [a for a in political if a.bias and a.bias.lean_score < -0.15]
    right_side = [a for a in political if a.bias and a.bias.lean_score > 0.15]
    left_terms = _headline_terms(left_side)
    right_terms = _headline_terms(right_side)

    return {
        "buckets": ordered,
        "source_count": len({a.source_id for a in articles}),
        "article_count": len(articles),
        "lean_spread": round(spread, 3),
        "distinct_left_terms": sorted(left_terms - right_terms)[:12] if right_side else [],
        "distinct_right_terms": sorted(right_terms - left_terms)[:12] if left_side else [],
        "opinion_count": sum(1 for a in articles if a.bias and a.bias.is_opinion),
        "disclaimer": (
            "Lean labels are analytical classifications generated by Daily-Brief "
            "from source metadata and article text, not statements of fact."
        ),
    }


def lean_badge(article: Article) -> dict:
    """The display-ready framing label for one article.

    Callers render this next to an outlet name in coverage lists. It exists so
    the UI never has to re-derive the rules, and so one specific distinction is
    always preserved: ``basis`` says whether the label came from analysing *this
    article* or is merely the publication-wide prior. Showing a source prior as
    though it were article-level analysis would misrepresent it.

    ``basis`` values:

    ``article``
        Produced by analysing this article's text (including a deliberate
        "no political framing here" verdict, which is a finding, not an absence).
    ``source``
        Article analysis reached no conclusion, so the publication's prior is
        shown instead — clearly marked as such.
    ``none``
        Nothing applicable is known.
    """
    bias = article.bias
    if bias is None:
        return {
            "lean": str(Lean.UNKNOWN),
            "basis": "none",
            "confidence": str(Confidence.LOW),
            "state_affiliated": False,
            "is_opinion": False,
        }

    if bias.lean is Lean.UNKNOWN and bias.source_lean not in {
        Lean.UNKNOWN,
        Lean.NOT_APPLICABLE,
    }:
        lean, basis = bias.source_lean, "source"
    elif bias.lean is Lean.UNKNOWN:
        lean, basis = Lean.UNKNOWN, "none"
    else:
        lean, basis = bias.lean, "article"

    return {
        "lean": str(lean),
        "basis": basis,
        "confidence": str(bias.confidence),
        "state_affiliated": bias.source_lean is Lean.STATE_AFFILIATED,
        "is_opinion": bias.is_opinion,
    }


def key_sentences(article: Article, limit: int = 2) -> list[str]:
    """The most framing-relevant sentences, for showing evidence in the UI."""
    scored: list[tuple[float, str]] = []
    for sentence in sentences(article.best_text)[:20]:
        tokens = set(tokenize(sentence, drop_stopwords=False))
        score = (
            2.0 * len(tokens & LOADED_TERMS)
            + 1.0 * len(tokens & (LEFT_FRAMING | RIGHT_FRAMING))
            + (1.0 if ATTRIBUTION_WEAK.search(sentence) else 0.0)
        )
        if score:
            scored.append((score, sentence))
    scored.sort(key=lambda item: -item[0])
    return [sentence for _, sentence in scored[:limit]]
