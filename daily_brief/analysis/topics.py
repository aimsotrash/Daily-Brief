"""Topic classification and the shared topic lexicon.

A keyword lexicon rather than a trained classifier. It is inspectable, editable,
instant, needs no model, and for a small set of broad news categories it is
competitive with anything heavier. ``LEXICON`` is also what turns a user's
free-text interests into matchable terms, so the two stay consistent by
construction.
"""

from __future__ import annotations

from collections import defaultdict

from ..models import Article
from ..text import contains_term, fold, tokenize

#: canonical topic -> terms that indicate it. Multi-word terms are matched as
#: substrings of the folded text; single words are matched against tokens.
LEXICON: dict[str, list[str]] = {
    "ai": [
        "ai", "a.i.", "artificial intelligence", "machine learning", "deep learning",
        "neural network", "llm", "large language model", "generative ai", "genai",
        "chatgpt", "gpt", "openai", "anthropic", "claude", "gemini", "deepmind",
        "llama", "mistral", "hugging face", "transformer", "diffusion model",
        "stable diffusion", "midjourney", "copilot", "inference", "fine-tuning",
        "agentic", "chatbot", "foundation model", "multimodal", "rag", "embeddings",
        "perplexity ai", "xai", "grok", "training data", "alignment", "agi",
    ],
    "linux": [
        "linux", "kernel", "ubuntu", "debian", "fedora", "arch linux", "gnome", "kde",
        "plasma", "systemd", "wayland", "x11", "xorg", "distro", "distribution",
        "gnu", "bash", "btrfs", "ext4", "zfs", "mesa", "steamos", "nixos", "opensuse",
        "red hat", "rhel", "centos", "linus torvalds", "flatpak", "snap", "appimage",
        "pipewire", "proton", "wine", "immutable", "kde plasma",
    ],
    "open-source": [
        "open source", "open-source", "foss", "gpl", "apache license", "mit license",
        "free software", "copyleft", "fork", "upstream", "maintainer", "git",
        "github", "gitlab", "contributor", "cla", "sbom",
    ],
    "gaming": [
        "game", "games", "gaming", "gamer", "playstation", "ps5", "xbox", "nintendo",
        "switch", "steam", "valve", "epic games", "esports", "console", "rpg", "fps",
        "mmo", "indie game", "dlc", "expansion", "speedrun", "twitch", "steam deck",
        "game pass", "ubisoft", "activision", "blizzard", "ea sports", "rockstar",
        "bethesda", "gta", "minecraft", "fortnite", "developer studio", "playtest",
    ],
    "hardware": [
        "cpu", "gpu", "chip", "chipset", "semiconductor", "silicon", "processor",
        "nvidia", "amd", "intel", "arm", "risc-v", "tsmc", "foundry", "nanometer",
        "ddr5", "gddr", "hbm", "motherboard", "ssd", "nvme", "ram", "overclock",
        "benchmark", "graphics card", "rtx", "radeon", "ryzen", "snapdragon",
        "apple silicon", "data center", "datacenter", "rack", "cooling", "wafer",
    ],
    "nvidia": [
        "nvidia", "geforce", "rtx", "cuda", "jensen huang", "dgx", "tensor core",
        "blackwell", "hopper", "grace", "h100", "h200", "b200", "gb200", "nvlink",
    ],
    "technology": [
        "technology", "tech", "software", "app", "platform", "startup", "silicon valley",
        "cloud", "saas", "api", "developer", "smartphone", "iphone", "android",
        "apple", "google", "microsoft", "amazon", "meta", "tesla", "spacex", "internet",
        "browser", "web", "quantum computing", "data center", "streaming", "5g",
    ],
    "security": [
        "security", "cybersecurity", "hack", "hacker", "breach", "ransomware",
        "malware", "vulnerability", "cve", "exploit", "zero-day", "phishing",
        "encryption", "privacy", "surveillance", "spyware", "data leak", "patch",
        "backdoor", "authentication", "botnet", "ddos", "infosec",
    ],
    "science": [
        "science", "research", "study", "scientists", "physics", "chemistry",
        "biology", "genome", "climate science", "experiment", "peer-reviewed",
        "nature journal", "telescope", "particle", "fusion", "quantum",
    ],
    "space": [
        "space", "nasa", "spacex", "rocket", "satellite", "orbit", "launch", "mars",
        "moon", "lunar", "asteroid", "esa", "isro", "starship", "falcon 9", "artemis",
        "telescope", "astronaut", "spacecraft",
    ],
    "business": [
        "business", "company", "revenue", "earnings", "profit", "acquisition",
        "merger", "ipo", "startup", "funding", "venture capital", "layoffs", "ceo",
        "shares", "valuation", "quarterly", "investors", "antitrust", "regulator",
    ],
    "markets": [
        "market", "markets", "stocks", "stock", "nasdaq", "s&p", "dow jones", "bond",
        "yield", "inflation", "interest rate", "federal reserve", "recession",
        "currency", "dollar", "rupee", "euro", "crypto", "bitcoin", "ethereum",
        "commodities", "oil price", "gdp", "tariff", "tariffs", "trade deal",
    ],
    "us-politics": [
        "white house", "congress", "senate", "house of representatives", "president",
        "republican", "democrat", "gop", "biden", "trump", "supreme court", "capitol",
        "governor", "election", "primary", "campaign", "impeachment", "filibuster",
        "legislation", "bill", "federal", "washington", "doj", "fbi", "pentagon",
        "immigration", "border", "midterms", "electoral",
    ],
    "geopolitics": [
        "geopolitics", "diplomacy", "sanctions", "treaty", "alliance", "nato",
        "united nations", "un security council", "summit", "foreign policy",
        "ambassador", "bilateral", "ceasefire", "war", "conflict", "military",
        "invasion", "border dispute", "sovereignty", "trade war", "export controls",
        "ukraine", "russia", "china", "taiwan", "israel", "gaza", "iran",
        "north korea", "middle east", "eu", "brics", "g7", "g20",
    ],
    "world": [
        "world", "international", "global", "country", "government", "minister",
        "prime minister", "parliament", "protest", "election", "crisis", "earthquake",
        "flood", "wildfire", "refugee", "humanitarian",
    ],
    "india": [
        "india", "indian", "delhi", "new delhi", "mumbai", "bengaluru", "bangalore",
        "chennai", "kolkata", "hyderabad", "modi", "narendra modi", "bjp", "congress party",
        "lok sabha", "rajya sabha", "rupee", "rbi", "isro", "upi", "aadhaar",
        "maharashtra", "karnataka", "tamil nadu", "kerala", "uttar pradesh", "gujarat",
        "bollywood", "ipl", "cricket", "pakistan", "bangladesh", "sri lanka",
    ],
    "europe": [
        "europe", "european union", "brussels", "germany", "france", "italy", "spain",
        "poland", "netherlands", "uk", "britain", "london", "brexit", "eurozone",
        "nato", "scandinavia", "ecb",
    ],
    "asia": [
        "asia", "china", "beijing", "japan", "tokyo", "korea", "seoul", "taiwan",
        "singapore", "indonesia", "vietnam", "philippines", "thailand", "hong kong",
        "asean", "xi jinping",
    ],
    "climate": [
        "climate", "climate change", "global warming", "emissions", "carbon",
        "renewable", "solar power", "wind power", "fossil fuel", "net zero", "cop29",
        "cop30", "heatwave", "drought", "biodiversity", "sustainability", "ev",
        "electric vehicle",
    ],
    "health": [
        "health", "medical", "disease", "vaccine", "hospital", "patient", "doctor",
        "fda", "who", "outbreak", "pandemic", "cancer", "mental health", "drug trial",
        "clinical", "obesity", "public health",
    ],
    "culture": [
        "film", "movie", "music", "album", "streaming show", "netflix", "hbo",
        "celebrity", "art", "book", "author", "festival", "box office", "tv series",
    ],
    "sports": [
        "sport", "sports", "football", "soccer", "basketball", "nba", "nfl", "cricket",
        "tennis", "olympics", "world cup", "formula 1", "f1", "premier league",
        "championship", "tournament",
    ],
}

#: Human-readable section headings for the briefing.
TOPIC_LABELS: dict[str, str] = {
    "ai": "AI & Machine Learning",
    "linux": "Linux & Open Source",
    "open-source": "Open Source",
    "gaming": "Gaming",
    "hardware": "Hardware & Chips",
    "nvidia": "NVIDIA",
    "technology": "Technology",
    "security": "Security & Privacy",
    "science": "Science",
    "space": "Space",
    "business": "Business",
    "markets": "Markets & Economy",
    "us-politics": "US Politics",
    "geopolitics": "Geopolitics",
    "world": "World",
    "india": "India",
    "europe": "Europe",
    "asia": "Asia",
    "climate": "Climate & Energy",
    "health": "Health",
    "culture": "Culture",
    "sports": "Sport",
    "general": "Also Happening",
}

#: Broader topics are demoted when a more specific one also matches, so that a
#: GPU story lands under "Hardware" rather than the catch-all "Technology".
_GENERIC = {"technology", "world", "business", "science"}

_MULTIWORD = {
    topic: [term for term in terms if " " in term or "." in term]
    for topic, terms in LEXICON.items()
}
_SINGLE = {
    topic: {term for term in terms if " " not in term and "." not in term}
    for topic, terms in LEXICON.items()
}


def score_topics(text: str, extra_terms: list[str] | None = None) -> dict[str, float]:
    """Score every canonical topic against a block of text."""
    folded = fold(text)
    tokens = set(tokenize(text, drop_stopwords=False))
    for term in extra_terms or []:
        tokens.update(tokenize(term, drop_stopwords=False))
        folded = f"{folded} {fold(term)}"

    scores: dict[str, float] = defaultdict(float)
    for topic, singles in _SINGLE.items():
        hits = len(tokens & singles)
        if hits:
            scores[topic] += hits
        for phrase in _MULTIWORD[topic]:
            if contains_term(folded, phrase):
                scores[topic] += 1.6
    return dict(scores)


#: Minimum score for a topic to be assigned at all.
_MIN_TOPIC_SCORE = 1.5


def classify(article: Article, max_topics: int = 3) -> list[str]:
    """Assign canonical topics to an article.

    The article's own text dominates. Categories declared on the feed *entry*
    are real per-article metadata and count for a moderate amount. Categories
    declared on the *source* describe the whole publication, so they are only a
    weak prior -- weighted below ``_MIN_TOPIC_SCORE`` so they can break a tie but
    can never assign a topic on their own. Without that separation an aggregator
    tagged ``[technology, linux]`` would file its political stories under Linux,
    and a newspaper's politics feed would file its sports reporting under
    politics.
    """
    text = f"{article.title}. {article.summary} {article.content[:2000]}"
    scores = score_topics(text)

    def _apply(hints: list[str], direct: float, lexical: float) -> None:
        for hint in hints:
            key = fold(hint).strip().replace(" ", "-")
            if key in LEXICON:
                scores[key] = scores.get(key, 0.0) + direct
            for topic, weight in score_topics(hint).items():
                scores[topic] = scores.get(topic, 0.0) + weight * lexical

    _apply(list(article.feed_categories), direct=1.4, lexical=0.35)
    if article.source is not None:
        _apply(list(article.source.categories), direct=0.55, lexical=0.12)

    if not scores:
        return ["general"]

    specific = {t for t, v in scores.items() if t not in _GENERIC and v >= 2.0}
    adjusted = {
        topic: (value * 0.45 if topic in _GENERIC and specific else value)
        for topic, value in scores.items()
    }

    ranked = sorted(adjusted.items(), key=lambda kv: (-kv[1], kv[0]))
    best = ranked[0][1]
    if best < _MIN_TOPIC_SCORE:
        return ["general"]
    # Keep topics within a band of the leader so an article can be multi-topic.
    return [
        topic
        for topic, value in ranked[:max_topics]
        if value >= max(_MIN_TOPIC_SCORE, best * 0.45)
    ]


def label_for(topic: str) -> str:
    return TOPIC_LABELS.get(topic, topic.replace("-", " ").title())
