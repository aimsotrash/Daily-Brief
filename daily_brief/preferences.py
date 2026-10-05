"""User personalization.

Kept separate from :mod:`daily_brief.config`: system configuration is an
operator concern living in TOML, personalization is a user concern living in the
database. The stored shape is an open JSON document, so future fields
(preferred/excluded sources, geographic focus, briefing length, bias display,
notifications) can be added without a schema migration.
"""

from __future__ import annotations

import logging

from .analysis.relevance import Interest, parse_interests
from .models import Preferences
from .repository import PreferencesRepository

log = logging.getLogger(__name__)

MAX_INTERESTS = 24


class PreferencesService:
    def __init__(self, repo: PreferencesRepository) -> None:
        self.repo = repo

    def load(self) -> Preferences:
        """Current preferences. A fresh install returns an un-onboarded default."""
        return self.repo.load() or Preferences()

    def is_first_run(self) -> bool:
        prefs = self.repo.load()
        return prefs is None or not prefs.onboarded

    def save_interests(self, raw: str | list[str], *, onboarded: bool = True) -> Preferences:
        """Interpret free-text interests and persist them.

        The raw text the user typed is kept alongside the parsed labels so it can
        be shown back for editing verbatim.
        """
        interests = parse_interests(raw)
        labels: list[str] = []
        for interest in interests:
            if interest.label not in labels:
                labels.append(interest.label)
        labels = labels[:MAX_INTERESTS]

        raw_text = raw if isinstance(raw, str) else ", ".join(str(x) for x in raw)

        prefs = self.load()
        prefs.interests = labels
        prefs.raw_interests_text = raw_text.strip()
        prefs.onboarded = onboarded or prefs.onboarded
        saved = self.repo.save(prefs)
        log.info("saved %d interest(s): %s", len(labels), ", ".join(labels))
        return saved

    def update(self, **fields) -> Preferences:
        """Update arbitrary preference fields; unknown keys go to ``extras``."""
        prefs = self.load()
        for key, value in fields.items():
            if key == "interests":
                return self.save_interests(value, onboarded=prefs.onboarded)
            if hasattr(prefs, key):
                setattr(prefs, key, value)
            else:
                prefs.extras[key] = value
        return self.repo.save(prefs)

    def reset(self) -> Preferences:
        """Clear personalization and return to the first-run state."""
        self.repo.reset()
        log.info("preferences reset")
        return Preferences()

    def interests(self) -> list[Interest]:
        """Parsed interests for the relevance scorer."""
        prefs = self.load()
        source = prefs.raw_interests_text or prefs.interests
        if not source:
            return []
        return parse_interests(source)
