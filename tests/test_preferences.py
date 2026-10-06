"""Preferences: first-run detection, saving, loading, modifying, resetting."""

from __future__ import annotations

from daily_brief.analysis.relevance import parse_interests
from daily_brief.preferences import PreferencesService
from daily_brief.repository import PreferencesRepository


def service(db) -> PreferencesService:
    return PreferencesService(PreferencesRepository(db))


class TestFirstRun:
    def test_fresh_install_is_first_run(self, db):
        assert service(db).is_first_run() is True

    def test_fresh_install_has_empty_defaults(self, db):
        prefs = service(db).load()
        assert prefs.onboarded is False
        assert prefs.interests == []
        assert prefs.raw_interests_text == ""

    def test_first_run_false_after_onboarding(self, db):
        svc = service(db)
        svc.save_interests("AI, Linux")
        assert svc.is_first_run() is False

    def test_saving_without_onboarding_keeps_first_run(self, db):
        svc = service(db)
        svc.save_interests("AI", onboarded=False)
        assert svc.is_first_run() is True


class TestSaveAndLoad:
    def test_saves_and_reloads_interests(self, db):
        svc = service(db)
        svc.save_interests("AI, Linux, gaming")
        # A separate service instance proves it round-trips through storage.
        reloaded = service(db).load()
        assert reloaded.onboarded is True
        assert "AI & Machine Learning" in reloaded.interests
        assert "Linux" in reloaded.interests
        assert "Gaming" in reloaded.interests

    def test_keeps_raw_text_for_editing(self, db):
        raw = "AI, Linux, gaming and things happening in India"
        svc = service(db)
        svc.save_interests(raw)
        assert svc.load().raw_interests_text == raw

    def test_accepts_a_list(self, db):
        svc = service(db)
        prefs = svc.save_interests(["AI", "Linux"])
        assert len(prefs.interests) == 2

    def test_caps_interest_count(self, db):
        svc = service(db)
        prefs = svc.save_interests(", ".join(f"topic{i}" for i in range(60)))
        assert len(prefs.interests) <= 24

    def test_unparseable_input_yields_no_interests(self, db):
        prefs = service(db).save_interests("!!! ??? ...")
        assert prefs.interests == []


class TestModify:
    def test_replaces_rather_than_appends(self, db):
        svc = service(db)
        svc.save_interests("AI, Linux")
        prefs = svc.save_interests("gaming")
        assert prefs.interests == ["Gaming"]

    def test_update_writes_extras_for_unknown_fields(self, db):
        svc = service(db)
        svc.save_interests("AI")
        # Forward compatibility: unknown keys must persist without a migration.
        prefs = svc.update(briefing_length="short", excluded_sources=["tech-daily"])
        assert prefs.extras["briefing_length"] == "short"
        assert service(db).load().extras["excluded_sources"] == ["tech-daily"]

    def test_update_interests_reparses(self, db):
        svc = service(db)
        svc.save_interests("AI")
        prefs = svc.update(interests="Linux, gaming")
        assert "Linux" in prefs.interests
        assert "AI & Machine Learning" not in prefs.interests


class TestReset:
    def test_reset_returns_to_first_run(self, db):
        svc = service(db)
        svc.save_interests("AI, Linux")
        svc.reset()
        assert svc.is_first_run() is True
        assert svc.load().interests == []

    def test_reset_is_idempotent(self, db):
        svc = service(db)
        svc.reset()
        svc.reset()
        assert svc.is_first_run() is True


class TestInterestParsing:
    """Interests are free text, so parsing must be forgiving."""

    def test_comma_separated(self):
        labels = [i.label for i in parse_interests("AI, Linux, gaming")]
        assert labels == ["AI & Machine Learning", "Linux", "Gaming"]

    def test_natural_language_sentence(self):
        interests = parse_interests(
            "AI, Linux, gaming, technology, geopolitics, "
            "US politics and interesting things happening in India"
        )
        labels = [i.label for i in interests]
        assert "India" in labels
        assert "US Politics" in labels
        assert "Geopolitics" in labels
        # "interesting things happening in" must be stripped, not kept as a topic.
        assert not any("interesting" in label.lower() for label in labels)

    def test_newline_and_bullet_separated(self):
        labels = [i.label for i in parse_interests("• AI\n• Linux\n• Gaming")]
        assert labels == ["AI & Machine Learning", "Linux", "Gaming"]

    def test_deduplicates_case_insensitively(self):
        assert len(parse_interests("AI, ai, A.I., artificial intelligence")) <= 2

    def test_unmapped_interest_kept_as_free_text(self):
        interests = parse_interests("competitive bonsai cultivation")
        assert len(interests) == 1
        assert interests[0].topic == ""
        assert interests[0].label == "Competitive Bonsai Cultivation"

    def test_mapped_interest_gets_expansions(self):
        linux = parse_interests("Linux")[0]
        assert linux.topic == "linux"
        assert "kernel" in linux.terms
        assert "ubuntu" in linux.terms

    def test_empty_input_yields_nothing(self):
        assert parse_interests("") == []
        assert parse_interests("   ") == []

    def test_no_hardcoded_user_interests(self):
        """Parsing must be driven purely by input, never by a baked-in profile."""
        assert parse_interests("knitting") != parse_interests("AI")
        assert [i.label for i in parse_interests("knitting")] == ["Knitting"]
