"""Configuration loading, layering and type coercion."""

from __future__ import annotations

import pytest

from daily_brief.config import Config, ConfigError, load_config


def write_config(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestDefaults:
    def test_defaults_load_without_a_file(self):
        config = Config()
        assert config.server.port == 8787
        assert config.llm.fallback_to_extractive is True

    def test_shipped_config_is_valid(self):
        """The config file in the repo must actually parse."""
        config = load_config("config/config.toml")
        assert config.server.port > 0
        assert config.sources_path().name == "sources.yaml"

    def test_data_dir_expands_home(self, tmp_path):
        config = Config()
        config.storage.data_dir = "~/somewhere"
        assert "~" not in str(config.storage.resolved_dir())

    def test_sources_path_resolves_relative_to_the_config_file(self, tmp_path):
        path = write_config(tmp_path, '[sources]\nregistry = "feeds.yaml"\n')
        config = load_config(path)
        assert config.sources_path() == tmp_path / "feeds.yaml"

    def test_absolute_registry_path_is_respected(self, tmp_path):
        path = write_config(tmp_path, '[sources]\nregistry = "/etc/feeds.yaml"\n')
        assert str(load_config(path).sources_path()) == "/etc/feeds.yaml"


class TestFileLoading:
    def test_values_override_defaults(self, tmp_path):
        path = write_config(tmp_path, "[server]\nport = 9999\nhost = \"0.0.0.0\"\n")
        config = load_config(path)
        assert config.server.port == 9999
        assert config.server.host == "0.0.0.0"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "nope.toml")

    def test_unknown_section_raises(self, tmp_path):
        path = write_config(tmp_path, "[nonsense]\nfoo = 1\n")
        with pytest.raises(ConfigError, match="unknown config section"):
            load_config(path)

    def test_unknown_option_raises(self, tmp_path):
        path = write_config(tmp_path, "[server]\nnot_an_option = 1\n")
        with pytest.raises(ConfigError, match="unknown option"):
            load_config(path)

    def test_partial_sections_keep_other_defaults(self, tmp_path):
        path = write_config(tmp_path, "[server]\nport = 9999\n")
        assert load_config(path).server.host == "127.0.0.1"


class TestEnvironmentOverrides:
    def test_string_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAILY_BRIEF_LLM_MODEL", "my-model")
        assert load_config(write_config(tmp_path, "")).llm.model == "my-model"

    def test_int_override_is_coerced(self, tmp_path, monkeypatch):
        """Regression: annotations are strings here, so coercion must resolve them."""
        monkeypatch.setenv("DAILY_BRIEF_SERVER_PORT", "9001")
        port = load_config(write_config(tmp_path, "")).server.port
        assert port == 9001 and isinstance(port, int)

    def test_float_override_is_coerced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAILY_BRIEF_LLM_TIMEOUT_SECONDS", "300")
        timeout = load_config(write_config(tmp_path, "")).llm.timeout_seconds
        assert timeout == 300.0 and isinstance(timeout, float)

    @pytest.mark.parametrize(
        "value,expected",
        [("true", True), ("1", True), ("yes", True), ("false", False), ("0", False)],
    )
    def test_bool_override_is_coerced(self, tmp_path, monkeypatch, value, expected):
        monkeypatch.setenv("DAILY_BRIEF_LLM_FALLBACK_TO_EXTRACTIVE", value)
        result = load_config(write_config(tmp_path, "")).llm.fallback_to_extractive
        assert result is expected

    def test_env_wins_over_file(self, tmp_path, monkeypatch):
        path = write_config(tmp_path, "[server]\nport = 9999\n")
        monkeypatch.setenv("DAILY_BRIEF_SERVER_PORT", "7777")
        assert load_config(path).server.port == 7777

    def test_invalid_numeric_override_is_reported_clearly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAILY_BRIEF_SERVER_PORT", "not-a-number")
        with pytest.raises(ConfigError, match="DAILY_BRIEF_SERVER_PORT"):
            load_config(write_config(tmp_path, ""))

    def test_api_key_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DAILY_BRIEF_LLM_API_KEY", "secret-token")
        assert load_config(write_config(tmp_path, "")).llm.api_key == "secret-token"


class TestProviderSelection:
    """The application must never be pinned to one inference stack."""

    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("openai_compat", "openai_compat"),
            ("ollama", "openai_compat"),
            ("llamacpp", "openai_compat"),
            ("vllm", "openai_compat"),
            ("none", "none"),
            ("", "none"),
            ("something-unknown", "none"),
        ],
    )
    def test_provider_registry(self, provider, expected):
        from daily_brief.config import LLMConfig
        from daily_brief.llm.factory import build_provider

        assert build_provider(LLMConfig(provider=provider)).name == expected

    async def test_null_provider_is_never_available(self):
        from daily_brief.llm.null import NullProvider

        provider = NullProvider()
        assert await provider.available() is False
