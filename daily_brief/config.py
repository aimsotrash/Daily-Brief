"""Application configuration.

Configuration is layered, lowest priority first:

1. the defaults in this module,
2. ``config/config.toml`` (or ``$DAILY_BRIEF_CONFIG``),
3. environment variables of the form ``DAILY_BRIEF_<SECTION>_<KEY>``.

Nothing else in the codebase reads files or environment variables for tunables --
everything goes through :func:`load_config`, so there are no constants scattered
across modules.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, get_type_hints

ENV_PREFIX = "DAILY_BRIEF"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    open_browser: bool = False


@dataclass
class StorageConfig:
    data_dir: str = ""
    db_filename: str = "daily_brief.db"

    def resolved_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir).expanduser()
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        return base / "daily-brief"

    def db_path(self) -> Path:
        return self.resolved_dir() / self.db_filename


@dataclass
class SourcesConfig:
    registry: str = "sources.yaml"


@dataclass
class IngestConfig:
    refresh_interval_minutes: int = 30
    max_articles_per_feed: int = 40
    retention_days: int = 21
    fetch_timeout_seconds: float = 20.0
    concurrency: int = 8
    user_agent: str = "daily-brief/0.1 (self-hosted personal news reader)"
    max_consecutive_failures: int = 10


@dataclass
class BriefingConfig:
    schedule_hour: int = 6
    schedule_minute: int = 30
    max_age_minutes: int = 240
    max_sections: int = 8
    max_stories_per_section: int = 5
    max_total_stories: int = 30
    lookback_hours: int = 36


@dataclass
class SearchConfig:
    candidate_limit: int = 120
    context_articles: int = 8
    min_score: float = 0.06
    history_turns: int = 6
    recent_hours: int = 24


@dataclass
class LLMConfig:
    provider: str = "openai_compat"
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:8b"
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 900
    timeout_seconds: float = 120.0
    fallback_to_extractive: bool = True
    health_cache_seconds: float = 60.0


@dataclass
class AnalysisConfig:
    llm_bias_analysis: bool = False
    llm_ingest_summaries: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "daily-brief.log"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    #: Directory the config file was loaded from; relative paths resolve against it.
    config_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "config")
    config_path: Path | None = None

    def sources_path(self) -> Path:
        p = Path(self.sources.registry).expanduser()
        return p if p.is_absolute() else (self.config_dir / p)

    def log_path(self) -> Path | None:
        if not self.logging.file:
            return None
        p = Path(self.logging.file).expanduser()
        return p if p.is_absolute() else (self.storage.resolved_dir() / p)


@lru_cache(maxsize=None)
def _field_types(section_type: type) -> dict[str, Any]:
    """Resolve a dataclass's field types.

    ``dataclasses.fields(...).type`` yields *strings* in this module because of
    ``from __future__ import annotations``, so it cannot be compared against
    ``int``/``float``/``bool`` directly. ``get_type_hints`` resolves them to real
    types; without this, ``DAILY_BRIEF_LLM_TIMEOUT_SECONDS=300`` would be stored
    as the string ``"300"`` and blow up much later inside the HTTP client.
    """
    hints = get_type_hints(section_type)
    return {f.name: hints.get(f.name, str) for f in fields(section_type)}


def _coerce(value: Any, target_type: Any) -> Any:
    if target_type is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        raise ConfigError(f"expected a boolean, got {value!r}")
    try:
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"expected a number, got {value!r}") from exc
    if target_type is str:
        return str(value)
    return value


def _apply_mapping(section: Any, values: dict[str, Any], where: str) -> None:
    known = _field_types(type(section))
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"unknown option '{key}' in [{where}]")
        setattr(section, key, _coerce(value, known[key]))


class ConfigError(Exception):
    """Raised when configuration is malformed."""


def _default_config_path() -> Path | None:
    env = os.environ.get(f"{ENV_PREFIX}_CONFIG")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.cwd() / "config" / "config.toml",
        PROJECT_ROOT / "config" / "config.toml",
        Path(
            os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
        ).expanduser()
        / "daily-brief"
        / "config.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _apply_env(config: Config) -> None:
    """Overlay ``DAILY_BRIEF_<SECTION>_<KEY>`` environment variables."""
    sections = {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if is_dataclass(getattr(config, f.name))
    }
    for section_name, section in sections.items():
        types = _field_types(type(section))
        for name, field_type in types.items():
            env_key = f"{ENV_PREFIX}_{section_name.upper()}_{name.upper()}"
            if env_key in os.environ:
                try:
                    setattr(section, name, _coerce(os.environ[env_key], field_type))
                except ConfigError as exc:
                    raise ConfigError(f"{env_key}: {exc}") from exc


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from ``path`` (or the first default location found)."""
    config = Config()
    resolved = Path(path).expanduser() if path else _default_config_path()

    if resolved is not None:
        if not resolved.is_file():
            raise ConfigError(f"config file not found: {resolved}")
        try:
            with resolved.open("rb") as handle:
                data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:  # pragma: no cover - malformed file
            raise ConfigError(f"could not parse {resolved}: {exc}") from exc
        known_sections = {
            f.name for f in fields(config) if is_dataclass(getattr(config, f.name))
        }
        for section_name, values in data.items():
            if section_name not in known_sections:
                raise ConfigError(f"unknown config section '[{section_name}]'")
            if not isinstance(values, dict):
                raise ConfigError(f"section '[{section_name}]' must be a table")
            _apply_mapping(getattr(config, section_name), values, section_name)
        config.config_path = resolved
        config.config_dir = resolved.parent

    _apply_env(config)

    # Secrets are conventionally supplied out-of-band.
    if not config.llm.api_key:
        config.llm.api_key = os.environ.get(f"{ENV_PREFIX}_LLM_API_KEY", "")

    return config
