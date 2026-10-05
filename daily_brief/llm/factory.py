"""Provider construction from configuration."""

from __future__ import annotations

import logging

from ..config import LLMConfig
from .base import LLMProvider
from .null import NullProvider
from .openai_compat import OpenAICompatProvider

log = logging.getLogger(__name__)

_REGISTRY = {
    "openai_compat": OpenAICompatProvider,
    "openai": OpenAICompatProvider,
    "ollama": OpenAICompatProvider,
    "llamacpp": OpenAICompatProvider,
    "vllm": OpenAICompatProvider,
    "none": NullProvider,
    "": NullProvider,
}


def build_provider(config: LLMConfig) -> LLMProvider:
    key = (config.provider or "").strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        log.warning(
            "unknown llm provider %r; falling back to the extractive engine", config.provider
        )
        return NullProvider()
    if factory is NullProvider:
        return NullProvider()
    return factory(config)
