"""The no-model provider.

Selected by ``[llm] provider = "none"``. It always reports itself unavailable,
which routes every request to the deterministic extractive engine. That engine
only ever quotes sentences from retrieved articles, so a Daily-Brief install
with no inference stack at all is still useful and structurally incapable of
fabricating news.
"""

from __future__ import annotations

from .base import ChatMessage, LLMProvider, LLMUnavailable


class NullProvider(LLMProvider):
    name = "none"
    model = "extractive"

    async def available(self) -> bool:
        return False

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise LLMUnavailable("no LLM provider configured")
