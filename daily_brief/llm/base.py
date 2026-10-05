"""Provider interface."""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass


class LLMUnavailable(Exception):
    """Raised when no model can serve a request.

    Callers must treat this as "fall back to the extractive engine", never as
    "answer from the model's own knowledge".
    """


@dataclass(slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


#: Reasoning models wrap their scratchpad in these; it is not part of the answer.
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK_RE = re.compile(r"^.*?</(?:think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)


def clean_completion(text: str) -> str:
    """Strip reasoning blocks and surrounding whitespace from a completion."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _ORPHAN_THINK_RE.sub("", cleaned)
    return cleaned.strip()


class LLMProvider(abc.ABC):
    """Minimal chat-completion interface."""

    name: str = "provider"
    model: str = ""

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return a completion, or raise :class:`LLMUnavailable`."""

    @abc.abstractmethod
    async def available(self) -> bool:
        """Whether the provider can currently serve a request."""

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None

    def describe(self) -> dict[str, str]:
        return {"provider": self.name, "model": self.model}
