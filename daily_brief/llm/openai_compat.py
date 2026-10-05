"""OpenAI-compatible chat-completions client.

Works against any server implementing ``POST {base_url}/chat/completions``:
Ollama (``http://127.0.0.1:11434/v1``), llama.cpp's ``llama-server``, vLLM,
LM Studio, LocalAI, or a hosted endpoint. Choosing the model and the server is
the operator's decision, expressed entirely in configuration.
"""

from __future__ import annotations

import logging
import time

import httpx

from ..config import LLMConfig
from .base import ChatMessage, LLMProvider, LLMUnavailable, clean_completion

log = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model
        self.base_url = config.base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._health: tuple[float, bool] | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.config.timeout_seconds, connect=8.0),
                headers=headers,
            )
        return self._client

    async def available(self) -> bool:
        """Probe the endpoint, caching the answer briefly.

        Health is cached so a down model does not add a connection timeout to
        every single request.
        """
        now = time.monotonic()
        if self._health is not None:
            checked_at, healthy = self._health
            if now - checked_at < self.config.health_cache_seconds:
                return healthy

        healthy = False
        try:
            response = await self._get_client().get("/models", timeout=5.0)
            healthy = response.status_code < 500
        except httpx.HTTPError as exc:
            log.debug("LLM health probe failed for %s: %s", self.base_url, exc)
        self._health = (now, healthy)
        return healthy

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        try:
            response = await self._get_client().post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            self._health = (time.monotonic(), False)
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            body = response.text[:400]
            if response.status_code >= 500:
                self._health = (time.monotonic(), False)
            raise LLMUnavailable(f"HTTP {response.status_code}: {body}")

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"unexpected response shape: {exc}") from exc

        self._health = (time.monotonic(), True)
        cleaned = clean_completion(content or "")
        if not cleaned:
            raise LLMUnavailable("model returned an empty completion")
        return cleaned

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
