"""Inference layer.

Deliberately thin and provider-agnostic. Daily-Brief never depends on a specific
model or vendor: anything exposing an OpenAI-compatible ``/chat/completions``
endpoint works (Ollama, llama.cpp's server, vLLM, LM Studio, LocalAI), and the
application remains fully functional with no model at all.
"""

from .base import ChatMessage, LLMProvider, LLMUnavailable
from .factory import build_provider

__all__ = ["ChatMessage", "LLMProvider", "LLMUnavailable", "build_provider"]
