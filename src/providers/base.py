"""
Pluggable LLM provider interface.

Every node in the pipeline (router, relevance grader, conflict resolver,
generator, hallucination grader) calls `LLMProvider.complete_json(...)` or
`LLMProvider.complete_text(...)`. Swapping providers never touches graph logic.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from typing import Any, Optional


class ProviderError(RuntimeError):
    """Raised when a provider call fails or returns unparseable output."""


@dataclass
class CompletionResult:
    text: str
    raw: Any = None
    model: str = ""
    provider: str = ""


class LLMProvider(abc.ABC):
    """Common interface every backend implements."""

    name: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abc.abstractmethod
    def complete_text(self, system: str, user: str) -> CompletionResult:
        """Plain text completion."""
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema_hint: str = "") -> dict:
        """
        Structured-output completion. Default implementation: instruct the
        model to return ONLY JSON, then parse defensively. Providers that
        support native JSON mode (OpenAI, Gemini) should override this.
        """
        strict_system = (
            f"{system}\n\n"
            "CRITICAL: Respond with ONLY a single valid JSON object. "
            "No markdown fences, no preamble, no explanation outside the JSON. "
            f"{schema_hint}"
        )
        result = self.complete_text(strict_system, user)
        return _safe_parse_json(result.text)


def _safe_parse_json(text: str) -> dict:
    """Strip code fences / stray text and parse JSON, raising ProviderError on failure."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

    # If the model wrapped the JSON in prose, grab the first {...} block.
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ProviderError(f"Failed to parse JSON from model output: {e}\nRaw: {text[:500]}")


def get_provider(name: Optional[str] = None, model: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory. `name` in {"groq", "openai", "anthropic", "gemini"}.
    Falls back to config.DEFAULT_PROVIDER (Groq) if not specified.
    """
    from src.config import settings

    provider_name = (name or settings.default_provider).lower()

    if provider_name == "groq":
        from src.providers.groq_provider import GroqProvider
        return GroqProvider(model=model or settings.groq_model, **kwargs)
    if provider_name == "openai":
        from src.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model or settings.openai_model, **kwargs)
    if provider_name == "anthropic":
        from src.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model or settings.anthropic_model, **kwargs)
    if provider_name == "gemini":
        from src.providers.gemini_provider import GeminiProvider
        return GeminiProvider(model=model or settings.gemini_model, **kwargs)

    raise ValueError(f"Unknown provider: {provider_name}")
