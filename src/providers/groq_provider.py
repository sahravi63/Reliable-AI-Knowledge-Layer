from __future__ import annotations

import json

from src.providers.base import CompletionResult, LLMProvider, ProviderError, _safe_parse_json


class GroqProvider(LLMProvider):
    """
    Groq-hosted open models (Llama 4 Scout, GPT-OSS-120B, Kimi K2, DeepSeek R1
    distill, etc). Native JSON mode via response_format={"type": "json_object"}.
    Free tier, fast inference — used as the default for dev iteration and
    for all grading/routing/conflict-resolution nodes.
    """

    name = "groq"

    def __init__(self, model: str = "openai/gpt-oss-120b", **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import Groq
            except ImportError as e:
                raise ProviderError(
                    "groq package not installed. Run: pip install groq"
                ) from e
            import os
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ProviderError("GROQ_API_KEY environment variable is not set.")
            self._client = Groq(api_key=api_key)
        return self._client

    def complete_text(self, system: str, user: str) -> CompletionResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return CompletionResult(text=text, raw=resp, model=self.model, provider=self.name)

    def complete_json(self, system: str, user: str, schema_hint: str = "") -> dict:
        strict_system = f"{system}\n\n{schema_hint}\nRespond with ONLY valid JSON."
        messages = [
            {"role": "system", "content": strict_system},
            {"role": "user", "content": user},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max(self.max_tokens, 2048),
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as exc:
            if "json_validate_failed" in str(exc) or "Failed to validate JSON" in str(exc):
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max(self.max_tokens, 2048),
                    messages=messages,
                )
            else:
                raise
        text = resp.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _safe_parse_json(text)
