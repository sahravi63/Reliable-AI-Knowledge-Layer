from __future__ import annotations

import json

from src.providers.base import CompletionResult, LLMProvider, ProviderError, _safe_parse_json


class OpenAIProvider(LLMProvider):
    """
    OpenAI backend. Defaults to gpt-5-nano (cheapest current structured-output
    model per the project's tech-stack research). gpt-4o-mini is legacy;
    avoid it unless explicitly requested.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-5-nano", **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ProviderError(
                    "openai package not installed. Run: pip install openai"
                ) from e
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ProviderError("OPENAI_API_KEY environment variable is not set.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def complete_text(self, system: str, user: str) -> CompletionResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return CompletionResult(text=text, raw=resp, model=self.model, provider=self.name)

    def complete_json(self, system: str, user: str, schema_hint: str = "") -> dict:
        strict_system = f"{system}\n\n{schema_hint}\nRespond with ONLY valid JSON."
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": strict_system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _safe_parse_json(text)
