from __future__ import annotations

import json

from src.providers.base import CompletionResult, LLMProvider, ProviderError, _safe_parse_json


class GeminiProvider(LLMProvider):
    """
    Google Gemini backend, via the `google-genai` SDK. Supports native
    JSON mode through response_mime_type="application/json".
    """

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ProviderError(
                    "google-genai package not installed. Run: pip install google-genai"
                ) from e
            import os
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def complete_text(self, system: str, user: str) -> CompletionResult:
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        text = resp.text or ""
        return CompletionResult(text=text, raw=resp, model=self.model, provider=self.name)

    def complete_json(self, system: str, user: str, schema_hint: str = "") -> dict:
        from google.genai import types
        strict_system = f"{system}\n\n{schema_hint}"
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=strict_system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        text = resp.text or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _safe_parse_json(text)
