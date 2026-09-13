from __future__ import annotations

from src.providers.base import CompletionResult, LLMProvider, ProviderError


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude backend. No native JSON-mode flag, so we rely on the
    base class's prompt-and-parse strategy (strong instruction + defensive
    parsing), which Claude models follow reliably.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(model=model, **kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ProviderError(
                    "anthropic package not installed. Run: pip install anthropic"
                ) from e
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ProviderError("ANTHROPIC_API_KEY environment variable is not set.")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def complete_text(self, system: str, user: str) -> CompletionResult:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return CompletionResult(text=text, raw=resp, model=self.model, provider=self.name)
