"""
Central configuration. Reads from environment (.env via python-dotenv).
Groq is the default provider everywhere — override per-call or via env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"


@dataclass
class Settings:
    # --- provider selection ---
    default_provider: str = field(default_factory=lambda: os.environ.get("DEFAULT_PROVIDER", "groq"))

    # --- per-provider default models (overridable via env) ---
    groq_model: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-5-nano"))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    gemini_model: str = field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))

    # --- role -> provider overrides (optional; lets you mix providers per node) ---
    router_provider: str = field(default_factory=lambda: os.environ.get("ROUTER_PROVIDER", ""))
    grader_provider: str = field(default_factory=lambda: os.environ.get("GRADER_PROVIDER", ""))
    conflict_provider: str = field(default_factory=lambda: os.environ.get("CONFLICT_PROVIDER", ""))
    generator_provider: str = field(default_factory=lambda: os.environ.get("GENERATOR_PROVIDER", ""))
    hallucination_provider: str = field(default_factory=lambda: os.environ.get("HALLUCINATION_PROVIDER", ""))

    # --- retrieval / pipeline knobs ---
    embedding_model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    top_k: int = field(default_factory=lambda: int(os.environ.get("TOP_K", "4")))
    relevance_sufficiency_threshold: int = field(
        default_factory=lambda: int(os.environ.get("SUFFICIENCY_THRESHOLD", "2"))
    )  # "generate only if >= N of top_k chunks are relevant"
    max_rewrite_attempts: int = field(default_factory=lambda: int(os.environ.get("MAX_REWRITE_ATTEMPTS", "2")))
    max_regenerations: int = field(default_factory=lambda: int(os.environ.get("MAX_REGENERATIONS", "2")))
    faithfulness_threshold: float = field(
        default_factory=lambda: float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.8"))
    )

    # --- web search fallback ---
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))

    def provider_for(self, role: str) -> str:
        """Resolve which provider a given pipeline role should use (falls back to default)."""
        override = getattr(self, f"{role}_provider", "") or ""
        return override or self.default_provider


settings = Settings()
