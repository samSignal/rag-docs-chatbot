"""Application settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader so we don't need python-dotenv. Existing env vars win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # Which backend generates answers: "openai", "ollama" or "offline"
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "offline"))
    # Which backend creates embeddings: "openai", "ollama" or "offline"
    embed_provider: str = field(default_factory=lambda: _env("EMBED_PROVIDER", "offline"))

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_embed_model: str = field(default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2"))
    ollama_embed_model: str = field(default_factory=lambda: _env("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "180")))       # words
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "40")))  # words
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.05")))
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", "data/index"))


def get_settings() -> Settings:
    return Settings()
