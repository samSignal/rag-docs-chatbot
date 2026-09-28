"""Embedding backends. All return L2-normalised float32 vectors, so dot product == cosine similarity."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

import numpy as np
import requests

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was were what when "
    "where which who why will with do does can you your we our they their".split()
)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Offline embedder: hashed unigrams + bigrams with sublinear TF.

    No model download and no API key, so the project runs (and CI tests pass) anywhere.
    It is lexical rather than semantic; switch EMBED_PROVIDER to openai/ollama for real use.
    """

    name = "offline-hashing"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _features(self, text: str) -> Counter:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        toks = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]  # crude plural folding
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in self._features(text).items():
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                sign = 1.0 if (h >> 1) & 1 else -1.0
                m[row, h % self.dim] += sign * (1.0 + math.log(count))
        return _normalise(m)


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embed_model
        self.name = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 100):  # batch to stay under request limits
            resp = self.client.embeddings.create(model=self.model, input=texts[i : i + 100])
            vectors.extend(d.embedding for d in resp.data)
        return _normalise(np.array(vectors, dtype=np.float32))


class OllamaEmbedder:
    def __init__(self, settings: Settings):
        self.url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_embed_model
        self.name = f"ollama:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = requests.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=120)
        resp.raise_for_status()
        return _normalise(np.array(resp.json()["embeddings"], dtype=np.float32))


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embed_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(settings)
    if provider == "ollama":
        return OllamaEmbedder(settings)
    return HashingEmbedder()
