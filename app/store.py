"""A small persistent vector store: a NumPy matrix for vectors + JSON for chunk metadata."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ingest import Chunk


class VectorStore:
    def __init__(self, index_dir: str, embedder_name: str):
        self.dir = Path(index_dir)
        self.embedder_name = embedder_name
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._load()

    # ---------- persistence ----------
    @property
    def _meta_path(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def _vec_path(self) -> Path:
        return self.dir / "vectors.npy"

    def _load(self) -> None:
        if not (self._meta_path.exists() and self._vec_path.exists()):
            return
        meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != self.embedder_name:
            # Vectors from a different model are not comparable; start fresh.
            return
        self.chunks = [Chunk(**c) for c in meta["chunks"]]
        self.vectors = np.load(self._vec_path)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = {"embedder": self.embedder_name, "chunks": [asdict(c) for c in self.chunks]}
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        np.save(self._vec_path, self.vectors if self.vectors is not None else np.zeros((0, 0), np.float32))

    # ---------- operations ----------
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None or self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        self.save()

    def remove_source(self, source: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else None
            self.save()
        return removed

    def clear(self) -> None:
        self.chunks, self.vectors = [], None
        self.save()

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ query_vec  # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
