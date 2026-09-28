"""The RAG pipeline: ingest -> chunk -> embed -> store, and retrieve -> prompt -> answer with citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .ingest import chunk_pages, read_document
from .llm import LLM, get_llm
from .store import VectorStore


@dataclass
class Source:
    ref: int
    source: str
    page: int
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    latency_ms: int = 0


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.llm = llm or get_llm(self.settings)
        self.store = VectorStore(self.settings.index_dir, self.embedder.name)

    def ingest(self, data: bytes, filename: str) -> int:
        """Index one document. Re-uploading a file replaces its old chunks. Returns chunks added."""
        pages = read_document(data, filename)
        chunks = chunk_pages(pages, filename, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(f"No extractable text found in '{filename}' (scanned PDFs need OCR first).")
        self.store.remove_source(filename)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if not self.store.chunks:
            return Answer(question, "No documents indexed yet. Upload a document first.")

        qvec = self.embedder.embed([question])[0]
        hits = [(c, s) for c, s in self.store.search(qvec, self.settings.top_k) if s >= self.settings.min_score]
        if not hits:
            return Answer(question, "I couldn't find that in the uploaded documents.",
                          latency_ms=int((time.perf_counter() - start) * 1000))

        text = self.llm.answer(question, [c.text for c, _ in hits])
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        if not cited and "couldn't find" in text.lower():
            return Answer(question, text, [], int((time.perf_counter() - start) * 1000))
        sources = [
            Source(ref=i, source=c.source, page=c.page, score=round(s, 3), text=c.text)
            for i, (c, s) in enumerate(hits, start=1)
            if not cited or i in cited  # show only passages the answer actually cites
        ]
        return Answer(question, text, sources, int((time.perf_counter() - start) * 1000))
