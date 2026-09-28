from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.ingest import chunk_pages, read_document
from app.rag import RAGPipeline
from app.store import VectorStore

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture
def rag(tmp_path):
    rag = RAGPipeline(Settings(llm_provider="offline", embed_provider="offline", index_dir=str(tmp_path)))
    for f in DOCS.iterdir():
        rag.ingest(f.read_bytes(), f.name)
    return rag


def test_chunking_overlaps_and_keeps_pages():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_pages([(1, words), (2, "second page")], "doc.txt", size=40, overlap=10)
    assert [c.page for c in chunks] == [1, 1, 1, 2]
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]  # overlap preserved
    assert chunks[2].text.split()[-1] == "w99"                             # nothing dropped


def test_chunking_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_pages([(1, "x")], "d", size=10, overlap=10)


def test_unsupported_file_type():
    with pytest.raises(ValueError):
        read_document(b"x", "image.png")


def test_embeddings_are_normalised_and_similar_texts_score_higher():
    e = HashingEmbedder()
    v = e.embed(["annual leave days", "how many annual leave days", "vpn client install"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_store_persists_and_replaces_source(tmp_path):
    e = HashingEmbedder()
    s = VectorStore(str(tmp_path), e.name)
    chunks = chunk_pages([(1, "hello world")], "a.txt")
    s.add(chunks, e.embed([c.text for c in chunks]))
    assert VectorStore(str(tmp_path), e.name).sources() == {"a.txt": 1}      # reloaded from disk
    assert VectorStore(str(tmp_path), "other-model").chunks == []            # incompatible index ignored
    assert s.remove_source("a.txt") == 1 and s.sources() == {}


@pytest.mark.parametrize(
    "question, expected_source, expected_text",
    [
        ("How many days of annual leave do I get?", "employee_handbook.md", "22 days"),
        ("How long must a password be?", "it_support_faq.txt", "14 characters"),
        ("How much can remote workers claim for home-office furniture?", "employee_handbook.md", "USD 300"),
        ("How do I report a phishing email?", "it_support_faq.txt", "Report phishing"),
    ],
)
def test_answers_are_grounded_and_cited(rag, question, expected_source, expected_text):
    a = rag.ask(question)
    assert expected_text in a.answer
    assert "[1]" in a.answer or "[2]" in a.answer
    assert a.sources and a.sources[0].source == expected_source


def test_reingest_does_not_duplicate(rag):
    before = len(rag.store.chunks)
    f = DOCS / "employee_handbook.md"
    rag.ingest(f.read_bytes(), f.name)
    assert len(rag.store.chunks) == before


def test_unanswerable_question(rag):
    a = rag.ask("What is the capital of Mongolia?")
    assert "couldn't find" in a.answer.lower()
    assert a.sources == []


def test_empty_index(tmp_path):
    rag = RAGPipeline(Settings(index_dir=str(tmp_path)))
    assert "No documents" in rag.ask("anything").answer


class FakeLLM:
    """Stands in for OpenAI/Ollama so the prompt->citation plumbing is tested without network."""

    name = "fake"

    def __init__(self):
        self.seen = None

    def answer(self, question, passages):
        self.seen = passages
        return "According to the handbook you get 22 days [2]."


def test_only_cited_passages_are_returned(tmp_path):
    llm = FakeLLM()
    rag = RAGPipeline(Settings(index_dir=str(tmp_path), top_k=3, min_score=0.0), llm=llm)
    for f in DOCS.iterdir():
        rag.ingest(f.read_bytes(), f.name)
    a = rag.ask("annual leave days")
    assert len(llm.seen) == 3
    assert [s.ref for s in a.sources] == [2]
