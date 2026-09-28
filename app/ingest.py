"""Load documents (PDF, TXT, MD) and split them into overlapping chunks."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Chunk:
    text: str
    source: str   # file name
    page: int     # 1-based page number (1 for plain-text files)


def read_document(data: bytes, filename: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) pairs for a document."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(sorted(SUPPORTED))}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]

    return [(1, data.decode("utf-8", errors="ignore"))]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_pages(pages: list[tuple[int, str]], source: str, size: int = 180, overlap: int = 40) -> list[Chunk]:
    """Split each page into word windows of `size` words, overlapping by `overlap` words.

    Overlap keeps sentences that straddle a boundary retrievable from either chunk.
    Chunks never span pages, so every chunk can be cited with an exact page number.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

    chunks: list[Chunk] = []
    step = size - overlap
    for page_no, raw in pages:
        words = _clean(raw).split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + size]
            chunks.append(Chunk(text=" ".join(window), source=source, page=page_no))
            if start + size >= len(words):
                break
    return chunks
