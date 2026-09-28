"""Command-line interface.

    python cli.py ingest sample_docs/          # index a folder or single file
    python cli.py ask "How many days of annual leave do I get?"
    python cli.py chat                        # interactive loop
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.ingest import SUPPORTED
from app.rag import RAGPipeline


def cmd_ingest(rag: RAGPipeline, target: str) -> None:
    path = Path(target)
    files = [p for p in (path.rglob("*") if path.is_dir() else [path]) if p.suffix.lower() in SUPPORTED]
    for f in files:
        n = rag.ingest(f.read_bytes(), f.name)
        print(f"indexed {f.name}: {n} chunks")
    print(f"done - {len(rag.store.chunks)} chunks in index")


def show(rag: RAGPipeline, q: str) -> None:
    a = rag.ask(q)
    print(f"\n{a.answer}\n")
    for s in a.sources:
        print(f"  [{s.ref}] {s.source} p.{s.page} (score {s.score})")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="RAG Docs Chatbot CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("path")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("chat")
    args = p.parse_args()

    rag = RAGPipeline()
    if args.cmd == "ingest":
        cmd_ingest(rag, args.path)
    elif args.cmd == "ask":
        show(rag, args.question)
    else:
        print(f"LLM: {rag.llm.name} | embeddings: {rag.embedder.name}. Type 'exit' to quit.")
        while (q := input("> ").strip()).lower() not in {"exit", "quit"}:
            if q:
                show(rag, q)


if __name__ == "__main__":
    main()
