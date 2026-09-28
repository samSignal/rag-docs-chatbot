"""FastAPI app: REST API + a minimal chat UI at /."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .rag import RAGPipeline

MAX_UPLOAD_MB = 20

app = FastAPI(title="RAG Docs Chatbot", version="1.0.0",
              description="Upload documents and ask questions; answers cite their sources.")
pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["How many days of annual leave do I get?"])


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm": pipeline.llm.name,
        "embedder": pipeline.embedder.name,
        "chunks": len(pipeline.store.chunks),
    }


@app.get("/documents")
def documents() -> dict:
    return {"documents": [{"name": n, "chunks": c} for n, c in pipeline.store.sources().items()]}


@app.post("/documents")
async def upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        added = pipeline.ingest(data, file.filename or "upload.txt")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"document": file.filename, "chunks": added}


@app.delete("/documents/{name}")
def delete_document(name: str) -> dict:
    removed = pipeline.store.remove_source(name)
    if not removed:
        raise HTTPException(404, f"No document named '{name}'")
    return {"document": name, "chunks_removed": removed}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return asdict(pipeline.ask(req.question))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
