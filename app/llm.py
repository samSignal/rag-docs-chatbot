"""Answer-generation backends."""
from __future__ import annotations

import re
from typing import Protocol

import requests

from .config import Settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions ONLY from the numbered context passages provided. "
    "Cite the passages you use with their numbers in square brackets, e.g. [1] or [2][3]. "
    "If the answer is not in the context, say: \"I couldn't find that in the uploaded documents.\" "
    "Do not invent facts. Keep answers concise."
)


def build_user_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer (with citations):"


class LLM(Protocol):
    name: str

    def answer(self, question: str, passages: list[str]) -> str: ...


class OpenAIChat:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.name = f"openai:{self.model}"

    def answer(self, question: str, passages: list[str]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, passages)},
            ],
        )
        return resp.choices[0].message.content.strip()


class OllamaChat:
    def __init__(self, settings: Settings):
        self.url = settings.ollama_url.rstrip("/")
        self.model = settings.ollama_model
        self.name = f"ollama:{self.model}"

    def answer(self, question: str, passages: list[str]) -> str:
        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, passages)},
                ],
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was were what when "
    "where which who why will with do does can you your we our they their many much".split()
)


def _terms(text: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _WORD.findall(text.lower()) if w not in _STOP}


class ExtractiveAnswerer:
    """Offline fallback: picks the context sentences that best match the question and cites them.

    It cannot paraphrase like an LLM, but it lets the whole pipeline run with no API key.
    """

    name = "offline-extractive"

    def answer(self, question: str, passages: list[str]) -> str:
        q = _terms(question)
        scored: list[tuple[float, int, str]] = []
        seen: set[str] = set()  # overlapping chunks repeat sentences; keep the first occurrence
        for idx, passage in enumerate(passages, start=1):
            for sent in re.split(r"(?<=[.!?])\s+", passage):
                sent = re.sub(r"#+\s*[^.?!]*?\s(?=[A-Z])", "", sent, count=1).strip()  # drop markdown headings
                overlap = len(q & _terms(sent))
                if overlap and sent not in seen:
                    seen.add(sent)
                    scored.append((overlap / (len(q) or 1), idx, sent))
        if not scored:
            return "I couldn't find that in the uploaded documents."
        scored.sort(key=lambda s: -s[0])
        best = scored[0][0]
        picked = [s for s in scored if s[0] >= best * 0.75][:2]
        return " ".join(f"{sent} [{idx}]" for _, idx, sent in picked)


def get_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAIChat(settings)
    if provider == "ollama":
        return OllamaChat(settings)
    return ExtractiveAnswerer()
