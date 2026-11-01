from __future__ import annotations

from typing import Sequence

from app.config import Settings
from app.core.types import RetrievedChunk
from app.llm.service import LLMService


class Reranker:
    """LLM cross-encoder style rerank with a deterministic fallback."""

    def __init__(self, llm: LLMService, settings: Settings):
        self.llm = llm
        self.settings = settings

    def rerank(
        self,
        question: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int | None = None,
        *,
        expansion: str = "",
    ) -> list[RetrievedChunk]:
        top_n = top_n or self.settings.rerank_top_n
        return self.llm.rerank(question, candidates, top_n=top_n, expansion=expansion)
