"""Context assembly: neighbour expansion, dedupe, budget and citation numbering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.config import Settings
from app.core.types import RetrievedChunk
from app.core.utils import estimate_tokens, normalize_text, sha256_text, truncate
from app.storage.db import Database


@dataclass
class ContextBundle:
    text: str
    sources: list[RetrievedChunk]
    tokens: int = 0
    chars: int = 0
    truncated: bool = False
    expanded: list[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def _expand_neighbors(self, candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        radius = max(0, self.settings.neighbor_expansion)
        if not radius or not candidates:
            return list(candidates)
        selected = list(candidates)
        seen = {c.chunk.id for c in selected}
        top = candidates[:2]
        neighbors = self.db.get_neighbor_chunks([c.chunk.id for c in top], radius=radius)
        for parent in top:
            for neighbor in neighbors.get(parent.chunk.id, []):
                if neighbor.id in seen:
                    continue
                seen.add(neighbor.id)
                selected.append(
                    RetrievedChunk(
                        chunk=neighbor,
                        score=parent.score * 0.72,
                        dense_score=parent.dense_score * 0.8,
                        bm25_score=parent.bm25_score * 0.8,
                        rerank_score=parent.rerank_score * 0.72,
                        sources=[*parent.sources, "neighbor"],
                        rerank_reason="相邻片段补全",
                        is_neighbor=True,
                    )
                )
        selected.sort(key=lambda c: (c.is_neighbor, -c.score))
        return selected

    def build(self, candidates: Sequence[RetrievedChunk]) -> ContextBundle:
        expanded = self._expand_neighbors(candidates)
        budget = max(400, self.settings.context_token_budget)
        max_chunks = max(1, self.settings.max_context_chunks)
        chosen: list[RetrievedChunk] = []
        blocks: list[str] = []
        total_tokens = 0
        total_chars = 0
        truncated = False
        seen_signatures: set[str] = set()
        seen_texts: list[str] = []
        per_doc: dict[str, int] = {}
        per_heading: dict[tuple[str, str], int] = {}

        for candidate in expanded:
            if len(chosen) >= max_chunks:
                truncated = True
                break
            text = truncate(candidate.chunk.text.strip(), 1400)
            if not text:
                continue
            normalized = re.sub(r"\s+", "", normalize_text(text))
            signature = sha256_text(normalized[:240])
            if signature in seen_signatures:
                continue
            if normalized and any(
                len(prev) >= 60 and (normalized in prev or prev in normalized)
                for prev in seen_texts
            ):
                continue
            seen_texts.append(normalized)
            if per_doc.get(candidate.chunk.doc_id, 0) >= 4:
                truncated = True
                continue
            heading_key = (candidate.chunk.doc_id, candidate.chunk.heading)
            if per_heading.get(heading_key, 0) >= 3:
                truncated = True
                continue
            seen_signatures.add(signature)
            per_doc[candidate.chunk.doc_id] = per_doc.get(candidate.chunk.doc_id, 0) + 1
            per_heading[heading_key] = per_heading.get(heading_key, 0) + 1
            page = ""
            if candidate.chunk.page_start:
                if candidate.chunk.page_end and candidate.chunk.page_end != candidate.chunk.page_start:
                    page = f" · 第 {candidate.chunk.page_start}-{candidate.chunk.page_end} 页"
                else:
                    page = f" · 第 {candidate.chunk.page_start} 页"
            heading = candidate.chunk.heading or "正文"
            index = len(chosen) + 1
            block = (
                f"[{index}] 《{candidate.chunk.doc_name}》 › {heading}{page}\n"
                f"{text}"
            )
            tokens = estimate_tokens(block)
            if chosen and total_tokens + tokens > budget:
                truncated = True
                break
            chosen.append(candidate)
            blocks.append(block)
            total_tokens += tokens
            total_chars += len(block)

        return ContextBundle(
            text="\n\n".join(blocks),
            sources=chosen,
            tokens=total_tokens,
            chars=total_chars,
            truncated=truncated,
            expanded=[c.chunk.id for c in expanded if c.is_neighbor],
        )
