"""Domain dataclasses shared by ingestion, retrieval, agent and API layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.utils import json_dumps, truncate


class Intent(str, Enum):
    KB_QUESTION = "kb_question"
    CHITCHAT = "chitchat"
    META = "meta"
    SUMMARIZE = "summarize"
    COMPARE = "compare"


@dataclass
class Block:
    """A structural unit produced by a loader (paragraph / heading / image)."""

    text: str
    page: int | None = None
    heading_path: list[str] = field(default_factory=list)
    content_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    title: str
    source_type: str
    blocks: list[Block]
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    ordinal: int
    text: str
    page_start: int | None = None
    page_end: int | None = None
    heading_path: list[str] = field(default_factory=list)
    content_type: str = "text"
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        return " › ".join(self.heading_path)

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "ordinal": self.ordinal,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": self.heading_path,
            "heading": self.heading,
            "content_type": self.content_type,
            "token_estimate": self.token_estimate,
        }
        if include_text:
            data["text"] = self.text
        return data


@dataclass
class QueryPlan:
    intent: Intent
    rewritten: str
    subqueries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    needs_retrieval: bool = True
    doc_ids: list[str] = field(default_factory=list)
    reason: str = ""
    source: str = "heuristic"

    def all_queries(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for q in [self.rewritten, *self.subqueries]:
            q = (q or "").strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "rewritten": self.rewritten,
            "subqueries": self.subqueries,
            "keywords": self.keywords,
            "needs_retrieval": self.needs_retrieval,
            "doc_ids": self.doc_ids,
            "reason": self.reason,
            "source": self.source,
        }


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    rrf_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    rerank_reason: str = ""
    is_neighbor: bool = False

    def to_source(self, index: int) -> "Source":
        return Source(
            index=index,
            chunk_id=self.chunk.id,
            doc_id=self.chunk.doc_id,
            doc_name=self.chunk.doc_name,
            page_start=self.chunk.page_start,
            page_end=self.chunk.page_end,
            heading=self.chunk.heading,
            content_type=self.chunk.content_type,
            text=self.chunk.text,
            snippet=truncate(self.chunk.text.replace("\n", " "), 240),
            score=round(self.score, 4),
            dense_score=round(self.dense_score, 4),
            bm25_score=round(self.bm25_score, 4),
            rerank_score=round(self.rerank_score, 4),
            sources=self.sources,
            is_neighbor=self.is_neighbor,
        )


@dataclass
class Source:
    index: int
    chunk_id: str
    doc_id: str
    doc_name: str
    page_start: int | None
    page_end: int | None
    heading: str
    content_type: str
    text: str
    snippet: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    sources: list[str] = field(default_factory=list)
    is_neighbor: bool = False

    def to_dict(self, *, include_text: bool = True, text_limit: int = 900) -> dict[str, Any]:
        data = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading": self.heading,
            "content_type": self.content_type,
            "snippet": self.snippet,
            "score": self.score,
            "dense_score": self.dense_score,
            "bm25_score": self.bm25_score,
            "rerank_score": self.rerank_score,
            "sources": self.sources,
            "is_neighbor": self.is_neighbor,
        }
        if include_text:
            data["text"] = truncate(self.text, text_limit)
        return data


@dataclass
class TraceStep:
    name: str
    label: str
    status: str = "ok"  # ok | skipped | retry | error
    duration_ms: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


@dataclass
class CitationReport:
    total: int = 0
    supported: int = 0
    unsupported: int = 0
    coverage: float = 0.0
    accuracy: float = 1.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "details": self.details,
        }


@dataclass
class AgentAnswer:
    message_id: str
    session_id: str
    answer: str
    sources: list[Source]
    confidence: float
    confidence_label: str
    citation_report: CitationReport
    plan: QueryPlan
    trace: list[TraceStep]
    latency_ms: int
    cached: bool = False
    retry_used: bool = False
    intent: str = Intent.KB_QUESTION.value
    usage: dict[str, Any] = field(default_factory=dict)
    retrieved_ids: list[str] = field(default_factory=list)

    def to_dict(self, *, include_trace: bool = True, include_text: bool = True) -> dict[str, Any]:
        data = {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "answer": self.answer,
            "sources": [s.to_dict(include_text=include_text) for s in self.sources],
            "citations": [s.to_dict(include_text=False) for s in self.sources],
            "confidence": round(self.confidence, 4),
            "confidence_label": self.confidence_label,
            "citation_report": self.citation_report.to_dict(),
            "plan": self.plan.to_dict(),
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "retry_used": self.retry_used,
            "intent": self.intent,
            "usage": self.usage,
            "retrieved_ids": self.retrieved_ids,
        }
        if include_trace:
            data["trace"] = [step.to_dict() for step in self.trace]
        return data

    def to_json(self) -> str:
        return json_dumps(self.to_dict())


@dataclass
class IngestResult:
    doc_id: str
    name: str
    status: str  # created | updated | unchanged | duplicate | failed
    chunk_count: int = 0
    version: int = 0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "name": self.name,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "version": self.version,
            "message": self.message,
            "metadata": self.metadata,
        }
