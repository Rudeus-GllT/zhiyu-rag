"""Citation verification and confidence estimation.

This is the guardrail that turns "looks like a RAG answer" into "every claim
points at a chunk we actually retrieved".  It works without an LLM: sentence
tokens are checked against the cited chunk, with exact-quote support as a
strong signal.  An optional LLM judge can be enabled for stricter checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.core.types import CitationReport, RetrievedChunk
from app.core.utils import keyword_overlap, split_sentences, tokenize, truncate

_CITATION = re.compile(r"\[(\d{1,2})\]")
_QUOTED = re.compile(r"[“\"「『]([^”\"」』]{6,80})[”\"」』]")
_TABLE_SEPARATOR = re.compile(r"^\|?[\s:\-|]+\|?$")


def _is_noise_claim(sentence: str) -> bool:
    if len(sentence.strip()) < 8:
        return True
    if sentence.startswith(("```", "【关键词】")):
        return True
    if _TABLE_SEPARATOR.match(sentence):
        return True
    lowered = sentence.lower()
    if '"id"' in lowered or '"question"' in lowered or '"doc_name"' in lowered:
        return True
    if sentence.count("{") + sentence.count("}") >= 2:
        return True
    if " / " in sentence and len(sentence) < 100 and not re.search(r"[。！？；，,]", sentence):
        return True
    return False


@dataclass
class VerificationResult:
    report: CitationReport
    confidence: float
    retrieval_confidence: float
    label: str
    unsupported_sentences: list[str]

    def to_dict(self) -> dict:
        return {
            "report": self.report.to_dict(),
            "confidence": round(self.confidence, 4),
            "label": self.label,
            "retrieval_confidence": round(self.retrieval_confidence, 4),
        }


class CitationVerifier:
    def __init__(self, min_support: float = 0.24, min_quote_len: int = 8):
        self.min_support = min_support
        self.min_quote_len = min_quote_len

    def verify(
        self,
        answer: str,
        sources: Sequence[RetrievedChunk],
        *,
        question: str = "",
        intent: str = "kb_question",
    ) -> VerificationResult:
        by_index = {idx: source for idx, source in enumerate(sources, start=1)}
        sentences = split_sentences(answer) or [answer]
        declarative = [s for s in sentences if not _is_noise_claim(s)]
        cited_sentences = 0
        total = 0
        supported = 0
        unsupported_sentences: list[str] = []
        details: list[dict] = []

        for sentence in declarative:
            citations = [int(x) for x in _CITATION.findall(sentence)]
            if not citations:
                continue
            cited_sentences += 1
            sentence_text = _CITATION.sub("", sentence).strip()
            sentence_tokens = tokenize(sentence_text, with_bigrams=True)
            plain_tokens = tokenize(sentence_text, with_bigrams=False)
            quotes = [q for q in _QUOTED.findall(sentence_text) if len(q) >= self.min_quote_len]
            sentence_ok = False
            for citation in dict.fromkeys(citations):
                total += 1
                source = by_index.get(citation)
                if not source:
                    details.append(
                        {
                            "sentence": truncate(sentence, 180),
                            "citation": citation,
                            "supported": False,
                            "overlap": 0.0,
                            "reason": "引用编号不存在",
                        }
                    )
                    continue
                overlap = keyword_overlap(sentence_tokens, source.chunk.text)
                quote_hit = any(quote in source.chunk.text for quote in quotes)
                # Short claims are easy to "support" by sharing only an entity
                # name, so require stronger lexical agreement for them.
                threshold = 0.5 if len(plain_tokens) <= 7 else max(self.min_support, 0.3)
                ok = overlap >= threshold or quote_hit
                if ok:
                    supported += 1
                    sentence_ok = True
                details.append(
                    {
                        "sentence": truncate(sentence_text, 180),
                        "citation": citation,
                        "chunk_id": source.chunk.id,
                        "doc_name": source.chunk.doc_name,
                        "supported": ok,
                        "overlap": round(overlap, 3),
                        "quote_hit": quote_hit,
                    }
                )
            if not sentence_ok:
                unsupported_sentences.append(truncate(sentence_text, 160))

        accuracy = supported / total if total else 0.0
        coverage = cited_sentences / len(declarative) if declarative else 0.0
        report = CitationReport(
            total=total,
            supported=supported,
            unsupported=max(0, total - supported),
            coverage=coverage,
            accuracy=accuracy,
            details=details[:24],
        )
        retrieval_conf = retrieval_confidence(sources)
        confidence = compute_confidence(
            sources, report, retrieval_conf, intent=intent, question=question
        )
        return VerificationResult(
            report=report,
            confidence=confidence,
            retrieval_confidence=retrieval_conf,
            label=confidence_label(confidence),
            unsupported_sentences=list(dict.fromkeys(unsupported_sentences))[:8],
        )


def retrieval_confidence(sources: Sequence[RetrievedChunk]) -> float:
    if not sources:
        return 0.0
    top = max(0.0, min(1.0, sources[0].rerank_score or sources[0].score))
    top3 = sum(max(0.0, s.rerank_score) for s in sources[:3]) / min(3, len(sources))
    breadth = min(1.0, len(sources) / 4.0)
    return max(0.0, min(1.0, 0.62 * top + 0.23 * top3 + 0.15 * breadth))


def compute_confidence(
    sources: Sequence[RetrievedChunk],
    report: CitationReport,
    retrieval_conf: float,
    *,
    intent: str = "kb_question",
    question: str = "",
) -> float:
    if not sources:
        return 0.05
    top = max(0.0, min(1.0, sources[0].rerank_score or sources[0].score))
    base = (
        0.40 * top
        + 0.20 * retrieval_conf
        + 0.18 * report.accuracy
        + 0.12 * report.coverage
        + 0.10 * min(1.0, len(sources) / 3.0)
    )
    if report.total == 0:
        base *= 0.64
    if report.unsupported:
        base -= 0.035 * min(4, report.unsupported)
    if intent == "chitchat":
        base = max(base, 0.8)
    return max(0.05, min(0.96, base))


def confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "高"
    if confidence >= 0.5:
        return "中"
    return "低"


def needs_second_pass(
    confidence: float,
    retrieval_conf: float,
    sources: Sequence[RetrievedChunk],
    *,
    already_retried: bool,
    min_confidence: float,
    retrieval_threshold: float,
) -> bool:
    if already_retried:
        return False
    if not sources:
        return True
    return confidence < min_confidence or retrieval_conf < retrieval_threshold
