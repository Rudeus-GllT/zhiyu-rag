"""The online agent pipeline.

Flow
----
session → layered cache → query analysis / routing → multi-query hybrid recall
→ rerank → (low-confidence second recall) → context assembly → generation
→ citation verification → (optional second pass) → persist + cache.

The same service backs the web console, evaluation runner and the reserved
mini-program gateway, so all clients share one contract and one brain.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from app.config import Settings
from app.core.cache import CacheHit, ResponseCache
from app.core.types import (
    AgentAnswer,
    CitationReport,
    Intent,
    QueryPlan,
    RetrievedChunk,
    Source,
    TraceStep,
)
from app.core.utils import keyword_overlap, tokenize, truncate
from app.llm.service import LLMService
from app.retrieval.context import ContextBundle, ContextBuilder
from app.retrieval.hybrid import HybridRetriever, RetrievalStats
from app.retrieval.reranker import Reranker
from app.retrieval.verifier import (
    CitationVerifier,
    needs_second_pass,
    retrieval_confidence,
)
from app.storage.db import Database
from app.agent.tools import meta_answer

logger = logging.getLogger(__name__)


@dataclass
class Prepared:
    question: str
    answer_question: str
    session_id: str
    user_message_id: str
    message_id: str
    plan: QueryPlan
    mode: str  # rag | direct | no_context | cached
    trace: list[TraceStep] = field(default_factory=list)
    direct_text: str = ""
    sources: list[RetrievedChunk] = field(default_factory=list)
    bundle: ContextBundle | None = None
    stats: RetrievalStats | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    retried: bool = False
    retrieval_conf: float = 0.0
    relevance: float = 0.0
    top_k: int = 8
    doc_ids: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    cached: AgentAnswer | None = None


class AgentService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        index,
        llm: LLMService,
        retriever: HybridRetriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        verifier: CitationVerifier,
        cache: ResponseCache,
    ):
        self.settings = settings
        self.db = db
        self.index = index
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.verifier = verifier
        self.cache = cache

    # ------------------------------------------------------------------ public
    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        debug: bool = False,
        client: str = "web",
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> AgentAnswer:
        prepared = self._prepare(
            question,
            session_id=session_id,
            top_k=top_k,
            doc_ids=doc_ids,
            client=client,
            use_cache=use_cache,
            request_id=request_id,
        )
        if prepared.mode == "cached" and prepared.cached:
            return prepared.cached
        if prepared.mode in {"direct", "no_context"}:
            verification = self.verifier.verify(
                prepared.direct_text, [], intent=prepared.plan.intent.value
            )
            confidence = 0.88 if prepared.plan.intent == Intent.META else 0.82
            if prepared.mode == "no_context":
                confidence = 0.12
            verification.confidence = confidence
            verification.label = "高" if confidence >= 0.7 else "低"
            verification.report = CitationReport()
            return self._finalize(
                prepared,
                prepared.direct_text,
                verification,
                provider="template",
                usage={},
                request_id=request_id,
            )

        evidence_sources = prepared.bundle.sources if prepared.bundle else []
        generation = self.llm.generate(
            prepared.answer_question,
            prepared.bundle.text if prepared.bundle else "",
            prepared.history,
            evidence_sources,
        )
        verification = self.verifier.verify(
            generation.text,
            evidence_sources,
            question=prepared.question,
            intent=prepared.plan.intent.value,
        )
        self._trace(
            prepared.trace,
            "generate",
            "答案生成",
            0,
            detail={
                "provider": generation.provider,
                "tokens": generation.usage.get("total_tokens", 0),
                "chars": len(generation.text),
            },
        )
        self._trace(
            prepared.trace,
            "citation",
            "引用核查",
            0,
            detail=verification.report.to_dict(),
        )

        # Post-generation guard: one extra recall+answer round when evidence
        # is weak.  Streaming skips regeneration to avoid duplicated output.
        if (
            not prepared.retried
            and evidence_sources
            and needs_second_pass(
                verification.confidence,
                verification.retrieval_confidence,
                evidence_sources,
                already_retried=False,
                min_confidence=self.settings.min_confidence,
                retrieval_threshold=self.settings.retrieval_retry_confidence,
            )
        ):
            second = self._second_pass(prepared)
            if second:
                second_sources, second_bundle, second_stats, second_retrieval_conf = second
                second_evidence = second_bundle.sources
                generation2 = self.llm.generate(
                    prepared.answer_question,
                    second_bundle.text,
                    prepared.history,
                    second_evidence,
                )
                verification2 = self.verifier.verify(
                    generation2.text,
                    second_evidence,
                    question=prepared.question,
                    intent=prepared.plan.intent.value,
                )
                adopted = verification2.confidence > verification.confidence + 0.02
                self._trace(
                    prepared.trace,
                    "second_pass",
                    "低置信度二次检索",
                    0,
                    status="retry",
                    detail={
                        "adopted": adopted,
                        "before": round(verification.confidence, 3),
                        "after": round(verification2.confidence, 3),
                        "new_chunks": len(second_sources),
                    },
                )
                if adopted:
                    prepared.sources = second_sources
                    prepared.bundle = second_bundle
                    prepared.stats = second_stats
                    prepared.retrieval_conf = second_retrieval_conf
                    generation = generation2
                    verification = verification2
                prepared.retried = True

        return self._finalize(
            prepared,
            generation.text,
            verification,
            provider=generation.provider,
            usage=generation.usage,
            request_id=request_id,
            cacheable=not debug,
        )

    def stream(
        self,
        question: str,
        *,
        session_id: str | None = None,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        client: str = "web",
        use_cache: bool = True,
        request_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        prepared = self._prepare(
            question,
            session_id=session_id,
            top_k=top_k,
            doc_ids=doc_ids,
            client=client,
            use_cache=use_cache,
            request_id=request_id,
        )
        yield {
            "event": "meta",
            "data": {
                "message_id": prepared.message_id,
                "session_id": prepared.session_id,
                "intent": prepared.plan.intent.value,
                "mode": prepared.mode,
                "retried": prepared.retried,
                "cached": prepared.mode == "cached",
            },
        }
        for step in prepared.trace:
            yield {"event": "trace", "data": step.to_dict()}

        if prepared.mode == "cached" and prepared.cached:
            answer = prepared.cached
            for idx in range(0, len(answer.answer), 18):
                yield {"event": "token", "data": {"text": answer.answer[idx : idx + 18]}}
            yield {"event": "sources", "data": self._sources_payload(answer)}
            yield {
                "event": "done",
                "data": {
                    "message_id": answer.message_id,
                    "session_id": answer.session_id,
                    "latency_ms": answer.latency_ms,
                    "cached": True,
                },
            }
            return

        if prepared.mode in {"direct", "no_context"}:
            text = prepared.direct_text
            for idx in range(0, len(text), 18):
                yield {"event": "token", "data": {"text": text[idx : idx + 18]}}
            verification = self.verifier.verify(text, [], intent=prepared.plan.intent.value)
            confidence = 0.88 if prepared.plan.intent == Intent.META else 0.82
            if prepared.mode == "no_context":
                confidence = 0.12
            verification.confidence = confidence
            verification.label = "高" if confidence >= 0.7 else "低"
            verification.report = CitationReport()
            answer = self._finalize(
                prepared,
                text,
                verification,
                provider="template",
                usage={},
                request_id=request_id,
            )
            yield {"event": "sources", "data": self._sources_payload(answer)}
            yield {
                "event": "done",
                "data": {
                    "message_id": answer.message_id,
                    "session_id": answer.session_id,
                    "latency_ms": answer.latency_ms,
                    "cached": False,
                },
            }
            return

        # RAG streaming
        accumulated: list[str] = []
        generation_started = time.perf_counter()
        evidence_sources = prepared.bundle.sources if prepared.bundle else []
        try:
            for token in self.llm.stream_generate(
                prepared.answer_question,
                prepared.bundle.text if prepared.bundle else "",
                prepared.history,
                evidence_sources,
            ):
                accumulated.append(token)
                yield {"event": "token", "data": {"text": token}}
        except Exception as exc:
            logger.exception("流式回答失败: %s", exc)
            yield {"event": "error", "data": {"message": f"生成中断: {exc}"}}
            return

        text = "".join(accumulated).strip()
        generation_ms = int((time.perf_counter() - generation_started) * 1000)
        verification = self.verifier.verify(
            text,
            evidence_sources,
            question=prepared.question,
            intent=prepared.plan.intent.value,
        )
        self._trace(
            prepared.trace,
            "generate",
            "答案生成",
            generation_ms,
            detail={"provider": self.llm.llm.name, "stream": True, "chars": len(text)},
        )
        self._trace(
            prepared.trace,
            "citation",
            "引用核查",
            0,
            detail=verification.report.to_dict(),
        )
        yield {"event": "trace", "data": prepared.trace[-2].to_dict()}
        yield {"event": "trace", "data": prepared.trace[-1].to_dict()}
        answer = self._finalize(
            prepared,
            text,
            verification,
            provider=self.llm.llm.name,
            usage={},
            request_id=request_id,
        )
        yield {"event": "sources", "data": self._sources_payload(answer)}
        yield {
            "event": "done",
            "data": {
                "message_id": answer.message_id,
                "session_id": answer.session_id,
                "latency_ms": answer.latency_ms,
                "cached": False,
            },
        }

    # ---------------------------------------------------------------- prepare
    def _prepare(
        self,
        question: str,
        *,
        session_id: str | None,
        top_k: int | None,
        doc_ids: Sequence[str] | None,
        client: str,
        use_cache: bool,
        request_id: str | None,
    ) -> Prepared:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        top_k = top_k or self.settings.retrieve_top_k
        session = self._ensure_session(session_id, question, client)
        session_id = session["id"]
        history = self.db.get_recent_messages(session_id, limit=6)
        user_message_id = "m_" + uuid.uuid4().hex[:16]
        self.db.add_message(
            {
                "id": user_message_id,
                "session_id": session_id,
                "role": "user",
                "content": question,
                "created_at": None,
                "request_id": request_id,
            }
        )
        prepared = Prepared(
            question=question,
            answer_question=question,
            session_id=session_id,
            user_message_id=user_message_id,
            message_id="m_" + uuid.uuid4().hex[:16],
            plan=QueryPlan(
                intent=Intent.KB_QUESTION, rewritten=question, needs_retrieval=True
            ),
            mode="rag",
            history=history,
            top_k=top_k,
            doc_ids=list(doc_ids or []),
            started=started,
        )

        # Cache lookup happens before any model call.
        if use_cache:
            cache_started = time.perf_counter()
            hit = self.cache.get(
                question, self.db.get_index_version(), top_k, list(doc_ids or [])
            )
            if hit:
                answer = self._from_cache(hit, prepared, request_id)
                prepared.mode = "cached"
                prepared.cached = answer
                prepared.trace.append(
                    TraceStep(
                        name="cache",
                        label="命中缓存",
                        status="ok",
                        duration_ms=int((time.perf_counter() - cache_started) * 1000),
                        detail={"kind": hit.kind, "score": round(hit.score, 4)},
                    )
                )
                prepared.plan = self._plan_from_dict(
                    (hit.value or {}).get("plan") or {}, question
                )
                return prepared
            prepared.trace.append(
                TraceStep(
                    name="cache",
                    label="缓存未命中",
                    status="skipped",
                    duration_ms=int((time.perf_counter() - cache_started) * 1000),
                    detail={"kind": "miss"},
                )
            )

        analyze_started = time.perf_counter()
        plan = self.llm.analyze_query(question, history)
        prepared.plan = plan
        prepared.answer_question = (
            plan.rewritten if plan.rewritten and plan.rewritten != question else question
        )
        self._trace(
            prepared.trace,
            "analyze",
            "查询理解与路由",
            int((time.perf_counter() - analyze_started) * 1000),
            detail={**plan.to_dict(), "fallback": self.llm.last_analysis_fallback},
        )

        if plan.intent == Intent.CHITCHAT:
            prepared.mode = "direct"
            prepared.direct_text = self.llm.chitchat(question)
            self._trace(prepared.trace, "route", "路由：直接回答", 0, detail={"intent": "chitchat"})
            return prepared
        if plan.intent == Intent.META:
            prepared.mode = "direct"
            prepared.direct_text = meta_answer(
                self.db, question, self.db.get_index_version()
            )
            self._trace(prepared.trace, "route", "路由：知识库工具", 0, detail={"intent": "meta"})
            return prepared

        if self.db.chunk_count() == 0:
            prepared.mode = "no_context"
            prepared.direct_text = (
                "知识库目前还没有可检索的内容。请先在「知识库」页上传 PDF、Markdown、"
                "图片或文本文件，然后再来提问。"
            )
            prepared.plan = plan
            self._trace(prepared.trace, "route", "路由：知识库为空", 0, detail={})
            return prepared

        retrieve_started = time.perf_counter()
        sources, bundle, stats = self._retrieve(
            plan, top_k=top_k, extra_queries=(), doc_ids=doc_ids
        )
        retrieval_conf = retrieval_confidence(sources)
        relevance = self._relevance_score(plan, sources)
        prepared.sources = sources
        prepared.bundle = bundle
        prepared.stats = stats
        prepared.retrieval_conf = retrieval_conf
        prepared.relevance = relevance
        self._trace(
            prepared.trace,
            "retrieve",
            "多路混合召回",
            int((time.perf_counter() - retrieve_started) * 1000),
            detail={
                **stats.to_dict(),
                "top_scores": [round(s.rerank_score, 3) for s in sources[:5]],
                "retrieval_confidence": round(retrieval_conf, 3),
                "relevance": round(relevance, 3),
            },
        )

        # Pre-generation low-confidence recall expansion (HyDE + wider top-k).
        if (
            sources
            and (
                retrieval_conf < self.settings.retrieval_retry_confidence
                or relevance < self.settings.min_relevance
            )
            and not prepared.retried
        ):
            retry_started = time.perf_counter()
            hyde_text = self.llm.hyde(plan.rewritten)
            sources2, bundle2, stats2 = self._retrieve(
                plan,
                top_k=max(top_k * 2, top_k + 4),
                extra_queries=(hyde_text,),
                doc_ids=doc_ids,
            )
            merged = self._merge_candidates(sources, sources2)
            ranked = self.reranker.rerank(
                plan.rewritten,
                merged,
                top_n=self.settings.rerank_top_n,
                expansion=" ".join(plan.subqueries),
            )
            bundle3 = self.context_builder.build(ranked)
            retrieval_conf2 = retrieval_confidence(ranked)
            relevance2 = self._relevance_score(plan, ranked)
            prepared.retried = True
            prepared.sources = ranked
            prepared.bundle = bundle3
            prepared.stats = stats2
            prepared.retrieval_conf = retrieval_conf2
            prepared.relevance = relevance2
            self._trace(
                prepared.trace,
                "retry",
                "低置信度二次召回",
                int((time.perf_counter() - retry_started) * 1000),
                status="retry",
                detail={
                    "strategy": "hyde+expanded_topk",
                    "before": round(retrieval_conf, 3),
                    "after": round(retrieval_conf2, 3),
                    "relevance_before": round(relevance, 3),
                    "relevance_after": round(relevance2, 3),
                    "merged_chunks": len(merged),
                },
            )
            sources = ranked
            bundle = bundle3
            relevance = relevance2

        if not sources or relevance < self.settings.min_relevance:
            prepared.mode = "no_context"
            prepared.sources = []
            prepared.bundle = None
            prepared.direct_text = (
                "我在知识库中没有找到与这个问题直接相关的内容。\n\n"
                "可以尝试：\n"
                "1. 换用更具体的术语或文档名；\n"
                "2. 上传相关的 PDF、Markdown 或图片；\n"
                "3. 在提问中补充时间、对象或限定条件。"
            )
            self._trace(
                prepared.trace,
                "route",
                "路由：无相关片段",
                0,
                detail={"relevance": round(relevance, 3)},
            )
            return prepared

        prepared.mode = "rag"
        self._trace(
            prepared.trace,
            "context",
            "上下文拼装",
            0,
            detail={
                "chunks": len(bundle.sources),
                "tokens": bundle.tokens,
                "expanded": len(bundle.expanded),
            },
        )
        return prepared

    def _retrieve(
        self,
        plan: QueryPlan,
        *,
        top_k: int,
        extra_queries: Sequence[str],
        doc_ids: Sequence[str] | None,
    ) -> tuple[list[RetrievedChunk], ContextBundle, RetrievalStats]:
        candidates, stats = self.retriever.retrieve(
            plan,
            top_k=top_k,
            extra_queries=extra_queries,
            doc_ids=doc_ids,
        )
        expansion = " ".join(plan.subqueries)
        ranked = self.reranker.rerank(
            plan.rewritten,
            candidates,
            top_n=self.settings.rerank_top_n,
            expansion=expansion,
        )
        bundle = self.context_builder.build(ranked)
        return ranked, bundle, stats

    def _second_pass(
        self, prepared: Prepared
    ) -> tuple[list[RetrievedChunk], ContextBundle, RetrievalStats, float] | None:
        try:
            hyde_text = self.llm.hyde(prepared.plan.rewritten)
            sources2, bundle2, stats2 = self._retrieve(
                prepared.plan,
                top_k=max(prepared.top_k * 2, prepared.top_k + 4),
                extra_queries=(hyde_text,),
                doc_ids=None,
            )
            merged = self._merge_candidates(prepared.sources, sources2)
            ranked = self.reranker.rerank(
                prepared.plan.rewritten,
                merged,
                top_n=self.settings.rerank_top_n,
                expansion=" ".join(prepared.plan.subqueries),
            )
            bundle3 = self.context_builder.build(ranked)
            return ranked, bundle3, stats2, retrieval_confidence(ranked)
        except Exception as exc:
            logger.warning("二次检索失败: %s", exc)
            return None

    _GENERIC_QUERY_TOKENS = {
        "pdf", "markdown", "md", "文档", "资料", "知识库", "检索", "内容",
        "系统", "如何", "什么", "怎么", "哪些", "介绍", "相关",
    }

    @classmethod
    def _relevance_score(cls, plan: QueryPlan, sources: Sequence[RetrievedChunk]) -> float:
        """Absolute relevance gate against off-topic questions.

        Hash embeddings produce low absolute cosine values, so the gate blends
        lexical coverage (which is reliable for Chinese technical notes) with
        the dense score.  A query that matches only generic words such as
        “PDF” or “文档” is heavily discounted unless semantics are strong.
        """
        if not sources:
            return 0.0
        query_text = f"{plan.rewritten} {' '.join(plan.keywords)}"
        query_tokens = set(tokenize(query_text, with_bigrams=True))
        top = sources[:3]
        dense = max((s.dense_score for s in top), default=0.0)
        coverage = max(
            (keyword_overlap(query_tokens, s.chunk.text) for s in top), default=0.0
        )
        heading = max(
            (keyword_overlap(query_tokens, s.chunk.heading) for s in top), default=0.0
        )
        doc_tokens: set[str] = set()
        for source in top:
            doc_tokens.update(tokenize(source.chunk.text + " " + source.chunk.heading, with_bigrams=True))
        informative = [
            token
            for token in query_tokens
            if token not in cls._GENERIC_QUERY_TOKENS and token in doc_tokens
        ]
        score = 0.50 * coverage + 0.30 * min(1.0, dense) + 0.20 * heading
        if coverage >= 0.15:
            score += 0.08
        if not informative and dense < 0.3:
            score *= 0.4
        elif not informative:
            score *= 0.75
        return max(0.0, min(1.0, score))

    @staticmethod
    def _merge_candidates(
        first: Sequence[RetrievedChunk], second: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        for candidate in [*first, *second]:
            existing = merged.get(candidate.chunk.id)
            if existing is None:
                merged[candidate.chunk.id] = candidate
            else:
                existing.dense_score = max(existing.dense_score, candidate.dense_score)
                existing.bm25_score = max(existing.bm25_score, candidate.bm25_score)
                existing.rrf_score = max(existing.rrf_score, candidate.rrf_score)
                existing.sources = sorted(set(existing.sources) | set(candidate.sources))
        return list(merged.values())

    # ---------------------------------------------------------------- session
    def _ensure_session(
        self, session_id: str | None, question: str, client: str
    ) -> dict[str, Any]:
        title = truncate(question.replace("\n", " "), 32)
        if session_id:
            session = self.db.get_session(session_id)
            if session:
                if not session.get("title") or session.get("title") == "新对话":
                    self.db.touch_session(session_id, title=title)
                else:
                    self.db.touch_session(session_id)
                return session
        session_id = session_id or ("s_" + uuid.uuid4().hex[:16])
        return self.db.create_session(
            session_id, title=title or "新对话", client=client
        )

    # ---------------------------------------------------------------- finalize
    def _finalize(
        self,
        prepared: Prepared,
        answer_text: str,
        verification,
        *,
        provider: str,
        usage: dict[str, Any],
        request_id: str | None,
        cacheable: bool = True,
    ) -> AgentAnswer:
        latency = int((time.perf_counter() - prepared.started) * 1000)
        sources: list[Source] = []
        if prepared.bundle and prepared.mode == "rag":
            sources = [
                candidate.to_source(index)
                for index, candidate in enumerate(prepared.bundle.sources, start=1)
            ]
        confidence = verification.confidence
        if prepared.mode in {"direct", "no_context"}:
            confidence = verification.confidence
        answer = AgentAnswer(
            message_id=prepared.message_id,
            session_id=prepared.session_id,
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            confidence_label=verification.label,
            citation_report=verification.report,
            plan=prepared.plan,
            trace=prepared.trace,
            latency_ms=latency,
            cached=False,
            retry_used=prepared.retried,
            intent=prepared.plan.intent.value,
            usage={
                **usage,
                "provider": provider,
                "context_tokens": prepared.bundle.tokens if prepared.bundle else 0,
            },
            retrieved_ids=[candidate.chunk.id for candidate in prepared.sources],
        )
        self._trace(
            prepared.trace,
            "persist",
            "写入会话",
            0,
            detail={"message_id": answer.message_id},
        )
        self.db.add_message(
            {
                "id": answer.message_id,
                "session_id": answer.session_id,
                "role": "assistant",
                "content": answer_text,
                "citations": [s.to_dict(include_text=False) for s in sources],
                "trace": [step.to_dict() for step in prepared.trace],
                "confidence": confidence,
                "latency_ms": latency,
                "cached": False,
                "request_id": request_id,
            }
        )
        if (
            cacheable
            and prepared.mode != "no_context"
            and prepared.mode != "direct"
            and confidence >= 0.35
        ):
            self.cache.set(
                prepared.question,
                self.db.get_index_version(),
                prepared.top_k,
                prepared.doc_ids,
                self._cache_payload(answer),
                confidence=confidence,
            )
        return answer

    def _cache_payload(self, answer: AgentAnswer) -> dict[str, Any]:
        return {
            "answer": answer.answer,
            "sources": [s.to_dict(include_text=True, text_limit=900) for s in answer.sources],
            "confidence": answer.confidence,
            "confidence_label": answer.confidence_label,
            "citation_report": answer.citation_report.to_dict(),
            "plan": answer.plan.to_dict(),
            "retry_used": answer.retry_used,
            "intent": answer.intent,
            "usage": answer.usage,
            "retrieved_ids": answer.retrieved_ids,
        }

    def _from_cache(
        self, hit: CacheHit, prepared: Prepared, request_id: str | None
    ) -> AgentAnswer:
        data = hit.value or {}
        sources = [self._source_from_dict(item) for item in (data.get("sources") or [])]
        report_data = data.get("citation_report") or {}
        report = CitationReport(
            total=int(report_data.get("total", 0)),
            supported=int(report_data.get("supported", 0)),
            unsupported=int(report_data.get("unsupported", 0)),
            coverage=float(report_data.get("coverage", 0.0)),
            accuracy=float(report_data.get("accuracy", 0.0)),
            details=report_data.get("details") or [],
        )
        plan = self._plan_from_dict(data.get("plan") or {}, prepared.question)
        latency = int((time.perf_counter() - prepared.started) * 1000)
        answer = AgentAnswer(
            message_id=prepared.message_id,
            session_id=prepared.session_id,
            answer=str(data.get("answer") or ""),
            sources=sources,
            confidence=float(data.get("confidence") or 0.0),
            confidence_label=str(data.get("confidence_label") or "中"),
            citation_report=report,
            plan=plan,
            trace=[
                TraceStep(
                    name="cache",
                    label=f"命中{'语义' if hit.kind == 'semantic' else '精确'}缓存",
                    detail={"kind": hit.kind, "score": round(hit.score, 4)},
                )
            ],
            latency_ms=latency,
            cached=True,
            retry_used=bool(data.get("retry_used")),
            intent=str(data.get("intent") or plan.intent.value),
            usage={**(data.get("usage") or {}), "provider": "cache"},
            retrieved_ids=list(data.get("retrieved_ids") or []),
        )
        self.db.add_message(
            {
                "id": answer.message_id,
                "session_id": answer.session_id,
                "role": "assistant",
                "content": answer.answer,
                "citations": [s.to_dict(include_text=False) for s in sources],
                "trace": [step.to_dict() for step in answer.trace],
                "confidence": answer.confidence,
                "latency_ms": latency,
                "cached": True,
                "request_id": request_id,
            }
        )
        return answer

    @staticmethod
    def _source_from_dict(data: dict[str, Any]) -> Source:
        return Source(
            index=int(data.get("index") or 0),
            chunk_id=str(data.get("chunk_id") or ""),
            doc_id=str(data.get("doc_id") or ""),
            doc_name=str(data.get("doc_name") or ""),
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            heading=str(data.get("heading") or ""),
            content_type=str(data.get("content_type") or "text"),
            text=str(data.get("text") or data.get("snippet") or ""),
            snippet=str(data.get("snippet") or ""),
            score=float(data.get("score") or 0.0),
            dense_score=float(data.get("dense_score") or 0.0),
            bm25_score=float(data.get("bm25_score") or 0.0),
            rerank_score=float(data.get("rerank_score") or 0.0),
            sources=list(data.get("sources") or []),
            is_neighbor=bool(data.get("is_neighbor")),
        )

    @staticmethod
    def _plan_from_dict(data: dict[str, Any], question: str) -> QueryPlan:
        try:
            intent = Intent(str(data.get("intent") or "kb_question"))
        except ValueError:
            intent = Intent.KB_QUESTION
        return QueryPlan(
            intent=intent,
            rewritten=str(data.get("rewritten") or question),
            subqueries=list(data.get("subqueries") or []),
            keywords=list(data.get("keywords") or []),
            needs_retrieval=bool(data.get("needs_retrieval", True)),
            doc_ids=list(data.get("doc_ids") or []),
            reason=str(data.get("reason") or ""),
            source=str(data.get("source") or "cache"),
        )

    # --------------------------------------------------------------- payloads
    @staticmethod
    def _sources_payload(answer: AgentAnswer) -> dict[str, Any]:
        return {
            "sources": [s.to_dict(include_text=False) for s in answer.sources],
            "citations": [s.to_dict(include_text=False) for s in answer.sources],
            "confidence": round(answer.confidence, 4),
            "confidence_label": answer.confidence_label,
            "citation_report": answer.citation_report.to_dict(),
            "retry_used": answer.retry_used,
            "cached": answer.cached,
            "plan": answer.plan.to_dict(),
            "latency_ms": answer.latency_ms,
        }

    @staticmethod
    def _trace(
        trace: list[TraceStep],
        name: str,
        label: str,
        duration_ms: int,
        *,
        status: str = "ok",
        detail: dict[str, Any] | None = None,
    ) -> None:
        trace.append(
            TraceStep(
                name=name,
                label=label,
                status=status,
                duration_ms=max(0, int(duration_ms)),
                detail=detail or {},
            )
        )
