"""High-level LLM facade used by the agent pipeline.

Every method degrades gracefully: if the configured remote model fails, the
heuristic implementation takes over and the trace records the fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from app.config import Settings
from app.core.types import Intent, QueryPlan, RetrievedChunk
from app.core.utils import (
    dedupe,
    parse_json_object,
    truncate,
)
from app.ingest.vision import VisionResult, VisionService
from app.llm.base import BaseLLM
from app.llm.embeddings import create_embedder
from app.llm.heuristics import (
    direct_answer,
    extractive_answer,
    heuristic_hyde,
    heuristic_rerank,
    plan_query,
)
from app.llm.mock import MockLLM
from app.llm.openai_client import OpenAICompatLLM
from app.llm.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    CHITCHAT_SYSTEM,
    HYDE_USER,
    QUERY_ANALYSIS_SYSTEM,
    QUERY_ANALYSIS_USER,
    RERANK_SYSTEM,
    RERANK_USER,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    provider: str = "heuristic"
    usage: dict[str, Any] = field(default_factory=dict)
    fallback: bool = False
    latency_ms: int = 0


def create_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLM()
    if provider in {"openai", "api", "auto"} and settings.llm_key:
        return OpenAICompatLLM(
            api_key=settings.llm_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            vision_model=settings.llm_vision_model,
            timeout=settings.llm_timeout_seconds,
            default_max_tokens=settings.llm_max_tokens,
            default_temperature=settings.llm_temperature,
        )
    if provider in {"openai", "api"} and not settings.llm_key:
        logger.warning("LLM_PROVIDER=%s 但未配置 LLM_API_KEY，使用本地离线模型", provider)
    return MockLLM()


def create_vision_llm(settings: Settings, main_llm: BaseLLM) -> BaseLLM:
    """Build a vision client, optionally against a different provider.

    ``LLM_VISION_MODEL`` is enough when the vision model lives on the same
    endpoint as the text model.  For a mixed setup (for example DeepSeek text
    plus Qwen vision), set ``LLM_VISION_BASE_URL`` and/or
    ``LLM_VISION_API_KEY`` and this returns a dedicated client.
    """
    model = settings.llm_vision_model.strip()
    if not model:
        return main_llm
    if settings.llm_provider.strip().lower() == "mock":
        return main_llm

    has_override = bool(
        settings.llm_vision_base_url.strip() or settings.llm_vision_api_key.strip()
    )
    if not has_override:
        return main_llm

    same_base = settings.vision_base_url == settings.llm_base_url.rstrip("/")
    if (
        settings.llm_vision_base_url.strip()
        and not same_base
        and not settings.llm_vision_api_key.strip()
    ):
        logger.warning(
            "视觉 endpoint 与文本 endpoint 不同，但未配置 LLM_VISION_API_KEY，VLM 不启用"
        )
        return MockLLM()

    if not (settings.vision_key and settings.vision_base_url):
        logger.warning("视觉模型缺少 API Key / Base URL，VLM 不启用")
        return MockLLM()

    logger.info(
        "视觉模型使用独立端点: model=%s base_url=%s",
        model,
        settings.vision_base_url,
    )
    return OpenAICompatLLM(
        api_key=settings.vision_key,
        base_url=settings.vision_base_url,
        model=model,
        vision_model=model,
        timeout=settings.llm_timeout_seconds,
        default_max_tokens=settings.llm_max_tokens,
        default_temperature=settings.llm_temperature,
    )


def format_history(history: Sequence[dict[str, Any]] | None, limit: int = 6) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for message in list(history)[-limit:]:
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        role = "用户" if message.get("role") == "user" else "助手"
        lines.append(f"{role}：{truncate(content, 400)}")
    return "\n".join(lines)


class LLMService:
    def __init__(self, settings: Settings, db=None):
        self.settings = settings
        self.llm = create_llm(settings)
        self.vision_llm = create_vision_llm(settings, self.llm)
        self.embedder = create_embedder(settings, db)
        self.vision = VisionService(settings, self.vision_llm)
        self.is_mock = bool(getattr(self.llm, "is_mock", False))
        self.last_analysis_fallback = False
        self.last_rerank_fallback = True

    # ------------------------------------------------------------------ meta
    def health(self) -> dict[str, Any]:
        return {
            "provider": self.llm.name,
            "model": self.llm.model,
            "mock_mode": self.is_mock,
            "vision": bool(
                self.settings.enable_vlm and self.vision_llm.supports_vision
            ),
            "embedding": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "llm_rerank": bool(self.settings.enable_llm_rerank and not self.is_mock),
            "llm_query_analysis": bool(
                self.settings.enable_llm_query_analysis and not self.is_mock
            ),
            "citation_judge": bool(
                self.settings.enable_llm_citation_judge and not self.is_mock
            ),
            "vision_status": self.vision.status(),
        }

    # ----------------------------------------------------------------- query
    def analyze_query(
        self, question: str, history: Sequence[dict[str, Any]] | None = None
    ) -> QueryPlan:
        fallback = plan_query(
            question, history, max_subqueries=self.settings.max_subqueries
        )
        if self.is_mock or not self.settings.enable_llm_query_analysis:
            return fallback
        history_text = format_history(history) or "（无）"
        messages = [
            {"role": "system", "content": QUERY_ANALYSIS_SYSTEM},
            {
                "role": "user",
                "content": QUERY_ANALYSIS_USER.format(
                    history=history_text, question=question
                ),
            },
        ]
        try:
            result = self.llm.chat(
                messages, json_mode=True, temperature=0.0, max_tokens=600
            )
            data = parse_json_object(result.text)
            if not data:
                raise ValueError("查询分析未返回 JSON")
            intent_value = str(data.get("intent") or "kb_question")
            try:
                intent = Intent(intent_value)
            except ValueError:
                intent = fallback.intent
            rewritten = truncate(str(data.get("rewritten") or "").strip(), 240)
            if not rewritten:
                rewritten = fallback.rewritten
            subqueries = [
                truncate(str(item).strip(), 160)
                for item in (data.get("subqueries") or [])
                if str(item).strip()
            ]
            subqueries = dedupe([s for s in subqueries if s != rewritten])[
                : max(0, self.settings.max_subqueries - 1)
            ]
            keywords = dedupe([str(k).strip() for k in (data.get("keywords") or []) if str(k).strip()])[:8]
            doc_ids = [str(d) for d in (data.get("doc_ids") or []) if str(d).strip()]
            needs = bool(data.get("needs_retrieval", fallback.needs_retrieval))
            self.last_analysis_fallback = False
            return QueryPlan(
                intent=intent,
                rewritten=rewritten,
                subqueries=subqueries,
                keywords=keywords or fallback.keywords,
                needs_retrieval=needs,
                doc_ids=doc_ids,
                reason=str(data.get("reason") or "")[:200],
                source="llm",
            )
        except Exception as exc:
            logger.info("查询分析回退到启发式: %s", exc)
            self.last_analysis_fallback = True
            return fallback

    # ---------------------------------------------------------------- rerank
    def rerank(
        self,
        question: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int,
        *,
        expansion: str = "",
    ) -> list[RetrievedChunk]:
        baseline = heuristic_rerank(
            f"{question} {expansion}".strip(), candidates, top_n=len(candidates)
        )
        if not baseline:
            return []
        if self.is_mock or not self.settings.enable_llm_rerank:
            self.last_rerank_fallback = True
            return baseline[:top_n]

        prompt_candidates = baseline[: min(8, len(baseline))]
        blocks: list[str] = []
        for idx, candidate in enumerate(prompt_candidates, start=1):
            heading = candidate.chunk.heading or "（无章节）"
            blocks.append(
                f"[{idx}] 文档《{candidate.chunk.doc_name}》/ {heading} / p.{candidate.chunk.page_start or '-'}\n"
                f"{truncate(candidate.chunk.text, 420)}"
            )
        messages = [
            {"role": "system", "content": RERANK_SYSTEM},
            {
                "role": "user",
                "content": RERANK_USER.format(
                    question=question, candidates="\n\n".join(blocks)
                ),
            },
        ]
        try:
            result = self.llm.chat(
                messages, json_mode=True, temperature=0.0, max_tokens=900
            )
            data = parse_json_object(result.text) or {}
            scores = data.get("scores") or []
            by_index: dict[int, dict[str, Any]] = {}
            for item in scores:
                try:
                    by_index[int(item.get("id"))] = item
                except Exception:
                    continue
            if not by_index:
                raise ValueError("重排未返回有效分数")
            heuristic_map = {c.chunk.id: c.rerank_score for c in baseline}
            for idx, candidate in enumerate(prompt_candidates, start=1):
                item = by_index.get(idx)
                if not item:
                    candidate.rerank_score = heuristic_map.get(candidate.chunk.id, 0.0)
                    continue
                llm_score = max(0.0, min(10.0, float(item.get("score", 0.0)))) / 10.0
                prior = heuristic_map.get(candidate.chunk.id, 0.0)
                candidate.rerank_score = max(0.0, min(1.0, 0.78 * llm_score + 0.22 * prior))
                candidate.score = candidate.rerank_score
                candidate.rerank_reason = truncate(
                    str(item.get("reason") or "模型重排"), 120
                )
            ranked = sorted(prompt_candidates, key=lambda c: c.rerank_score, reverse=True)
            self.last_rerank_fallback = False
            return ranked[:top_n]
        except Exception as exc:
            logger.info("LLM 重排回退到启发式: %s", exc)
            self.last_rerank_fallback = True
            return baseline[:top_n]

    # -------------------------------------------------------------- generate
    def generate(
        self,
        question: str,
        context: str,
        history: Sequence[dict[str, Any]] | None,
        fallback_sources: Sequence[RetrievedChunk],
    ) -> GenerationResult:
        if self.is_mock:
            return GenerationResult(
                text=extractive_answer(question, fallback_sources),
                provider="local-extractive",
                usage={},
            )
        history_text = format_history(history)
        history_block = f"对话历史：\n{history_text}\n\n" if history_text else ""
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": ANSWER_USER.format(
                    context=context or "（无相关资料）",
                    history_block=history_block,
                    question=question,
                ),
            },
        ]
        started = time.perf_counter()
        try:
            result = self.llm.chat(messages, temperature=self.settings.llm_temperature)
            text = result.text.strip()
            if not text:
                raise ValueError("模型返回空答案")
            return GenerationResult(
                text=text,
                provider=result.model or self.llm.model,
                usage=result.usage,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            logger.warning("生成失败，回退到原文摘录: %s", exc)
            return GenerationResult(
                text=extractive_answer(question, fallback_sources),
                provider="local-extractive",
                usage={},
                fallback=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

    def stream_generate(
        self,
        question: str,
        context: str,
        history: Sequence[dict[str, Any]] | None,
        fallback_sources: Sequence[RetrievedChunk],
    ) -> Iterator[str]:
        if self.is_mock:
            text = extractive_answer(question, fallback_sources)
            for idx in range(0, len(text), 16):
                yield text[idx : idx + 16]
                time.sleep(0.006)
            return
        history_text = format_history(history)
        history_block = f"对话历史：\n{history_text}\n\n" if history_text else ""
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": ANSWER_USER.format(
                    context=context or "（无相关资料）",
                    history_block=history_block,
                    question=question,
                ),
            },
        ]
        produced = False
        try:
            for token in self.llm.stream(
                messages, temperature=self.settings.llm_temperature
            ):
                if token:
                    produced = True
                    yield token
        except Exception as exc:
            logger.warning("流式生成失败: %s", exc)
            if produced:
                raise
            text = extractive_answer(question, fallback_sources)
            for idx in range(0, len(text), 16):
                yield text[idx : idx + 16]

    def chitchat(self, question: str) -> str:
        if self.is_mock:
            return direct_answer(question)
        try:
            result = self.llm.chat(
                [
                    {"role": "system", "content": CHITCHAT_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0.4,
                max_tokens=300,
            )
            return result.text.strip() or direct_answer(question)
        except Exception:
            return direct_answer(question)

    def hyde(self, question: str) -> str:
        if self.is_mock:
            return heuristic_hyde(question)
        try:
            result = self.llm.chat(
                [{"role": "user", "content": HYDE_USER.format(question=question)}],
                temperature=0.3,
                max_tokens=260,
            )
            return result.text.strip() or heuristic_hyde(question)
        except Exception:
            return heuristic_hyde(question)

    # ---------------------------------------------------------------- vision
    def caption_image(self, path: Path, *, allow_vlm: bool = True) -> VisionResult:
        return self.vision.extract(path, allow_vlm=allow_vlm)
