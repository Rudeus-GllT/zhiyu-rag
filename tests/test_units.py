from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.types import Block, ParsedDocument, QueryPlan, RetrievedChunk, Chunk, Intent
from app.core.utils import estimate_tokens, reciprocal_rank_fusion
from app.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, token_f1
from app.ingest.chunker import chunk_document
from app.ingest.vision import VisionService
from app.llm.base import ChatResult
from app.llm.heuristics import detect_intent, plan_query, condense_question
from app.llm.service import LLMService
from app.retrieval.verifier import CitationVerifier

ROOT = Path(__file__).resolve().parents[1]


def _settings():
    class S:
        chunk_size = 300
        chunk_overlap = 40
        max_chunk_chars = 600
        min_chunk_chars = 40
    return S()


def test_chunker_preserves_heading_path():
    doc = ParsedDocument(
        title="测试",
        source_type="markdown",
        blocks=[
            Block(text="第一段内容，介绍主题。", heading_path=["总览"], content_type="text"),
            Block(text="第二段内容，继续解释。", heading_path=["总览", "细节"], content_type="text"),
            Block(text="x = 1", heading_path=["总览", "细节"], content_type="code"),
        ],
    )
    chunks = chunk_document(doc, doc_id="d1", doc_name="测试.md", settings=_settings())
    assert chunks
    assert chunks[0].heading_path == ["总览"]
    assert any(c.content_type == "code" for c in chunks)
    assert all(c.token_estimate > 0 for c in chunks)
    assert all(c.id.startswith("c_") for c in chunks)


def test_rrf_and_metrics():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused["b"] > fused["a"]
    relevant = {"b"}
    assert hit_at_k(["a", "b"], relevant, 5) == 1.0
    assert mrr_at_k(["a", "b"], relevant, 5) == 0.5
    assert 0 < ndcg_at_k(["a", "b"], relevant, 5) <= 1
    assert token_f1("混合检索融合", "混合检索融合") == 1.0


def test_query_heuristics():
    assert detect_intent("你好，你能做什么？") == Intent.CHITCHAT
    assert detect_intent("知识库有哪些文档？") == Intent.META
    assert detect_intent("Chroma 和 BM25 有什么区别？") == Intent.COMPARE
    plan = plan_query("混合检索是怎么做融合的？")
    assert plan.intent == Intent.KB_QUESTION
    assert plan.subqueries
    history = [{"role": "user", "content": "增量索引是怎么设计的？"}]
    assert "增量索引" in condense_question("它怎么处理删除？", history)


def test_verifier_flags_unsupported_citation():
    chunk = Chunk(
        id="c1",
        doc_id="d1",
        doc_name="note.md",
        ordinal=0,
        text="蓝鲸协议包含创建、更新和删除三个步骤。",
        heading_path=["协议"],
        content_type="text",
        token_estimate=20,
    )
    source = RetrievedChunk(chunk=chunk, score=0.8, rerank_score=0.8)
    verifier = CitationVerifier()
    good = verifier.verify("蓝鲸协议包含创建、更新和删除三个步骤[1]。", [source])
    assert good.report.total == 1
    assert good.report.supported == 1
    bad = verifier.verify("蓝鲸协议由火星人发明[1]。", [source])
    assert bad.report.unsupported == 1
    assert bad.confidence < good.confidence


def test_mixed_text_and_vision_endpoints():
    settings = Settings(
        llm_provider="openai",
        llm_api_key="deepseek-key",
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-chat",
        llm_vision_model="qwen3-vl-plus",
        llm_vision_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_vision_api_key="bailian-key",
        enable_vlm=True,
        embedding_provider="hash",
    )
    service = LLMService(settings)
    assert service.llm.base_url == "https://api.deepseek.com/v1"
    assert service.llm.model == "deepseek-chat"
    assert service.vision_llm.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert service.vision_llm.model == "qwen3-vl-plus"
    assert service.vision_llm.api_key == "bailian-key"
    assert service.health()["vision"] is True
    assert service.vision.status()["vlm_enabled"] is True
    assert (
        service.vision.status()["vlm_base_url"]
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_vision_endpoint_requires_its_own_key():
    settings = Settings(
        llm_provider="openai",
        llm_api_key="deepseek-key",
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-chat",
        llm_vision_model="qwen3-vl-plus",
        llm_vision_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_vision_api_key="",
        enable_vlm=True,
        embedding_provider="hash",
    )
    service = LLMService(settings)
    assert service.vision_llm.is_mock
    assert service.health()["vision"] is False
    assert service.vision.status()["vlm_enabled"] is False


def test_vision_retries_without_json_mode():
    class FakeSettings:
        enable_vlm = True
        ocr_enabled = False
        llm_vision_model = "vision-test"

    class FakeLLM:
        supports_vision = True

        def __init__(self):
            self.calls: list[bool] = []

        def chat(self, messages, *, json_mode=False, **kwargs):
            self.calls.append(json_mode)
            if json_mode:
                raise RuntimeError("400 response_format is not supported")
            return ChatResult(
                text='{"caption":"示意图","text":"检索链路","keywords":["RRF"]}'
            )

    llm = FakeLLM()
    service = VisionService(FakeSettings(), llm)
    result = service.extract(ROOT / "samples" / "images" / "检索链路示意图.png")
    assert result.provider == "vlm"
    assert "检索链路" in result.text
    assert llm.calls == [True, False]
