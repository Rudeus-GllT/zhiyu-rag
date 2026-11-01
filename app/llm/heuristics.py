"""Offline / fallback intelligence.

Every LLM-backed step has a deterministic heuristic here.  This keeps the app
fully functional without an API key and provides a graceful degradation path
when a provider times out.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.core.types import Intent, QueryPlan, RetrievedChunk
from app.core.utils import (
    extract_keywords,
    keyword_overlap,
    min_max_normalize,
    normalize_text,
    split_sentences,
    tokenize,
    truncate,
)

_GREETING = re.compile(r"^(你好|您好|hi|hello|嗨|哈喽|在吗|早上好|晚上好)", re.I)
_CHITCHAT = re.compile(
    r"^(你好|您好|hi|hello|嗨|哈喽|在吗|谢谢|感谢|再见|bye|你是谁|你叫什么|"
    r"你能做什么|你会什么|介绍一下你自己|介绍下你自己)[!！?？。.,，\s]*$",
    re.I,
)
_CAPABILITY = re.compile(
    r"(你能做什么|你会什么|你是谁|你叫什么|介绍一下你自己|介绍下你自己|"
    r"可以做什么|能帮我做什么|有什么功能|怎么用你)",
    re.I,
)
_META = re.compile(
    r"(知识库|资料库|文档库)(里|中)?(有|包含|收录|存了)?(哪些|多少|什么)(文档|资料|文件|笔记)|"
    r"(文档|资料|文件)(列表|清单|数量|统计|概况)|"
    r"(列出|统计|查看).{0,6}(文档|资料|文件)|"
    r"一共有多少(篇|个)?(文档|资料|文件)|知识库多大",
    re.I,
)
_SUMMARIZE = re.compile(r"(总结|概括|摘要|梳理|归纳|提炼|讲了什么|主要内容|要点)", re.I)
_COMPARE = re.compile(r"(对比|比较|区别|差异|优缺点|异同|vs\.?)", re.I)
_FOLLOWUP = re.compile(r"(它|他|她|这个|那个|上述|前面|该|此|这些|那些|其中)")
_FILLERS = [
    "请帮我", "帮我", "麻烦", "请问", "我想知道", "想了解一下", "能不能", "可以", "可以帮我",
    "告诉我", "解释一下", "说明一下", "一下", "谢谢",
]

# A small domain lexicon for the offline lexical retriever.  With a trained
# embedding model this is unnecessary, but it closes common vocabulary gaps
# (e.g. “判断更新” ↔ “SHA-256 / 内容变化”) without any model download.
_QUERY_EXPANSIONS = {
    "更新": ["内容变化", "sha256", "hash"],
    "判断": ["识别", "检测"],
    "删除": ["移除", "delete"],
    "新增": ["添加", "add"],
    "适合": ["擅长", "适用", "分工"],
    "场景": ["用途", "适用"],
    "流程": ["步骤", "链路"],
    "离线": ["索引", "解析"],
    "解析": ["提取", "识别", "ocr"],
    "扫描": ["ocr", "渲染"],
    "融合": ["rrf", "rank fusion"],
    "引用": ["citation", "支持"],
    "缓存": ["cache", "相似"],
    "评测": ["评估", "指标"],
    "重排": ["rerank", "排序"],
    "切片": ["切块", "chunk"],
    "多模态": ["图片", "pdf", "ocr"],
}

CONTENT_TYPE_BOOST = {
    "text": 1.0,
    "table": 1.0,
    "ocr": 0.95,
    "vlm": 0.9,
    "image": 0.9,
    "code": 0.82,
}


def condense_question(question: str, history: Sequence[dict] | None) -> str:
    question = normalize_text(question)
    if not history:
        return question
    previous_user = ""
    for message in reversed(history):
        if message.get("role") == "user":
            previous_user = normalize_text(str(message.get("content") or ""))
            break
    if not previous_user:
        return question
    is_short = len(question) <= 26
    has_pronoun = bool(_FOLLOWUP.search(question))
    topic_keywords = extract_keywords(previous_user, 5)
    if topic_keywords:
        topic = "".join(topic_keywords[:2]) if any(
            re.search(r"[\u4e00-\u9fff]", keyword) for keyword in topic_keywords[:2]
        ) else " ".join(topic_keywords[:2])
    else:
        topic = truncate(previous_user, 40)
    if has_pronoun and is_short:
        rewritten = re.sub(r"(它|他|她|这个|那个|该|此)", topic, question, count=1)
        if rewritten == question:
            rewritten = f"{topic} {question}"
        return rewritten
    if is_short and re.match(r"^(那|那么|还有|另外|如果|删除|更新|新增)", question):
        return f"{topic} {question}"
    return question


def clean_query(question: str) -> str:
    text = normalize_text(question)
    for filler in _FILLERS:
        text = text.replace(filler, "")
    text = re.sub(r"^[，,。.\s]+|[，,。.\s]+$", "", text)
    return text or normalize_text(question)


def detect_intent(question: str) -> Intent:
    q = question.strip()
    if _CHITCHAT.search(q):
        return Intent.CHITCHAT
    if _CAPABILITY.search(q) and len(q) <= 40:
        return Intent.CHITCHAT
    if _GREETING.search(q) and len(q) <= 18:
        return Intent.CHITCHAT
    if _META.search(q):
        return Intent.META
    if _COMPARE.search(q):
        return Intent.COMPARE
    if _SUMMARIZE.search(q):
        return Intent.SUMMARIZE
    return Intent.KB_QUESTION


def plan_query(
    question: str,
    history: Sequence[dict] | None = None,
    *,
    max_subqueries: int = 3,
) -> QueryPlan:
    question = normalize_text(question)
    intent = detect_intent(question)
    rewritten = clean_query(condense_question(question, history))
    if len(rewritten) < 2:
        rewritten = question
    rewritten = truncate(rewritten, 220)

    keywords = extract_keywords(rewritten, 8)
    subqueries: list[str] = []
    keyword_query = " ".join(keywords[:6])
    if keyword_query and keyword_query != rewritten and len(keywords) >= 2:
        subqueries.append(keyword_query)

    expanded: list[str] = []
    for word in keywords:
        for synonym in _QUERY_EXPANSIONS.get(word, []):
            if synonym not in expanded and synonym not in keywords:
                expanded.append(synonym)
    if len(expanded) >= 2:
        expansion_query = " ".join(expanded[:6])
        if expansion_query not in subqueries:
            subqueries.insert(1 if subqueries else 0, expansion_query)

    if intent == Intent.COMPARE:
        parts = re.split(r"\s*(?:和|与|跟|vs\.?|对比|比较)\s*", rewritten, maxsplit=1)
        if len(parts) == 2:
            left = re.sub(r"的?(区别|差异|优缺点|异同).*$", "", parts[0]).strip()
            right = re.sub(r"的?(区别|差异|优缺点|异同).*$", "", parts[1]).strip()
            if left:
                subqueries.append(f"{left} 的特点 适用场景")
            if right:
                subqueries.append(f"{right} 的特点 适用场景")
    elif intent == Intent.SUMMARIZE:
        subqueries.append(f"{rewritten} 核心要点 结论")

    seen = {rewritten}
    unique_subs: list[str] = []
    for sub in subqueries:
        sub = sub.strip()
        if sub and sub not in seen:
            seen.add(sub)
            unique_subs.append(sub)
    unique_subs = unique_subs[: max(0, max_subqueries - 1)]

    needs_retrieval = intent in {Intent.KB_QUESTION, Intent.SUMMARIZE, Intent.COMPARE}
    return QueryPlan(
        intent=intent,
        rewritten=rewritten,
        subqueries=unique_subs,
        keywords=keywords,
        needs_retrieval=needs_retrieval,
        reason={
            Intent.CHITCHAT: "寒暄或能力询问，无需检索",
            Intent.META: "询问知识库文档与统计信息，走工具查询",
            Intent.SUMMARIZE: "文档总结任务，需要先取回相关片段",
            Intent.COMPARE: "对比任务，拆成多路检索",
            Intent.KB_QUESTION: "知识型问题，走混合检索",
        }[intent],
        source="heuristic",
    )


def heuristic_rerank(
    question: str, candidates: Sequence[RetrievedChunk], top_n: int = 6
) -> list[RetrievedChunk]:
    if not candidates:
        return []
    query_tokens = tokenize(question, with_bigrams=True)
    dense_norm = min_max_normalize({c.chunk.id: c.dense_score for c in candidates})
    bm25_norm = min_max_normalize({c.chunk.id: c.bm25_score for c in candidates})
    code_intent = bool(
        re.search(r"(代码|示例|json|接口|api|请求|响应|字段|配置)", question, re.I)
    )
    query_words = [token for token in tokenize(question) if len(token) >= 2]
    query_phrases = [
        query_words[i] + query_words[i + 1]
        for i in range(len(query_words) - 1)
        if len(query_words[i] + query_words[i + 1]) >= 4
    ]
    for candidate in candidates:
        coverage = keyword_overlap(query_tokens, candidate.chunk.text)
        heading_cov = keyword_overlap(query_tokens, candidate.chunk.heading)
        type_boost = CONTENT_TYPE_BOOST.get(candidate.chunk.content_type, 0.85)
        text = candidate.chunk.text
        if candidate.chunk.content_type == "code" and not code_intent:
            type_boost = 0.62
            if re.search(r'"[A-Za-z_][\w-]*"\s*:', text):
                type_boost = 0.4
        length_factor = 1.0 if len(text) >= 80 else 0.82
        score = (
            0.30 * dense_norm.get(candidate.chunk.id, 0.0)
            + 0.24 * bm25_norm.get(candidate.chunk.id, 0.0)
            + 0.34 * coverage
            + 0.07 * heading_cov
            + 0.05 * type_boost
        )
        for phrase in query_phrases:
            if phrase in text or phrase in candidate.chunk.heading:
                score += 0.075
        if (
            candidate.chunk.heading
            and coverage < 0.25
            and not any(word in candidate.chunk.heading for word in query_words)
        ):
            score *= 0.88
        score *= length_factor
        # Question lists / routing examples are poor evidence for a how-to answer.
        question_marks = text.count("?") + text.count("？")
        if question_marks >= 2:
            score *= 0.72
        if text.count("→") >= 2 or (question_marks >= 2 and text.count("“") >= 2):
            score *= 0.78
        if candidate.is_neighbor:
            score *= 0.94
        candidate.rerank_score = max(0.0, min(1.0, score))
        candidate.score = candidate.rerank_score
        candidate.rerank_reason = (
            f"语义 {dense_norm.get(candidate.chunk.id, 0.0):.2f} · "
            f"BM25 {bm25_norm.get(candidate.chunk.id, 0.0):.2f} · "
            f"关键词覆盖 {coverage:.0%}"
        )
    candidates = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    return list(candidates[:top_n])


def heuristic_hyde(question: str) -> str:
    keywords = extract_keywords(question, 8)
    topic = "、".join(keywords) if keywords else question
    return (
        f"围绕“{topic}”，可以从核心概念、实现步骤、关键参数、适用场景与限制五个方面展开。"
        f"通常先明确目标与输入输出，再准备数据与工具链，随后按流程逐步实现，"
        f"并通过指标评估和迭代优化。具体做法需要结合知识库中的资料。"
    )


def direct_answer(question: str) -> str:
    q = question.strip()
    if re.search(r"(你是谁|你叫什么|介绍一下你自己|介绍下你自己)", q):
        return (
            "我是「知屿」，一个面向个人知识管理的多模态 RAG 助手。"
            "我可以检索你的 PDF、Markdown、图片和文本笔记，给出带出处引用的回答，"
            "也能做文档总结、跨文档对比和检索效果评测。"
        )
    if re.search(r"(谢谢|感谢)", q):
        return "不客气。如果还想继续查资料、做总结或对比文档，直接把问题发给我就好。"
    if re.search(r"(再见|bye)", q, re.I):
        return "再见，随时回来继续整理你的知识库。"
    return (
        "你好，我是知屿。你可以直接问我知识库里的内容，例如："
        "“混合检索是怎么做融合的？”、“总结一下多模态解析笔记”、"
        "“Chroma 和 BM25 各适合什么场景？”。"
        "我会先检索资料，再给出带来源编号的回答。"
    )


_NOISE_HEADING = re.compile(r"^#{1,6}\s*")
_TABLE_SEPARATOR = re.compile(r"^\|?[\s:\-|]+\|?$")


def _clean_claim(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = _NOISE_HEADING.sub("", sentence).strip()
    sentence = re.sub(r"^(\d+[.、]|[-*])\s+", "", sentence).strip()
    if sentence.startswith("|"):
        sentence = sentence.strip("| ").replace("|", " · ").strip()
    return sentence.strip()


def _is_noise_sentence(sentence: str) -> bool:
    if len(sentence) < 12:
        return True
    if sentence.startswith(("```", "```text", "【关键词】")):
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
    # Routing examples such as `“你好” → chitchat` are not answers.
    if "→" in sentence and re.search(r"[“\"「]", sentence):
        return True
    if re.fullmatch(r"[\W\d_]+", sentence):
        return True
    if re.match(r"^(表|图)\s*\d+", sentence) and len(sentence) < 50:
        return True
    return False


def extractive_answer(
    question: str,
    sources: Sequence[RetrievedChunk],
    *,
    max_sentences: int = 5,
) -> str:
    if not sources:
        return (
            "知识库中暂时没有检索到能回答这个问题的内容。"
            "你可以先上传相关 PDF、Markdown 或图片，或者换一种问法再试一次。"
        )
    query_tokens = tokenize(question, with_bigrams=True)
    picked: list[tuple[int, int, str, float]] = []
    seen: set[str] = set()
    wants_params = bool(re.search(r"(参数|配置|默认|设置|取值|阈值)", question))
    indexed_sources = list(enumerate(sources[:4], start=1))
    strong_sources = [(idx, s) for idx, s in indexed_sources if s.rerank_score >= 0.55]
    if not strong_sources:
        strong_sources = indexed_sources[:2]
    param_sentence: tuple[int, str] | None = None
    # Only the strongest few sources contribute; the tail is usually noise.
    for source_index, source in strong_sources:
        if wants_params and param_sentence is None:
            pairs = re.findall(
                r"(?m)^\s*([A-Za-z_][A-Za-z0-9_\-]{2,})\s+(\d+(?:\.\d+)?)\b",
                source.chunk.text,
            )
            if len(pairs) >= 2:
                summary = "、".join(f"{key} {value}" for key, value in pairs[:8])
                param_sentence = (source_index, f"默认参数包括：{summary}")
        per_source = 0
        for pos, raw_sentence in enumerate(split_sentences(source.chunk.text)):
            sentence = _clean_claim(raw_sentence)
            if _is_noise_sentence(sentence):
                continue
            key = re.sub(r"\s+", "", sentence)[:40]
            if key in seen:
                continue
            overlap = keyword_overlap(query_tokens, sentence)
            length_bonus = 0.08 if 18 <= len(sentence) <= 180 else -0.03
            score = (
                overlap
                + 0.22 * source.rerank_score
                + length_bonus
                - 0.004 * pos
            )
            if overlap <= 0.02 and source.rerank_score < 0.5:
                continue
            seen.add(key)
            picked.append((source_index, pos, sentence.rstrip("。；; "), score))
            per_source += 1
            if per_source >= 2:
                break
    if param_sentence:
        source_index, sentence = param_sentence
        return f"根据知识库中的资料：\n{sentence}[{source_index}]"
    if not picked:
        return (
            "知识库中找到了相关资料，但资料里没有可直接回答该问题的句子。"
            "建议把问题问得更具体一些，或补充上传相关文档。"
        )
    picked.sort(key=lambda item: item[3], reverse=True)
    selected = picked[:max_sentences]
    selected.sort(key=lambda item: (item[0], item[1]))

    q = question.lower()
    if re.search(r"(如何|怎么|怎样|步骤|流程|怎么做)", q):
        lead = "可以按下面的思路来做："
    elif re.search(r"(是什么|什么是|定义|含义|概念)", q):
        lead = "可以概括为："
    elif re.search(r"(区别|差异|对比|比较)", q):
        lead = "主要区别和各自特点如下："
    elif re.search(r"(为什么|原因)", q):
        lead = "主要原因包括："
    else:
        lead = "根据知识库中的资料："

    lines = [lead]
    for idx, (source_index, _, sentence, _) in enumerate(selected):
        prefix = f"{idx + 1}. " if len(selected) > 2 else ""
        lines.append(f"{prefix}{sentence}[{source_index}]")
    answer = "\n".join(lines)
    if len(answer) > 900:
        answer = truncate(answer, 900)
    return answer
