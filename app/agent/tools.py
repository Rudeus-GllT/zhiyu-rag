"""Small deterministic tools used by the meta route (and exposed for clients)."""

from __future__ import annotations

from app.storage.db import Database


def knowledge_stats_tool(db: Database, index_version: int) -> dict:
    stats = db.stats()
    return {
        "documents": stats["documents"],
        "chunks": stats["chunks"],
        "by_type": stats.get("by_type", {}),
        "sessions": stats.get("sessions", 0),
        "index_version": index_version,
    }


def list_documents_tool(db: Database, limit: int = 30) -> list[dict]:
    documents = db.list_documents()[:limit]
    return [
        {
            "id": doc["id"],
            "name": doc["name"],
            "status": doc.get("status"),
            "chunks": int(doc.get("chunk_count") or 0),
            "type": doc.get("metadata", {}).get("source_type")
            or (doc.get("mime") or "").split("/")[0],
            "updated_at": doc.get("updated_at"),
        }
        for doc in documents
    ]


def meta_answer(db: Database, question: str, index_version: int) -> str:
    stats = knowledge_stats_tool(db, index_version)
    documents = list_documents_tool(db, limit=12)
    if not documents:
        return (
            "知识库目前还没有文档。你可以上传 PDF、Markdown、图片或文本文件，"
            "我会解析、切片并建立 BM25 + 向量混合索引。"
        )
    lines = [
        f"知识库当前共有 **{stats['documents']}** 篇文档、**{stats['chunks']}** 个片段，"
        f"索引版本 v{stats['index_version']}。",
        "",
    ]
    type_labels = {"pdf": "PDF", "markdown": "Markdown", "image": "图片", "other": "其他"}
    if stats.get("by_type"):
        summary = "、".join(
            f"{type_labels.get(k, k)} {v} 篇" for k, v in stats["by_type"].items()
        )
        lines.append(f"类型分布：{summary}。")
        lines.append("")
    lines.append("最近的文档：")
    for idx, doc in enumerate(documents[:10], start=1):
        status = {"ready": "", "error": "（解析失败）", "degraded": "（降级）"}.get(
            doc["status"] or "", ""
        )
        lines.append(f"{idx}. 《{doc['name']}》 · {doc['chunks']} 个片段{status}")
    if len(documents) > 10:
        lines.append(f"… 以及另外 {len(documents) - 10} 篇文档。")
    lines.append("")
    lines.append(
        "你可以直接提问具体内容，例如“混合检索怎么做融合？”或“总结一下多模态解析笔记”。"
    )
    return "\n".join(lines)


TOOL_REGISTRY = [
    {
        "name": "search_knowledge_base",
        "description": "混合检索个人知识库，返回带页码与章节的片段",
        "used_by": "kb_question / summarize / compare",
    },
    {
        "name": "list_documents",
        "description": "列出知识库文档、状态与片段数量",
        "used_by": "meta",
    },
    {
        "name": "knowledge_stats",
        "description": "文档数、片段数、类型分布与索引版本",
        "used_by": "meta",
    },
    {
        "name": "citation_check",
        "description": "核查回答中的 [n] 引用是否有原文支持",
        "used_by": "all RAG answers",
    },
]
