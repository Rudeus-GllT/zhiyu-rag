from __future__ import annotations


DOC = """# 蓝鲸协议

蓝鲸协议是一个用于验证增量索引的虚构协议。
它包含创建、更新和删除三个步骤。

## 更新

更新时先比较文档的 SHA-256，内容变化后才重新解析和嵌入。
""".encode("utf-8")


def test_incremental_index_and_qa(pipeline):
    ctx = pipeline
    created = ctx.index.ingest_bytes(DOC, "blue-whale.md")
    assert created.status == "created"
    assert created.chunk_count > 0
    assert ctx.db.doc_count() == 1

    unchanged = ctx.index.ingest_bytes(DOC, "blue-whale.md")
    assert unchanged.status == "unchanged"

    answer = ctx.agent.ask("蓝鲸协议包含哪些步骤？", use_cache=False)
    assert answer.sources, "retrieval should find the test document"
    assert "蓝鲸协议" in answer.answer
    assert answer.citation_report.total >= 1
    assert 0 <= answer.confidence <= 1

    cached_first = ctx.agent.ask("蓝鲸协议包含哪些步骤？")
    cached_second = ctx.agent.ask("蓝鲸协议包含哪些步骤？")
    assert cached_second.cached
    assert cached_second.answer == cached_first.answer

    updated_doc = DOC + "\n\n## 版本二\n\n新增内容：海豚星云。\n".encode("utf-8")
    updated = ctx.index.ingest_bytes(updated_doc, "blue-whale.md")
    assert updated.status == "updated"
    assert updated.chunk_count >= created.chunk_count

    answer2 = ctx.agent.ask("海豚星云是什么？", use_cache=False)
    assert any("海豚星云" in source.text for source in answer2.sources)

    assert ctx.index.delete_document(updated.doc_id)
    assert ctx.db.doc_count() == 0
    assert ctx.db.chunk_count() == 0
