from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_ctx, require_dashboard
from app.context import AppContext
from app.core.errors import AppError, NotFoundError, UnsupportedFileError
from app.core.utils import human_size

router = APIRouter(
    prefix="/api/v1/documents", tags=["documents"], dependencies=[Depends(require_dashboard)]
)


def _public_document(doc: dict) -> dict:
    metadata = doc.get("metadata") or {}
    return {
        "id": doc["id"],
        "name": doc["name"],
        "source_key": doc.get("source_key"),
        "mime": doc.get("mime"),
        "size": int(doc.get("size") or 0),
        "size_human": human_size(int(doc.get("size") or 0)),
        "sha256": doc.get("sha256"),
        "status": doc.get("status"),
        "error": doc.get("error"),
        "page_count": int(doc.get("page_count") or 0),
        "chunk_count": int(doc.get("chunk_count") or 0),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "metadata": {
            "title": metadata.get("title"),
            "source_type": metadata.get("source_type"),
            "parser": metadata.get("parser"),
            "warnings": metadata.get("warnings") or [],
            "chunk_types": metadata.get("chunk_types") or [],
        },
    }


@router.get("")
async def list_documents(request: Request, include_error: bool = Query(default=True)):
    ctx: AppContext = get_ctx(request)
    docs = ctx.db.list_documents()
    if not include_error:
        docs = [d for d in docs if d.get("status") != "error"]
    return {
        "documents": [_public_document(d) for d in docs],
        "index_version": ctx.db.get_index_version(),
        "stats": ctx.db.stats(),
    }


@router.post("/upload")
async def upload_documents(request: Request, files: list[UploadFile] = File(...)):
    ctx: AppContext = get_ctx(request)
    settings = ctx.settings
    results: list[dict] = []
    errors: list[dict] = []
    for upload in files:
        filename = upload.filename or "document"
        suffix = Path(filename).suffix.lower()
        if suffix not in settings.allowed_suffixes:
            errors.append({"name": filename, "error": f"不支持的文件类型: {suffix or 'unknown'}"})
            continue
        data = await upload.read()
        if not data:
            errors.append({"name": filename, "error": "空文件"})
            continue
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            errors.append(
                {"name": filename, "error": f"文件超过 {settings.max_upload_mb} MB 限制"}
            )
            continue
        try:
            result = await run_in_threadpool(ctx.index.ingest_bytes, data, filename)
            results.append(result.to_dict())
        except Exception as exc:
            errors.append({"name": filename, "error": str(exc)})
    return {
        "results": results,
        "errors": errors,
        "index_version": ctx.db.get_index_version(),
        "stats": ctx.db.stats(),
    }


@router.get("/{doc_id}")
async def document_detail(request: Request, doc_id: str):
    ctx: AppContext = get_ctx(request)
    doc = ctx.db.get_document(doc_id)
    if not doc:
        raise NotFoundError("文档不存在")
    chunks = ctx.db.get_doc_chunks(doc_id)
    data = _public_document(doc)
    data["chunks_preview"] = [
        {
            "id": c.id,
            "ordinal": c.ordinal,
            "heading": c.heading,
            "page_start": c.page_start,
            "content_type": c.content_type,
            "text": c.text[:500],
        }
        for c in chunks[:20]
    ]
    return data


@router.get("/{doc_id}/chunks")
async def document_chunks(
    request: Request, doc_id: str, limit: int = Query(default=100, ge=1, le=500)
):
    ctx: AppContext = get_ctx(request)
    if not ctx.db.get_document(doc_id):
        raise NotFoundError("文档不存在")
    chunks = ctx.db.get_doc_chunks(doc_id)
    return {
        "total": len(chunks),
        "chunks": [c.to_dict() for c in chunks[:limit]],
    }


@router.post("/{doc_id}/reindex")
async def reindex_document(request: Request, doc_id: str):
    ctx: AppContext = get_ctx(request)
    result = await run_in_threadpool(ctx.index.reindex_document, doc_id)
    if result.status == "failed":
        raise AppError(result.message, code="reindex_failed", status_code=400)
    return result.to_dict()


@router.delete("/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    ctx: AppContext = get_ctx(request)
    ok_deleted = await run_in_threadpool(ctx.index.delete_document, doc_id)
    if not ok_deleted:
        raise NotFoundError("文档不存在")
    return {"deleted": True, "doc_id": doc_id, "index_version": ctx.db.get_index_version()}
