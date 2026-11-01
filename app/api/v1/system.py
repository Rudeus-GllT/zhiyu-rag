from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.api.deps import get_ctx, require_dashboard
from app.agent.tools import TOOL_REGISTRY
from app.context import AppContext

router = APIRouter(prefix="/api/v1/system", tags=["system"], dependencies=[Depends(require_dashboard)])


@router.get("/info")
async def system_info(request: Request):
    ctx: AppContext = get_ctx(request)
    settings = ctx.settings
    stats = ctx.db.stats()
    llm_health = ctx.llm.health()
    return {
        "app": {
            "name": settings.app_name,
            "subtitle": settings.app_subtitle,
            "version": __version__,
            "environment": settings.app_env,
            "uptime_seconds": int(time.time() - getattr(request.app.state, "started_at", time.time())),
        },
        "llm": llm_health,
        "index": ctx.index.stats(),
        "stats": stats,
        "cache": ctx.cache.stats(),
        "retrieval": {
            "dense_top_k": settings.dense_top_k,
            "bm25_top_k": settings.bm25_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "context_token_budget": settings.context_token_budget,
            "min_confidence": settings.min_confidence,
            "embedding": ctx.llm.embedder.name,
            "vector_backend": ctx.index.vectors.name,
        },
        "features": {
            "multimodal": True,
            "ocr": settings.ocr_enabled,
            "vlm": llm_health.get("vision", False),
            "incremental_index": True,
            "semantic_cache": settings.semantic_cache_enabled,
            "evaluation": True,
            "streaming": True,
        },
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "allowed_extensions": sorted(settings.allowed_suffixes),
        },
    }


@router.get("/health")
async def health(request: Request):
    ctx: AppContext = get_ctx(request)
    return {
        "status": "ok",
        "ready": ctx.index.ready,
        "index_version": ctx.db.get_index_version(),
        "documents": ctx.db.doc_count(),
        "chunks": ctx.db.chunk_count(),
        "llm": ctx.llm.llm.name,
        "embedding": ctx.llm.embedder.name,
        "vector_backend": ctx.index.vectors.name,
    }


@router.post("/cache/clear")
async def clear_cache(request: Request):
    ctx: AppContext = get_ctx(request)
    ctx.cache.clear()
    return {"cleared": True, "stats": ctx.cache.stats()}


@router.post("/index/repair")
async def repair_index(request: Request):
    ctx: AppContext = get_ctx(request)
    count = await run_in_threadpool(ctx.index.repair_vectors)
    return {"repaired_vectors": count, "index": ctx.index.stats()}


@router.get("/tools")
async def list_tools(request: Request):
    get_ctx(request)
    return {"tools": TOOL_REGISTRY}
