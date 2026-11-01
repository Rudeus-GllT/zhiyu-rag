from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.api.errors import install_error_handlers
from app.api.v1.router import api_router
from app.config import settings
from app.context import build_context
from app.core.logging_setup import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    started = time.time()
    logger.info("启动 %s v%s ...", settings.app_name, __version__)
    ctx = await run_in_threadpool(build_context, settings)
    app.state.ctx = ctx
    app.state.started_at = started
    if settings.auto_seed:
        try:
            results = await run_in_threadpool(ctx.index.seed_samples)
            changed = [r for r in results if r.status in {"created", "updated"}]
            if changed:
                logger.info(
                    "示例知识库初始化完成: %d 篇新增/更新, 索引 v%d",
                    len(changed),
                    ctx.db.get_index_version(),
                )
        except Exception as exc:
            logger.warning("示例知识库初始化失败: %s", exc)
    logger.info(
        "就绪: http://%s:%d  (LLM=%s, embedding=%s, vectors=%s)",
        settings.host,
        settings.port,
        ctx.llm.llm.name,
        ctx.llm.embedder.name,
        ctx.index.vectors.name,
    )
    yield
    logger.info("服务停止")


def create_app() -> FastAPI:
    app = FastAPI(
        title=f"{settings.app_name} API",
        description=(
            "面向个人的多模态 RAG 知识库智能体：PDF / Markdown / 图片解析、"
            "BM25 + 向量混合检索、重排与引用核查、增量索引、语义缓存、"
            "RAGAS 风格评测。"
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=settings.cors_origin_list != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or "req_" + uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        if request.url.path.startswith("/static/"):
            # Local-first app: always revalidate UI assets so a restart is enough
            # to pick up theme/CSS changes instead of relying on browser cache.
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    install_error_handlers(app)
    app.include_router(api_router)

    web_dir = settings.web_dir
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(
                web_dir / "index.html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @app.get("/favicon.svg", include_in_schema=False)
        async def favicon():
            path = web_dir / "favicon.svg"
            if path.exists():
                return FileResponse(path, media_type="image/svg+xml")
            return JSONResponse({"detail": "not found"}, status_code=404)
    else:
        @app.get("/", include_in_schema=False)
        async def index_missing():
            return JSONResponse(
                {
                    "app": settings.app_name,
                    "message": "Web console assets are missing; use /docs for the API.",
                    "docs": "/docs",
                }
            )

    @app.get("/health", include_in_schema=False)
    async def health(request: Request):
        ctx = getattr(request.app.state, "ctx", None)
        if not ctx:
            return JSONResponse({"status": "starting"}, status_code=503)
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": __version__,
            "documents": ctx.db.doc_count(),
            "chunks": ctx.db.chunk_count(),
            "index_version": ctx.db.get_index_version(),
        }

    return app


app = create_app()
