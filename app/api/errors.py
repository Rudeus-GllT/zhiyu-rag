from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.errors import AppError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or "req_" + uuid.uuid4().hex[:12]


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "code": exc.code,
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        message = "请求参数不合法"
        if errors:
            first = errors[0]
            location = ".".join(str(x) for x in first.get("loc", []) if x != "body")
            message = f"{location}: {first.get('msg', message)}" if location else str(first.get("msg", message))
        return JSONResponse(
            status_code=422,
            content={"detail": message, "code": "validation_error", "request_id": _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("未处理异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务内部错误",
                "code": "internal_error",
                "request_id": _request_id(request),
            },
        )
