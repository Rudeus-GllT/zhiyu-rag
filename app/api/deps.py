from __future__ import annotations

import hmac

from fastapi import Header, Request

from app.context import AppContext
from app.core.errors import AppError


def get_ctx(request: Request) -> AppContext:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise AppError("服务尚未初始化", code="not_ready", status_code=503)
    return ctx


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def require_dashboard(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Optional bearer auth for the console API.

    Disabled by default for local use; set DASHBOARD_TOKEN to enforce it.
    """
    token = request.app.state.ctx.settings.dashboard_token.strip()
    if not token:
        return
    supplied = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, token):
        raise AppError("未授权访问", code="unauthorized", status_code=401)
