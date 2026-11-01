from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_ctx, require_dashboard
from app.api.schemas import ChatRequest, FeedbackRequest
from app.context import AppContext
from app.core.errors import NotFoundError
from app.core.utils import utc_now_iso

router = APIRouter(prefix="/api/v1", tags=["chat"], dependencies=[Depends(require_dashboard)])

SUGGESTED_QUESTIONS = [
    {"text": "混合检索是怎么做融合的？", "category": "检索"},
    {"text": "总结一下多模态文档解析的流程", "category": "总结"},
    {"text": "Chroma 和 BM25 各适合什么场景？", "category": "对比"},
    {"text": "增量索引如何判断文档需要更新？", "category": "索引"},
    {"text": "知识库里有哪些文档？", "category": "工具"},
]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    ctx: AppContext = get_ctx(request)
    answer = await run_in_threadpool(
        ctx.agent.ask,
        payload.message,
        session_id=payload.session_id,
        top_k=payload.top_k,
        doc_ids=payload.doc_ids,
        debug=payload.debug,
    )
    return answer.to_dict(include_trace=True)


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    ctx: AppContext = get_ctx(request)

    def generator():
        try:
            for event in ctx.agent.stream(
                payload.message,
                session_id=payload.session_id,
                top_k=payload.top_k,
                doc_ids=payload.doc_ids,
            ):
                yield _sse(event["event"], event["data"])
        except Exception as exc:  # pragma: no cover - defensive
            yield _sse("error", {"message": f"流式服务异常: {exc}"})

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/suggestions")
async def suggestions(request: Request):
    get_ctx(request)
    return {"suggestions": SUGGESTED_QUESTIONS}


@router.get("/sessions")
async def list_sessions(
    request: Request, client: str = Query(default="web"), limit: int = Query(default=40, ge=1, le=100)
):
    ctx: AppContext = get_ctx(request)
    sessions = ctx.db.list_sessions(client=client, limit=limit)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
async def session_messages(request: Request, session_id: str, limit: int = Query(default=100, ge=1, le=300)):
    ctx: AppContext = get_ctx(request)
    session = ctx.db.get_session(session_id)
    if not session:
        raise NotFoundError("会话不存在")
    messages = ctx.db.get_messages(session_id, limit=limit)
    return {"session": session, "messages": messages}


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    ctx: AppContext = get_ctx(request)
    if not ctx.db.get_session(session_id):
        raise NotFoundError("会话不存在")
    ctx.db.delete_session(session_id)
    return {"deleted": True, "session_id": session_id}


@router.post("/messages/{message_id}/feedback")
async def add_feedback(request: Request, message_id: str, payload: FeedbackRequest):
    ctx: AppContext = get_ctx(request)
    message = ctx.db.get_message(message_id)
    if not message:
        raise NotFoundError("消息不存在")
    session_id = payload.session_id or message.get("session_id") or ""
    ctx.db.add_feedback(
        "fb_" + uuid.uuid4().hex[:14],
        message_id,
        session_id,
        payload.rating,
        payload.comment,
    )
    return {"ok": True, "created_at": utc_now_iso()}
