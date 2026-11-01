from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, documents, eval, system

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(eval.router)
api_router.include_router(system.router)
