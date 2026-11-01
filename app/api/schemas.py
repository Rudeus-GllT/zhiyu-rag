from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None
    debug: bool = False

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message 不能为空")
        return value


class FeedbackRequest(BaseModel):
    message_id: str
    session_id: str | None = None
    rating: Literal[-1, 1]
    comment: str = Field(default="", max_length=1000)


class EvalRunRequest(BaseModel):
    dataset: str
    limit: int | None = Field(default=None, ge=1, le=200)
