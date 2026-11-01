"""Deterministic offline model.

The orchestration layer uses task-specific heuristics when ``is_mock`` is
true, so this class is only a last-resort implementation of the transport
interface (and keeps unit tests hermetic).
"""

from __future__ import annotations

import time
from typing import Any, Iterator

from app.llm.base import BaseLLM, ChatResult


class MockLLM(BaseLLM):
    name = "local-extractive"
    model = "zhiyu-local"
    supports_vision = False
    is_mock = True

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> ChatResult:
        user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                user = content if isinstance(content, str) else ""
                break
        if json_mode:
            text = "{}"
        else:
            text = (
                "（离线模式）我已收到问题："
                + user[:160]
                + "\n\n当前未配置在线大模型，回答由知识库原文摘录生成，可在 .env 中配置 "
                "LLM_API_KEY 与 LLM_BASE_URL 获得完整生成能力。"
            )
        return ChatResult(text=text, model=self.model, usage={"total_tokens": 0}, latency_ms=0)

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        text = self.chat(messages).text
        for idx in range(0, len(text), 18):
            yield text[idx : idx + 18]
            time.sleep(0.006)
