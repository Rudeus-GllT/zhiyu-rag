"""Minimal OpenAI-compatible chat client (DeepSeek, Qwen, OpenAI, vLLM...)."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import httpx

from app.core.errors import LLMError
from app.llm.base import BaseLLM, ChatResult


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        vision_model: str = "",
        timeout: float = 60.0,
        default_max_tokens: int = 1200,
        default_temperature: float = 0.15,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model.strip()
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.name = "openai-compatible"
        self.supports_vision = bool(self.vision_model)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        stream: bool,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> ChatResult:
        if not self.enabled:
            raise LLMError("未配置可用的模型 API Key")
        started = time.perf_counter()
        payload = self._payload(
            messages, temperature, max_tokens, json_mode, stream=False, model=model
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    detail = response.text[:300]
                    raise LLMError(f"模型服务返回 {response.status_code}: {detail}")
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError("模型返回为空")
                text = (choices[0].get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                return ChatResult(
                    text=text.strip(),
                    model=data.get("model") or self.model,
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except LLMError:
                raise
            except Exception as exc:  # network / JSON errors
                last_error = exc
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
                    continue
        raise LLMError(f"模型调用失败: {last_error}")

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        if not self.enabled:
            raise LLMError("未配置可用的模型 API Key")
        payload = self._payload(messages, temperature, max_tokens, False, stream=True)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")[:300]
                        raise LLMError(f"模型服务返回 {response.status_code}: {body}")
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            yield content
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"流式模型调用失败: {exc}") from exc
