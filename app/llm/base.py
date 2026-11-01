from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ChatResult:
    text: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0


class BaseLLM(ABC):
    name: str = "base"
    model: str = ""
    supports_vision: bool = False
    is_mock: bool = False

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError
