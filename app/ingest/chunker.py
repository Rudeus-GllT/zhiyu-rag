"""Structure-aware chunking with LangChain's recursive splitter (fallback kept)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.config import Settings
from app.core.types import Block, Chunk, ParsedDocument
from app.core.utils import (
    clean_extracted_text,
    estimate_tokens,
    normalize_text,
    stable_chunk_id,
    truncate,
)

try:  # declared stack dependency; the custom splitter is only a safety net
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None  # type: ignore

_TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
_CODE_SEPARATORS = ["\n\n", "\n", "    ", "  ", " "]
_TABLE_SEPARATORS = ["\n", " | ", " "]


@dataclass
class _Section:
    heading: list[str]
    content_type: str
    parts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(part for part in self.parts if part).strip()

    @property
    def size(self) -> int:
        return len(self.text)


def _fallback_split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind("。"), window.rfind("；"))
            if cut > size * 0.55:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def _split(
    text: str, *, content_type: str, chunk_size: int, overlap: int
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    separators = (
        _CODE_SEPARATORS
        if content_type == "code"
        else _TABLE_SEPARATORS
        if content_type == "table"
        else _TEXT_SEPARATORS
    )
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(120, chunk_size),
            chunk_overlap=max(0, min(overlap, chunk_size // 2)),
            separators=separators,
            length_function=len,
            keep_separator=False,
        )
        pieces = splitter.split_text(text)
    else:
        pieces = _fallback_split(text, chunk_size, overlap)
    return [p.strip() for p in pieces if p and p.strip()]


def _split_block(block: Block, size: int, overlap: int) -> list[str]:
    return _split(
        block.text,
        content_type=block.content_type,
        chunk_size=size,
        overlap=overlap,
    )


def chunk_document(
    document: ParsedDocument,
    *,
    doc_id: str,
    doc_name: str,
    settings: Settings,
) -> list[Chunk]:
    chunk_size = max(200, settings.chunk_size)
    overlap = max(0, min(settings.chunk_overlap, chunk_size // 2))
    max_chars = max(chunk_size, settings.max_chunk_chars)
    min_chars = max(10, settings.min_chunk_chars)

    sections: list[_Section] = []
    current = _Section(heading=[], content_type="text")

    def flush() -> None:
        nonlocal current
        if current.parts and current.text:
            sections.append(current)
        current = _Section(heading=[], content_type="text")

    for block in document.blocks:
        text = clean_extracted_text(block.text)
        if not text:
            continue
        content_type = block.content_type or "text"
        heading = [truncate(h, 80) for h in block.heading_path][-4:]
        special = content_type in {"code", "table", "image", "vlm", "ocr"}
        if current.parts and (special or current.content_type != content_type or current.heading != heading):
            flush()
        if not current.parts:
            current = _Section(heading=heading, content_type=content_type)
        current.parts.append(text)
        if block.page is not None:
            current.pages.append(block.page)
        if block.metadata:
            for key in ("provider", "caption", "language"):
                if key in block.metadata and key not in current.metadata:
                    current.metadata[key] = block.metadata[key]
        if current.size >= max_chars * 1.35 or special:
            flush()
    flush()

    chunks: list[Chunk] = []
    ordinal = 0
    for section_index, section in enumerate(sections):
        body = section.text
        heading_line = " / ".join(section.heading).strip()

        pieces = _split(
            body,
            content_type=section.content_type,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not pieces:
            continue
        page_start = min(section.pages) if section.pages else None
        page_end = max(section.pages) if section.pages else None

        for piece in pieces:
            text = clean_extracted_text(piece)
            if not text:
                continue
            if (
                len(text) < min_chars
                and section.content_type == "text"
                and len(pieces) > 1
            ):
                continue
            if len(text) > max_chars * 1.6:
                text = truncate(text, int(max_chars * 1.6))
            metadata = {
                "section_index": section_index,
                "heading_path": section.heading,
                **section.metadata,
            }
            chunk = Chunk(
                id=stable_chunk_id(doc_id, ordinal, text),
                doc_id=doc_id,
                doc_name=doc_name,
                ordinal=ordinal,
                text=text,
                page_start=page_start,
                page_end=page_end,
                heading_path=section.heading,
                content_type=section.content_type,
                token_estimate=estimate_tokens(text),
                metadata=metadata,
            )
            chunks.append(chunk)
            ordinal += 1

    return chunks
