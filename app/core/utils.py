"""Small, dependency-light helpers used across the pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- time


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def utc_now_ts() -> float:
    return utc_now().timestamp()


# ------------------------------------------------------------------------- hashes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    raw = f"{doc_id}|{ordinal}|{sha256_text(text)[:16]}".encode("utf-8")
    return "c_" + hashlib.sha1(raw).hexdigest()[:22]


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:n]


# --------------------------------------------------------------------------- text

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_MULTI_BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


_PAGE_NOISE = re.compile(
    r"^\s*(?:[-—–]\s*)?(?:第\s*)?\d{1,4}\s*(?:页|/\s*\d{1,4})?(?:\s*[-—–])?\s*$"
)


def clean_extracted_text(text: str) -> str:
    """Remove typical PDF / OCR boilerplate while keeping structure."""
    text = normalize_text(text)
    if not text:
        return ""
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _PAGE_NOISE.match(stripped):
            continue
        kept.append(line.rstrip())
    return normalize_text("\n".join(kept))


_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= 2]


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK.findall(text)) / max(1, len(text))


def estimate_tokens(text: str) -> int:
    """Rough token estimate that works for mixed Chinese/English text."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    other = max(0, len(text) - cjk - sum(len(w) for w in re.findall(r"[A-Za-z0-9_]+", text)))
    return int(cjk * 1.0 + latin * 1.3 + other * 0.25) + 1


def human_size(num: int | float) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


# --------------------------------------------------------------------- tokenizing

_STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "它", "他", "她", "我们", "你们", "他们", "以及", "或者",
    "但是", "因为", "所以", "如果", "可以", "如何", "什么", "怎么", "哪些", "哪个",
    "请问", "一下", "关于", "进行", "通过", "对于", "这个", "那个", "这些", "那些",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and",
    "or", "in", "on", "for", "with", "by", "this", "that", "these", "those", "it",
    "as", "at", "from", "how", "what", "which", "who", "when", "where", "why",
}

_EN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_+\-\.]*")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def _get_jieba():
    import logging

    import jieba

    jieba.setLogLevel(logging.WARNING)
    return jieba


@lru_cache(maxsize=8192)
def tokenize_cached(text: str, with_bigrams: bool = False) -> tuple[str, ...]:
    text = normalize_text(text).lower()
    if not text:
        return ()
    tokens: list[str] = []
    for word in _get_jieba().cut(text, cut_all=False):
        word = word.strip()
        if not word:
            continue
        if word in _STOPWORDS or len(word) == 1 and not _CJK_CHAR.match(word):
            continue
        if len(word) == 1 and word in _STOPWORDS:
            continue
        tokens.append(word)
    tokens.extend(w.lower() for w in _EN_WORD.findall(text))
    if with_bigrams:
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token and token not in seen and len(token) <= 24:
            seen.add(token)
            out.append(token)
    return tuple(out)


def tokenize(text: str, with_bigrams: bool = False) -> list[str]:
    return list(tokenize_cached(text, with_bigrams))


def keyword_overlap(query_tokens: Iterable[str], text: str) -> float:
    q = {t for t in query_tokens if t}
    if not q:
        return 0.0
    doc = set(tokenize(text, with_bigrams=True))
    return len(q & doc) / len(q)


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    for token in tokenize(text, with_bigrams=False):
        if len(token) < 2 and not _EN_WORD.match(token):
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [word for word, _ in ranked[:limit]]


# ------------------------------------------------------------------------- vectors


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ------------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Sequence[float] | None = None
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for idx, ranking in enumerate(rankings):
        weight = 1.0 if not weights or idx >= len(weights) else float(weights[idx])
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return scores


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if math.isclose(hi, lo):
        return {key: 1.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


# --------------------------------------------------------------------------- json


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_loads(value: str | bytes | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(cleaned)):
        char = cleaned[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : idx + 1]
                try:
                    value = json.loads(candidate)
                    return value if isinstance(value, dict) else None
                except Exception:
                    return None
    return None


# --------------------------------------------------------------------------- misc


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


_SAFE_FILENAME = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._\- ()\[\]]+")


def safe_filename(name: str, fallback: str = "document") -> str:
    name = Path(name or "").name
    name = _SAFE_FILENAME.sub("_", name).strip(" ._")
    if not name:
        name = fallback
    return name[:120]


def guess_mime(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(suffix.lower(), "application/octet-stream")


def iso_duration(started: float, ended: float | None = None) -> int:
    end = utc_now_ts() if ended is None else ended
    return max(0, int(round((end - started) * 1000)))
