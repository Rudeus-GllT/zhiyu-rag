"""Two-layer response cache.

L1: in-process LRU used for embeddings inside the embedder itself.
L2: SQLite exact-match cache (normalized question + params + index version).
L3: semantic cache (cosine similarity against recent questions of the same
    index version), which catches paraphrases such as “混合检索怎么融合”
    vs “混合检索的融合方式”.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.config import Settings
from app.core.utils import cosine_similarity, sha256_text
from app.llm.embeddings import BaseEmbedder
from app.storage.db import Database


def normalize_question(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"[\s，。！？、；：,.!?;:（）()\[\]【】\"'“”‘’]+", "", text)
    return text[:300]


@dataclass
class CacheHit:
    value: dict[str, Any]
    kind: str
    score: float = 1.0


class ResponseCache:
    def __init__(self, db: Database, embedder: BaseEmbedder, settings: Settings):
        self.db = db
        self.embedder = embedder
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.cache_enabled

    def _key(self, question: str, version: int, top_k: int, doc_ids: Sequence[str]) -> str:
        norm = normalize_question(question)
        doc_part = ",".join(sorted(doc_ids))
        return "qa:" + sha256_text(f"v{version}|{norm}|{top_k}|{doc_part}")

    def get(
        self, question: str, version: int, top_k: int, doc_ids: Sequence[str] = ()
    ) -> CacheHit | None:
        if not self.enabled:
            return None
        key = self._key(question, version, top_k, doc_ids)
        value = self.db.cache_get(key)
        if value:
            return CacheHit(value=value, kind="exact", score=1.0)
        if not self.settings.semantic_cache_enabled:
            return None
        norm = normalize_question(question)
        if len(norm) < 5:
            return None
        try:
            vector = self.embedder.embed([question])[0]
        except Exception:
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for row in self.db.semantic_candidates(
            version, self.settings.semantic_cache_max_rows
        ):
            if row["query_norm"] == norm:
                continue
            score = cosine_similarity(vector, row["vector"])
            if score >= self.settings.semantic_cache_threshold:
                if best is None or score > best[0]:
                    best = (score, row)
        if not best:
            return None
        row = best[1]
        self.db.semantic_hit(row["id"])
        return CacheHit(value=row["value"], kind="semantic", score=best[0])

    def set(
        self,
        question: str,
        version: int,
        top_k: int,
        doc_ids: Sequence[str],
        value: dict[str, Any],
        *,
        confidence: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        key = self._key(question, version, top_k, doc_ids)
        self.db.cache_set(
            key, "qa", value, ttl=max(60, self.settings.cache_ttl_seconds)
        )
        if (
            self.settings.semantic_cache_enabled
            and confidence >= 0.4
            and len(normalize_question(question)) >= 5
        ):
            try:
                vector = self.embedder.embed([question])[0]
                self.db.semantic_set(
                    entry_id="sc_" + sha256_text(key)[:20],
                    query=question,
                    query_norm=normalize_question(question),
                    embedding=np.asarray(vector, dtype=np.float32),
                    value=value,
                    version=version,
                )
                self.db.prune_semantic(version, self.settings.semantic_cache_max_rows)
            except Exception:
                pass

    def clear(self) -> None:
        self.db.cache_clear()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "semantic_enabled": self.settings.semantic_cache_enabled,
            "threshold": self.settings.semantic_cache_threshold,
            **self.db.cache_stats(),
        }
