"""BM25 sparse index over jieba tokens with a versioned pickle cache."""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from app.core.types import Chunk
from app.core.utils import tokenize

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore


class _TinyBM25:
    """Minimal Okapi BM25 fallback (k1=1.5, b=0.75)."""

    def __init__(self, corpus: Sequence[Sequence[str]]):
        self.corpus = [list(doc) for doc in corpus]
        self.n = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(1, self.n)
        self.df: dict[str, int] = {}
        for doc in self.corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        k1, b = 1.5, 0.75
        for i, doc in enumerate(self.corpus):
            dl = len(doc)
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token in query:
                if token not in counts:
                    continue
                df = self.df.get(token, 0)
                idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5))
                tf = counts[token]
                score += idf * tf * (k1 + 1) / (
                    tf + k1 * (1 - b + b * dl / max(1e-6, self.avgdl))
                )
            scores[i] = score
        return scores


class BM25Index:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._bm25 = None
        self._version = -1

    @property
    def version(self) -> int:
        return self._version

    def build(self, chunks: Sequence[Chunk], version: int, *, persist: bool = True) -> None:
        ids = [c.id for c in chunks]
        if not ids:
            with self._lock:
                self._ids = []
                self._bm25 = None
                self._version = version
            return
        corpus = [
            list(tokenize(f"{c.heading}\n{c.text}", with_bigrams=True)) for c in chunks
        ]
        bm25 = BM25Okapi(corpus) if BM25Okapi is not None else _TinyBM25(corpus)
        with self._lock:
            self._ids = ids
            self._bm25 = bm25
            self._version = version
        if persist:
            try:
                cache_path = self.index_dir / f"bm25_{version}.pkl"
                with cache_path.open("wb") as fh:
                    pickle.dump({"version": version, "ids": ids, "corpus": corpus}, fh)
                # Keep only the last three caches.
                for old in sorted(self.index_dir.glob("bm25_*.pkl"))[:-3]:
                    old.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("BM25 缓存写入失败: %s", exc)

    def load(self, version: int, expected_count: int) -> bool:
        cache_path = self.index_dir / f"bm25_{version}.pkl"
        if not cache_path.exists():
            return False
        try:
            with cache_path.open("rb") as fh:
                data = pickle.load(fh)
            if data.get("version") != version:
                return False
            ids = data.get("ids") or []
            corpus = data.get("corpus") or []
            if len(ids) != expected_count or len(ids) != len(corpus):
                return False
            bm25 = BM25Okapi(corpus) if BM25Okapi is not None else _TinyBM25(corpus)
            with self._lock:
                self._ids = list(ids)
                self._bm25 = bm25
                self._version = version
            return True
        except Exception as exc:
            logger.debug("BM25 缓存读取失败: %s", exc)
            return False

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query, with_bigrams=True)
        with self._lock:
            if not tokens or self._bm25 is None or not self._ids:
                return []
            scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
            if scores.size != len(self._ids):
                return []
            indices = np.argsort(-scores)
            selected = [int(i) for i in indices[: max(1, top_k)] if scores[int(i)] > 0]
            if not selected:
                return []
            top_score = float(scores[selected[0]])
            denom = top_score if top_score > 0 else 1.0
            return [(self._ids[i], float(scores[i]) / denom) for i in selected]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def reset(self) -> None:
        with self._lock:
            self._ids = []
            self._bm25 = None
            self._version = -1

    def health(self) -> dict:
        return {"backend": "bm25", "count": self.count, "version": self._version}
