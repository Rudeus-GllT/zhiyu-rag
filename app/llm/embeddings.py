"""Embedding providers.

``HashEmbedder`` is a deterministic, dependency-free lexical embedder built
from jieba words, CJK bigrams/trigrams and hashed random projections.  It is
not a replacement for a trained model, but it makes the whole retrieval stack
usable offline and is a solid baseline for a personal Chinese knowledge base.
Point ``EMBEDDING_PROVIDER=openai`` at any OpenAI-compatible embedding API
(OpenAI, SiliconFlow/BGE, Qwen, ...) for production quality.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Iterable, Sequence

import httpx
import numpy as np

from app.core.utils import l2_normalize, normalize_text, sha256_text, tokenize

logger = logging.getLogger(__name__)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")


class BaseEmbedder:
    name = "base"
    dim = 0

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 1024):
        self.dim = int(dim)
        self.name = f"hash-{self.dim}"

    def _features(self, text: str) -> Counter[str]:
        text = normalize_text(text).lower()
        features: Counter[str] = Counter()
        for token in tokenize(text, with_bigrams=True):
            features[token] += 3 if len(token) >= 2 else 1
        # Character trigrams catch names / terms that jieba splits incorrectly.
        for run in _CJK_RUN.findall(text):
            for i in range(len(run) - 2):
                features[run[i : i + 3]] += 1
        return features

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        features = self._features(text)
        if not features:
            return vector
        for token, count in features.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = (1.0 + 0.12 * min(len(token), 8)) * (1.0 + math.log1p(count))
            vector[index] += sign * weight
        return l2_normalize(vector)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(text) for text in texts])


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int = 32,
        timeout: float = 60.0,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = int(dim)
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self.name = f"openai:{model}"

    def _batch(self, texts: Sequence[str]) -> np.ndarray:
        payload = {"model": self.model, "input": list(texts)}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"{response.status_code}: {response.text[:240]}")
                data = response.json().get("data") or []
                data.sort(key=lambda item: item.get("index", 0))
                vectors = [item.get("embedding") or [] for item in data]
                if len(vectors) != len(texts):
                    raise RuntimeError("embedding 返回数量不匹配")
                return l2_normalize(np.asarray(vectors, dtype=np.float32))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"embedding 调用失败: {last_error}")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            outputs.append(self._batch(texts[start : start + self.batch_size]))
        matrix = np.vstack(outputs)
        if matrix.shape[1] != self.dim:
            # Trust the API's actual dimension instead of failing hard.
            self.dim = int(matrix.shape[1])
        return matrix


class CachedEmbedder(BaseEmbedder):
    """Persistent embedding cache keyed by (model signature, text hash)."""

    def __init__(self, inner: BaseEmbedder, db):
        self.inner = inner
        self.db = db
        self.name = inner.name
        self.dim = inner.dim
        self.last_hit_count = 0

    @property
    def signature(self) -> str:
        return self.inner.signature

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys: list[str] = []
        for text in texts:
            keys.append(sha256_text(f"{self.signature}|{text}"))
        cached = self.db.get_embeddings(keys)
        missing_indices = [i for i, key in enumerate(keys) if key not in cached]
        self.last_hit_count = len(texts) - len(missing_indices)
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            vectors = self.inner.embed(missing_texts)
            if vectors.shape[1] != self.dim:
                self.dim = int(vectors.shape[1])
            entries = [(keys[i], self.signature, vectors[pos]) for pos, i in enumerate(missing_indices)]
            self.db.put_embeddings(entries)
            for pos, i in enumerate(missing_indices):
                cached[keys[i]] = vectors[pos]
        matrix = np.vstack([cached[key] for key in keys])
        return matrix.astype(np.float32)


def create_embedder(settings, db=None) -> BaseEmbedder:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"openai", "api"}:
        if settings.embed_key:
            inner: BaseEmbedder = OpenAIEmbedder(
                api_key=settings.embed_key,
                base_url=settings.embed_base_url,
                model=settings.embedding_model,
                dim=settings.embedding_dim,
                batch_size=settings.embedding_batch_size,
            )
        else:
            logger.warning("EMBEDDING_PROVIDER=openai 但缺少 API Key，回退到本地 HashEmbedder")
            inner = HashEmbedder(settings.embedding_dim)
    elif provider == "hash":
        inner = HashEmbedder(settings.embedding_dim)
    else:  # auto: a safe, fully local default
        inner = HashEmbedder(settings.embedding_dim)
    return CachedEmbedder(inner, db) if db is not None else inner
