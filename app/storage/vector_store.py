"""Vector backends.

Chroma is the primary backend (persistent, cosine space).  A small SQLite +
NumPy backend is the automatic fallback so the app still boots on machines
where Chroma's native wheels are unavailable.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import Settings
from app.core.types import Chunk
from app.core.utils import l2_normalize

logger = logging.getLogger(__name__)


@dataclass
class VectorHit:
    chunk_id: str
    score: float
    distance: float = 0.0


class BaseVectorStore:
    name = "base"

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray, version: int = 0) -> None:
        raise NotImplementedError

    def delete(self, *, doc_id: str | None = None, chunk_ids: Sequence[str] | None = None) -> None:
        raise NotImplementedError

    def search(
        self, query_embedding: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[VectorHit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def health(self) -> dict:
        return {"backend": self.name, "count": self.count()}


class ChromaVectorStore(BaseVectorStore):
    name = "chroma"

    def __init__(self, settings: Settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.chroma_dir
        self.collection_name = settings.chroma_collection
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray, version: int = 0) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "ordinal": int(c.ordinal),
                "page_start": int(c.page_start or 0),
                "page_end": int(c.page_end or 0),
                "content_type": c.content_type,
                "heading": c.heading[:300],
                "index_version": int(version),
            }
            for c in chunks
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            metadatas=metadatas,
            documents=[c.text[:4000] for c in chunks],
        )

    def delete(self, *, doc_id: str | None = None, chunk_ids: Sequence[str] | None = None) -> None:
        if doc_id:
            self.collection.delete(where={"doc_id": doc_id})
        if chunk_ids:
            self.collection.delete(ids=list(chunk_ids))

    def search(
        self, query_embedding: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[VectorHit]:
        total = self.count()
        if total <= 0 or top_k <= 0:
            return []
        where = None
        if doc_ids:
            where = {"doc_id": {"$in": list(doc_ids)}}
        result = self.collection.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
            n_results=min(top_k, total),
            where=where,
            include=["distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[VectorHit] = []
        for chunk_id, distance in zip(ids, distances):
            dist = float(distance)
            hits.append(VectorHit(chunk_id=chunk_id, score=max(0.0, 1.0 - dist), distance=dist))
        return hits

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def health(self) -> dict:
        return {"backend": self.name, "count": self.count(), "path": str(self.path)}


class LocalVectorStore(BaseVectorStore):
    name = "local"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (chunk_id TEXT PRIMARY KEY, doc_id TEXT, dim INTEGER, vector BLOB)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_doc ON vectors(doc_id)")
        self._conn.commit()
        self._ids: list[str] = []
        self._doc_ids: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        rows = self._conn.execute("SELECT chunk_id, doc_id, dim, vector FROM vectors").fetchall()
        self._ids = [row[0] for row in rows]
        self._doc_ids = [row[1] for row in rows]
        if rows:
            dim = int(rows[0][2])
            matrix = np.zeros((len(rows), dim), dtype=np.float32)
            for idx, row in enumerate(rows):
                vec = np.frombuffer(row[3], dtype=np.float32)
                if vec.size == dim:
                    matrix[idx] = vec
            self._matrix = l2_normalize(matrix)
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray, version: int = 0) -> None:
        if not chunks:
            return
        embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors(chunk_id, doc_id, dim, vector) VALUES(?,?,?,?)",
                [
                    (c.id, c.doc_id, int(embeddings[i].shape[0]), embeddings[i].tobytes())
                    for i, c in enumerate(chunks)
                ],
            )
            self._conn.commit()
            self._load()

    def delete(self, *, doc_id: str | None = None, chunk_ids: Sequence[str] | None = None) -> None:
        with self._lock:
            if doc_id:
                self._conn.execute("DELETE FROM vectors WHERE doc_id=?", (doc_id,))
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                self._conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({placeholders})", list(chunk_ids))
            self._conn.commit()
            self._load()

    def search(
        self, query_embedding: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[VectorHit]:
        with self._lock:
            if not self._ids or self._matrix.size == 0:
                return []
            query = l2_normalize(np.asarray(query_embedding, dtype=np.float32))
            scores = self._matrix @ query
            indices = np.argsort(-scores)
            hits: list[VectorHit] = []
            allowed = set(doc_ids) if doc_ids else None
            for idx in indices:
                chunk_id = self._ids[int(idx)]
                if allowed is not None and self._doc_ids[int(idx)] not in allowed:
                    continue
                score = float(scores[int(idx)])
                hits.append(VectorHit(chunk_id=chunk_id, score=max(0.0, min(1.0, score)), distance=1 - score))
                if len(hits) >= top_k:
                    break
            return hits

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()
            self._load()

    def health(self) -> dict:
        return {"backend": self.name, "count": self.count(), "path": str(self.path)}


def create_vector_store(settings: Settings) -> BaseVectorStore:
    backend = settings.vector_backend.strip().lower()
    if backend == "local":
        return LocalVectorStore(settings.data_dir / "local_vectors.db")
    if backend in {"chroma", "auto"}:
        try:
            store = ChromaVectorStore(settings)
            logger.info("向量后端: Chroma (%s)", settings.chroma_dir)
            return store
        except Exception as exc:
            if backend == "chroma":
                raise
            logger.warning("Chroma 初始化失败，回退到本地向量后端: %s", exc)
    return LocalVectorStore(settings.data_dir / "local_vectors.db")
