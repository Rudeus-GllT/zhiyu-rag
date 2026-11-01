"""Incremental index manager.

Document-level content hashing gives cheap add / update / delete detection.
Every successful mutation bumps a global ``index_version`` used by all cache
keys, then rebuilds the in-memory BM25 snapshot (atomic swap) and upserts
vectors.  Unchanged files are a no-op, which makes folder re-sync cheap.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, Sequence

from app.config import Settings
from app.core.types import Chunk, IngestResult
from app.core.utils import (
    guess_mime,
    safe_filename,
    sha256_bytes,
    sha256_file,
    utc_now_iso,
)
from app.ingest.chunker import chunk_document
from app.ingest.loaders import load_document
from app.llm.service import LLMService
from app.storage.bm25_index import BM25Index
from app.storage.db import Database, make_doc_id
from app.storage.vector_store import BaseVectorStore, create_vector_store

logger = logging.getLogger(__name__)


class IndexManager:
    def __init__(self, settings: Settings, db: Database, llm: LLMService):
        self.settings = settings
        self.db = db
        self.llm = llm
        self.vectors: BaseVectorStore = create_vector_store(settings)
        self.bm25 = BM25Index(settings.index_dir)
        self._lock = threading.RLock()
        self._ready = False

    # ------------------------------------------------------------------ boot
    def ensure_ready(self) -> None:
        with self._lock:
            version = self.db.get_index_version()
            chunks = self.db.all_chunks()
            if not chunks:
                self.bm25.reset()
                self._ready = True
                return
            if not self.bm25.load(version, len(chunks)):
                logger.info("构建 BM25 索引 (%d chunks, version=%d)", len(chunks), version)
                self.bm25.build(chunks, version)
            signature = self.llm.embedder.signature
            stored = self.db.get_kv("embedder_signature")
            vector_count = self.vectors.count()
            if stored != signature or vector_count != len(chunks):
                logger.info(
                    "向量索引需要修复: embedder=%s stored=%s vectors=%d chunks=%d",
                    signature, stored, vector_count, len(chunks),
                )
                signature_changed = stored is not None and stored != signature
                self.repair_vectors()
                if signature_changed:
                    # A new embedding space invalidates semantic answers too.
                    version = self.db.bump_index_version()
                    self.db.cache_clear()
                    self.bm25.build(self.db.all_chunks(), version)
            else:
                self.db.set_kv("embedder_signature", signature)
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    # --------------------------------------------------------------- ingest
    def ingest_path(
        self,
        path: Path,
        *,
        name: str | None = None,
        source_key: str | None = None,
        mime: str | None = None,
        force: bool = False,
    ) -> IngestResult:
        path = Path(path).resolve()
        if not path.exists() or not path.is_file():
            return IngestResult(
                doc_id="", name=name or path.name, status="failed", message="文件不存在"
            )
        name = name or path.name
        source_key = source_key or f"file:{name}"
        digest = sha256_file(path)
        existing = self.db.get_document_by_source(source_key)
        if (
            existing
            and existing.get("sha256") == digest
            and existing.get("status") == "ready"
            and not force
        ):
            return IngestResult(
                doc_id=existing["id"],
                name=existing["name"],
                status="unchanged",
                chunk_count=int(existing.get("chunk_count") or 0),
                version=self.db.get_index_version(),
                message="内容未变化，跳过索引",
            )
        doc_id = existing["id"] if existing else make_doc_id(source_key)
        with self._lock:
            try:
                parsed = load_document(
                    path,
                    name=name,
                    settings=self.settings,
                    vision=self.llm.vision,
                    doc_id=doc_id,
                )
                chunks = chunk_document(
                    parsed, doc_id=doc_id, doc_name=name, settings=self.settings
                )
                if not chunks:
                    raise ValueError("没有可索引的文本（请检查文件内容或 OCR 配置）")
                embeddings = self.llm.embedder.embed(
                    [f"{c.heading}\n{c.text}" if c.heading else c.text for c in chunks]
                )
                doc = self._document_record(
                    doc_id=doc_id,
                    name=name,
                    source_key=source_key,
                    path=path,
                    mime=mime or guess_mime(path.suffix),
                    digest=digest,
                    existing=existing,
                    parsed=parsed,
                    chunks=chunks,
                )
                self.db.save_document_chunks(doc, chunks)
                version = self.db.bump_index_version()
                vector_error = ""
                try:
                    self.vectors.delete(doc_id=doc_id)
                    self.vectors.add(chunks, embeddings, version=version)
                except Exception as exc:
                    vector_error = str(exc)
                    logger.exception("向量写入失败（仍保留 BM25 检索）: %s", exc)
                self._rebuild_bm25(version)
                self.db.set_kv("embedder_signature", self.llm.embedder.signature)
                self.db.prune_semantic(version, self.settings.semantic_cache_max_rows)
                status = "created" if not existing else "updated"
                message = "索引完成" if not vector_error else "索引完成（稠密向量降级，可执行修复）"
                if parsed.warnings:
                    message += f"；{len(parsed.warnings)} 条解析提示"
                return IngestResult(
                    doc_id=doc_id,
                    name=name,
                    status=status,
                    chunk_count=len(chunks),
                    version=version,
                    message=message,
                    metadata={
                        "warnings": parsed.warnings[:10],
                        "parser": parsed.metadata.get("parser"),
                        "source_type": parsed.source_type,
                        "vector_error": vector_error,
                    },
                )
            except Exception as exc:
                logger.exception("索引失败: %s", exc)
                if not existing:
                    self.db.upsert_document(
                        {
                            "id": doc_id,
                            "name": name,
                            "source_key": source_key,
                            "path": str(path),
                            "mime": mime or guess_mime(path.suffix),
                            "size": path.stat().st_size,
                            "sha256": digest,
                            "status": "error",
                            "error": str(exc)[:500],
                            "created_at": utc_now_iso(),
                        }
                    )
                return IngestResult(
                    doc_id=doc_id,
                    name=name,
                    status="failed",
                    version=self.db.get_index_version(),
                    message=str(exc),
                )

    def ingest_bytes(
        self, data: bytes, filename: str, *, source_key: str | None = None
    ) -> IngestResult:
        safe_name = safe_filename(filename)
        digest = sha256_bytes(data)
        target_dir = self.settings.uploads_dir / digest[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{digest[:12]}_{safe_name}"
        if not target.exists():
            target.write_bytes(data)
        return self.ingest_path(
            target,
            name=safe_name,
            source_key=source_key or f"upload:{safe_name}",
        )

    def _document_record(
        self,
        *,
        doc_id: str,
        name: str,
        source_key: str,
        path: Path,
        mime: str,
        digest: str,
        existing: dict | None,
        parsed,
        chunks: Sequence[Chunk],
    ) -> dict:
        warnings = list(dict.fromkeys(parsed.warnings))[:20]
        metadata = {
            "title": parsed.title,
            "source_type": parsed.source_type,
            "parser": parsed.metadata.get("parser"),
            "warnings": warnings,
            "parsed_metadata": parsed.metadata,
            "chunk_chars": sum(len(c.text) for c in chunks),
            "chunk_types": sorted({c.content_type for c in chunks}),
        }
        return {
            "id": doc_id,
            "name": name,
            "source_key": source_key,
            "path": str(path),
            "mime": mime,
            "size": path.stat().st_size,
            "sha256": digest,
            "status": "ready",
            "error": None,
            "page_count": int(parsed.page_count or 0),
            "chunk_count": len(chunks),
            "created_at": (existing or {}).get("created_at") or utc_now_iso(),
            "metadata": metadata,
        }

    # --------------------------------------------------------------- delete
    def delete_document(self, doc_id: str) -> bool:
        doc = self.db.get_document(doc_id)
        if not doc:
            return False
        with self._lock:
            try:
                self.vectors.delete(doc_id=doc_id)
            except Exception as exc:
                logger.warning("删除向量失败: %s", exc)
            self.db.delete_document(doc_id)
            version = self.db.bump_index_version()
            self._rebuild_bm25(version)
            self.db.prune_semantic(version, self.settings.semantic_cache_max_rows)
        return True

    def reindex_document(self, doc_id: str) -> IngestResult:
        doc = self.db.get_document(doc_id)
        if not doc:
            return IngestResult(doc_id=doc_id, name="", status="failed", message="文档不存在")
        path = Path(doc.get("path") or "")
        if not path.exists():
            return IngestResult(
                doc_id=doc_id, name=doc["name"], status="failed", message="原始文件已丢失"
            )
        return self.ingest_path(
            path,
            name=doc["name"],
            source_key=doc["source_key"],
            mime=doc.get("mime"),
            force=True,
        )

    # ------------------------------------------------------------- repair
    def repair_vectors(self) -> int:
        with self._lock:
            chunks = self.db.all_chunks()
            version = self.db.get_index_version()
            try:
                self.vectors.reset()
            except Exception as exc:
                logger.warning("重置向量库失败: %s", exc)
            if not chunks:
                self.db.set_kv("embedder_signature", self.llm.embedder.signature)
                return 0
            batch_size = 48
            total = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                embeddings = self.llm.embedder.embed(
                    [f"{c.heading}\n{c.text}" if c.heading else c.text for c in batch]
                )
                self.vectors.add(batch, embeddings, version=version)
                total += len(batch)
            self.db.set_kv("embedder_signature", self.llm.embedder.signature)
            logger.info("向量索引修复完成: %d chunks", total)
            return total

    def _rebuild_bm25(self, version: int) -> None:
        chunks = self.db.all_chunks()
        self.bm25.build(chunks, version)

    # -------------------------------------------------------------- samples
    def seed_samples(self, *, force: bool = False) -> list[IngestResult]:
        if not self.settings.auto_seed and not force:
            return []
        if self.db.get_kv("seeded") == "1" and not force:
            return []
        root = self.settings.samples_dir
        if not root.exists():
            return []
        results: list[IngestResult] = []
        for path in sorted(self._iter_sample_files(root)):
            rel = path.relative_to(root).as_posix()
            result = self.ingest_path(
                path,
                name=path.name,
                source_key=f"sample:{rel}",
                force=force,
            )
            results.append(result)
        self.db.set_kv("seeded", "1")
        return results

    def _iter_sample_files(self, root: Path) -> Iterable[Path]:
        allowed = self.settings.allowed_suffixes
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "eval" in path.relative_to(root).parts:
                continue
            if path.name.lower().startswith("readme"):
                continue
            # Image sidecar captions are consumed by the image loader; do not
            # index them as standalone documents.
            if path.name.lower().endswith(
                (".png.txt", ".jpg.txt", ".jpeg.txt", ".webp.txt", ".gif.txt")
            ):
                continue
            if path.name.lower().endswith(".caption.txt") or path.name.lower().endswith(".caption.md"):
                continue
            if path.suffix.lower() in allowed:
                yield path

    # ---------------------------------------------------------------- health
    def stats(self) -> dict:
        return {
            "index_version": self.db.get_index_version(),
            "documents": self.db.doc_count(),
            "chunks": self.db.chunk_count(),
            "bm25": self.bm25.health(),
            "vectors": self.vectors.health(),
            "embedder": {
                "name": self.llm.embedder.name,
                "dim": self.llm.embedder.dim,
                "signature": self.llm.embedder.signature,
            },
        }
