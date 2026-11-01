"""SQLite metadata store: documents, chunks, sessions, caches and evals.

The database is the source of truth for chunk text and metadata; vector
backends only store ids + vectors and are rebuilt from here when needed.
A fresh connection is opened per operation so the store is safe to use from
FastAPI's threadpool and background evaluation threads.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from app.core.types import Chunk
from app.core.utils import json_dumps, json_loads, sha256_text, utc_now_iso, utc_now_ts

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_key TEXT NOT NULL UNIQUE,
    path TEXT,
    mime TEXT,
    size INTEGER DEFAULT 0,
    sha256 TEXT,
    status TEXT DEFAULT 'ready',
    error TEXT,
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    indexed_at TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    doc_name TEXT,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    heading_path TEXT,
    content_type TEXT,
    token_estimate INTEGER,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    client TEXT DEFAULT 'web',
    user_id TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    trace TEXT,
    confidence REAL,
    latency_ms INTEGER DEFAULT 0,
    cached INTEGER DEFAULT 0,
    request_id TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    rating INTEGER,
    comment TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    dataset TEXT,
    status TEXT,
    progress INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    config TEXT,
    metrics TEXT,
    cases TEXT,
    error TEXT,
    created_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    kind TEXT,
    value TEXT,
    created_at TEXT,
    expires_at REAL,
    hits INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS semantic_cache (
    id TEXT PRIMARY KEY,
    query TEXT,
    query_norm TEXT,
    embedding BLOB,
    value TEXT,
    version INTEGER,
    created_at REAL,
    hits INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_semantic_version ON semantic_cache(version);

CREATE TABLE IF NOT EXISTS embedding_cache (
    hash TEXT PRIMARY KEY,
    model TEXT,
    dim INTEGER,
    vector BLOB,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    openid TEXT UNIQUE,
    nickname TEXT,
    avatar TEXT,
    created_at TEXT,
    last_seen_at TEXT
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------ basics
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
            if "doc_name" not in columns:
                conn.execute("ALTER TABLE chunks ADD COLUMN doc_name TEXT")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.transaction() as conn:
            cur = conn.execute(sql, params)
            return int(cur.lastrowid or 0)

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # --------------------------------------------------------------- documents
    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return self._doc_row(row)

    def get_document_by_source(self, source_key: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM documents WHERE source_key = ?", (source_key,))
        return self._doc_row(row)

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.query_all(
            "SELECT * FROM documents ORDER BY datetime(updated_at) DESC, name ASC"
        )
        return [self._doc_row(r) for r in rows if r]

    def upsert_document(self, doc: dict[str, Any]) -> None:
        params = (
            doc["id"], doc["name"], doc["source_key"], doc.get("path"), doc.get("mime", ""),
            int(doc.get("size", 0)), doc.get("sha256", ""), doc.get("status", "ready"),
            doc.get("error"), int(doc.get("page_count", 0)), int(doc.get("chunk_count", 0)),
            doc.get("created_at") or utc_now_iso(), utc_now_iso(), doc.get("indexed_at"),
            json_dumps(doc.get("metadata") or {}),
        )
        self.execute(
            """
            INSERT INTO documents
              (id, name, source_key, path, mime, size, sha256, status, error,
               page_count, chunk_count, created_at, updated_at, indexed_at, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
               name=excluded.name, source_key=excluded.source_key, path=excluded.path,
               mime=excluded.mime, size=excluded.size, sha256=excluded.sha256,
               status=excluded.status, error=excluded.error,
               page_count=excluded.page_count, chunk_count=excluded.chunk_count,
               updated_at=excluded.updated_at, indexed_at=excluded.indexed_at,
               metadata=excluded.metadata
            """,
            params,
        )

    def save_document_chunks(self, doc: dict[str, Any], chunks: Sequence[Chunk]) -> None:
        """Atomically upsert a document and replace all of its chunks."""
        now = utc_now_iso()
        doc = dict(doc)
        doc.setdefault("created_at", now)
        doc["updated_at"] = now
        doc["chunk_count"] = len(chunks)
        doc["indexed_at"] = now
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO documents
                  (id, name, source_key, path, mime, size, sha256, status, error,
                   page_count, chunk_count, created_at, updated_at, indexed_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, source_key=excluded.source_key, path=excluded.path,
                   mime=excluded.mime, size=excluded.size, sha256=excluded.sha256,
                   status=excluded.status, error=excluded.error,
                   page_count=excluded.page_count, chunk_count=excluded.chunk_count,
                   updated_at=excluded.updated_at, indexed_at=excluded.indexed_at,
                   metadata=excluded.metadata
                """,
                (
                    doc["id"], doc["name"], doc["source_key"], doc.get("path"), doc.get("mime", ""),
                    int(doc.get("size", 0)), doc.get("sha256", ""), doc.get("status", "ready"),
                    doc.get("error"), int(doc.get("page_count", 0)), len(chunks),
                    doc["created_at"], now, now, json_dumps(doc.get("metadata") or {}),
                ),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc["id"],))
            conn.executemany(
                """
                INSERT INTO chunks (id, doc_id, doc_name, ordinal, text, page_start, page_end,
                                    heading_path, content_type, token_estimate, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        c.id, c.doc_id, c.doc_name, c.ordinal, c.text, c.page_start, c.page_end,
                        json_dumps(c.heading_path), c.content_type, c.token_estimate,
                        json_dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )

    def set_document_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        self.execute(
            "UPDATE documents SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, utc_now_iso(), doc_id),
        )

    def delete_document(self, doc_id: str) -> None:
        self.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def doc_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM documents")
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ chunks
    def replace_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        now = utc_now_iso()
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.executemany(
                """
                INSERT INTO chunks (id, doc_id, doc_name, ordinal, text, page_start, page_end,
                                    heading_path, content_type, token_estimate, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        c.id, c.doc_id, c.doc_name, c.ordinal, c.text, c.page_start, c.page_end,
                        json_dumps(c.heading_path), c.content_type, c.token_estimate,
                        json_dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )
            conn.execute(
                "UPDATE documents SET chunk_count=?, indexed_at=?, updated_at=? WHERE id=?",
                (len(chunks), now, now, doc_id),
            )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self.query_one("SELECT * FROM chunks WHERE id=?", (chunk_id,))
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Iterable[str]) -> dict[str, Chunk]:
        ids = [c for c in dict.fromkeys(chunk_ids) if c]
        if not ids:
            return {}
        out: dict[str, Chunk] = {}
        for start in range(0, len(ids), 400):
            batch = ids[start : start + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = self.query_all(f"SELECT * FROM chunks WHERE id IN ({placeholders})", batch)
            for row in rows:
                chunk = self._row_to_chunk(row)
                out[chunk.id] = chunk
        return out

    def get_doc_chunks(self, doc_id: str) -> list[Chunk]:
        rows = self.query_all("SELECT * FROM chunks WHERE doc_id=? ORDER BY ordinal", (doc_id,))
        return [self._row_to_chunk(r) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        rows = self.query_all("SELECT * FROM chunks ORDER BY doc_id, ordinal")
        return [self._row_to_chunk(r) for r in rows]

    def chunk_count(self) -> int:
        row = self.query_one("SELECT COUNT(*) AS n FROM chunks")
        return int(row["n"]) if row else 0

    def get_neighbor_chunks(self, chunk_ids: Sequence[str], radius: int = 1) -> dict[str, list[Chunk]]:
        """Return ±radius neighbours for each source chunk, keyed by source id."""
        out: dict[str, list[Chunk]] = {}
        for cid in dict.fromkeys(chunk_ids):
            chunk = self.get_chunk(cid)
            if not chunk:
                continue
            rows = self.query_all(
                "SELECT * FROM chunks WHERE doc_id=? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
                (chunk.doc_id, chunk.ordinal - radius, chunk.ordinal + radius),
            )
            out[cid] = [self._row_to_chunk(r) for r in rows if r["id"] != cid]
        return out

    # --------------------------------------------------------------------- kv
    def get_kv(self, key: str, default: str | None = None) -> str | None:
        row = self.query_one("SELECT value FROM kv WHERE key=?", (key,))
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO kv(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_index_version(self) -> int:
        return int(self.get_kv("index_version", "0") or 0)

    def bump_index_version(self) -> int:
        with self.transaction() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key='index_version'").fetchone()
            version = int(row["value"]) + 1 if row else 1
            conn.execute(
                "INSERT INTO kv(key, value) VALUES('index_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(version),),
            )
        return version

    # ---------------------------------------------------------------- sessions
    def create_session(
        self, session_id: str, title: str = "新对话", client: str = "web", user_id: str | None = None
    ) -> dict[str, Any]:
        now = utc_now_iso()
        self.execute(
            "INSERT OR IGNORE INTO sessions(id, title, client, user_id, created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (session_id, title, client, user_id, now, now),
        )
        session = self.get_session(session_id)
        return session or {"id": session_id, "title": title, "client": client, "user_id": user_id}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM sessions WHERE id=?", (session_id,))

    def touch_session(self, session_id: str, title: str | None = None) -> None:
        if title:
            self.execute(
                "UPDATE sessions SET updated_at=?, title=? WHERE id=?",
                (utc_now_iso(), title, session_id),
            )
        else:
            self.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (utc_now_iso(), session_id)
            )

    def list_sessions(self, client: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if client:
            rows = self.query_all(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS message_count "
                "FROM sessions s WHERE client=? ORDER BY datetime(updated_at) DESC LIMIT ?",
                (client, limit),
            )
        else:
            rows = self.query_all(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS message_count "
                "FROM sessions s ORDER BY datetime(updated_at) DESC LIMIT ?",
                (limit,),
            )
        return rows

    def delete_session(self, session_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ---------------------------------------------------------------- messages
    def add_message(self, message: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO messages(id, session_id, role, content, citations, trace, confidence,
                                 latency_ms, cached, request_id, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                message["id"], message["session_id"], message["role"], message["content"],
                json_dumps(message.get("citations") or []),
                json_dumps(message.get("trace") or []),
                float(message.get("confidence") or 0.0),
                int(message.get("latency_ms") or 0),
                1 if message.get("cached") else 0,
                message.get("request_id"),
                message.get("created_at") or utc_now_iso(),
            ),
        )

    def get_messages(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.query_all(
            "SELECT * FROM messages WHERE session_id=? ORDER BY datetime(created_at) ASC, rowid ASC LIMIT ?",
            (session_id, limit),
        )
        for row in rows:
            row["citations"] = json_loads(row.get("citations"), []) or []
            row["trace"] = json_loads(row.get("trace"), []) or []
            row["cached"] = bool(row.get("cached"))
        return rows

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.query_all(
            "SELECT * FROM messages WHERE session_id=? ORDER BY datetime(created_at) DESC, rowid DESC LIMIT ?",
            (session_id, limit),
        )
        rows.reverse()
        for row in rows:
            row["citations"] = json_loads(row.get("citations"), []) or []
            row["trace"] = []
        return rows

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM messages WHERE id=?", (message_id,))
        if row:
            row["citations"] = json_loads(row.get("citations"), []) or []
            row["trace"] = json_loads(row.get("trace"), []) or []
        return row

    # ---------------------------------------------------------------- feedback
    def add_feedback(
        self, feedback_id: str, message_id: str, session_id: str, rating: int, comment: str = ""
    ) -> None:
        self.execute(
            "INSERT INTO feedback(id, message_id, session_id, rating, comment, created_at) VALUES(?,?,?,?,?,?)",
            (feedback_id, message_id, session_id, rating, comment, utc_now_iso()),
        )

    # ------------------------------------------------------------------- cache
    def cache_get(self, key: str) -> dict[str, Any] | None:
        now = utc_now_ts()
        row = self.query_one(
            "SELECT value, expires_at FROM cache_entries WHERE key=? AND expires_at > ?",
            (key, now),
        )
        if not row:
            return None
        self.execute("UPDATE cache_entries SET hits=hits+1 WHERE key=?", (key,))
        return json_loads(row["value"], None)

    def cache_set(self, key: str, kind: str, value: Any, ttl: int) -> None:
        self.execute(
            "INSERT INTO cache_entries(key, kind, value, created_at, expires_at, hits) VALUES(?,?,?,?,?,0) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at, "
            "expires_at=excluded.expires_at",
            (key, kind, json_dumps(value), utc_now_iso(), utc_now_ts() + ttl),
        )

    def cache_clear(self) -> None:
        self.execute("DELETE FROM cache_entries")
        self.execute("DELETE FROM semantic_cache")

    def cache_stats(self) -> dict[str, Any]:
        exact = self.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(hits),0) AS hits FROM cache_entries WHERE expires_at > ?",
            (utc_now_ts(),),
        ) or {"n": 0, "hits": 0}
        semantic = self.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(hits),0) AS hits FROM semantic_cache"
        ) or {"n": 0, "hits": 0}
        return {
            "exact_entries": int(exact["n"]),
            "exact_hits": int(exact["hits"]),
            "semantic_entries": int(semantic["n"]),
            "semantic_hits": int(semantic["hits"]),
        }

    def semantic_candidates(self, version: int, limit: int = 400) -> list[dict[str, Any]]:
        rows = self.query_all(
            "SELECT id, query, query_norm, embedding, value, version FROM semantic_cache "
            "WHERE version=? ORDER BY created_at DESC LIMIT ?",
            (version, limit),
        )
        for row in rows:
            blob = row.get("embedding") or b""
            row["vector"] = np.frombuffer(blob, dtype=np.float32) if blob else np.zeros(0, dtype=np.float32)
            row["value"] = json_loads(row.get("value"), None)
        return rows

    def semantic_set(
        self, entry_id: str, query: str, query_norm: str, embedding: np.ndarray,
        value: Any, version: int,
    ) -> None:
        self.execute(
            "INSERT OR REPLACE INTO semantic_cache(id, query, query_norm, embedding, value, version, created_at, hits) "
            "VALUES(?,?,?,?,?,?,?,0)",
            (
                entry_id, query, query_norm, np.asarray(embedding, dtype=np.float32).tobytes(),
                json_dumps(value), version, utc_now_ts(),
            ),
        )

    def semantic_hit(self, entry_id: str) -> None:
        self.execute("UPDATE semantic_cache SET hits=hits+1 WHERE id=?", (entry_id,))

    def prune_semantic(self, keep_version: int, keep_rows: int = 400) -> None:
        self.execute("DELETE FROM semantic_cache WHERE version != ?", (keep_version,))
        self.execute(
            "DELETE FROM semantic_cache WHERE id NOT IN "
            "(SELECT id FROM semantic_cache ORDER BY created_at DESC LIMIT ?)",
            (keep_rows,),
        )

    # ------------------------------------------------------------- embeddings
    def get_embeddings(self, hashes: Sequence[str]) -> dict[str, np.ndarray]:
        if not hashes:
            return {}
        out: dict[str, np.ndarray] = {}
        for start in range(0, len(hashes), 500):
            batch = list(hashes[start : start + 500])
            placeholders = ",".join("?" for _ in batch)
            rows = self.query_all(
                f"SELECT hash, vector FROM embedding_cache WHERE hash IN ({placeholders})", batch
            )
            for row in rows:
                out[row["hash"]] = np.frombuffer(row["vector"], dtype=np.float32)
        return out

    def put_embeddings(self, entries: Sequence[tuple[str, str, np.ndarray]]) -> None:
        if not entries:
            return
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache(hash, model, dim, vector, created_at) VALUES(?,?,?,?,?)",
                [
                    (h, model, int(vec.shape[0]), np.asarray(vec, dtype=np.float32).tobytes(), utc_now_ts())
                    for h, model, vec in entries
                ],
            )

    # -------------------------------------------------------------------- eval
    def create_eval_run(self, run: dict[str, Any]) -> None:
        self.execute(
            "INSERT INTO eval_runs(id, dataset, status, progress, total, config, created_at) VALUES(?,?,?,?,?,?,?)",
            (
                run["id"], run["dataset"], run.get("status", "queued"), 0,
                int(run.get("total", 0)), json_dumps(run.get("config") or {}), utc_now_iso(),
            ),
        )

    def update_eval_run(
        self, run_id: str, *, status: str | None = None, progress: int | None = None,
        metrics: Any = None, cases: Any = None, error: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if status is not None:
            sets.append("status=?")
            params.append(status)
        if progress is not None:
            sets.append("progress=?")
            params.append(progress)
        if metrics is not None:
            sets.append("metrics=?")
            params.append(json_dumps(metrics))
        if cases is not None:
            sets.append("cases=?")
            params.append(json_dumps(cases))
        if error is not None:
            sets.append("error=?")
            params.append(error)
        if status in {"completed", "failed"}:
            sets.append("finished_at=?")
            params.append(utc_now_iso())
        params.append(run_id)
        self.execute(f"UPDATE eval_runs SET {', '.join(sets)} WHERE id=?", params)

    def get_eval_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.query_one("SELECT * FROM eval_runs WHERE id=?", (run_id,))
        return self._eval_row(row)

    def list_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.query_all("SELECT * FROM eval_runs ORDER BY datetime(created_at) DESC LIMIT ?", (limit,))
        return [self._eval_row(r) for r in rows if r]

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        docs = self.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(chunk_count),0) AS chunks FROM documents WHERE status='ready'"
        ) or {"n": 0, "chunks": 0}
        by_type = self.query_all(
            "SELECT CASE "
            "  WHEN lower(name) LIKE '%.pdf' THEN 'pdf' "
            "  WHEN lower(name) LIKE '%.md' OR lower(name) LIKE '%.markdown' THEN 'markdown' "
            "  WHEN lower(name) LIKE '%.png' OR lower(name) LIKE '%.jpg' OR lower(name) LIKE '%.jpeg' "
            "       OR lower(name) LIKE '%.webp' OR lower(name) LIKE '%.gif' THEN 'image' "
            "  ELSE 'other' END AS kind, COUNT(*) AS n "
            "FROM documents WHERE status='ready' GROUP BY kind"
        )
        return {
            "documents": int(docs["n"]),
            "chunks": int(docs["chunks"]),
            "sessions": int((self.query_one("SELECT COUNT(*) AS n FROM sessions") or {"n": 0})["n"]),
            "messages": int((self.query_one("SELECT COUNT(*) AS n FROM messages") or {"n": 0})["n"]),
            "by_type": {row["kind"]: int(row["n"]) for row in by_type},
        }

    # ------------------------------------------------------------------ users
    def upsert_user(self, user: dict[str, Any]) -> dict[str, Any]:
        now = utc_now_iso()
        self.execute(
            "INSERT INTO users(id, openid, nickname, avatar, created_at, last_seen_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET nickname=excluded.nickname, avatar=excluded.avatar, "
            "last_seen_at=excluded.last_seen_at",
            (user["id"], user.get("openid"), user.get("nickname"), user.get("avatar"), now, now),
        )
        return self.query_one("SELECT * FROM users WHERE id=?", (user["id"],)) or user

    # ------------------------------------------------------------------ mappers
    @staticmethod
    def _doc_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        row["metadata"] = json_loads(row.get("metadata"), {}) or {}
        return row

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> Chunk:
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            doc_name=row.get("doc_name") or "",
            ordinal=int(row["ordinal"]),
            text=row["text"],
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            heading_path=json_loads(row.get("heading_path"), []) or [],
            content_type=row.get("content_type") or "text",
            token_estimate=int(row.get("token_estimate") or 0),
            metadata=json_loads(row.get("metadata"), {}) or {},
        )

    @staticmethod
    def _eval_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        row["config"] = json_loads(row.get("config"), {}) or {}
        row["metrics"] = json_loads(row.get("metrics"), {}) or {}
        row["cases"] = json_loads(row.get("cases"), []) or []
        return row


def make_doc_id(source_key: str) -> str:
    return "d_" + sha256_text(source_key)[:18]
