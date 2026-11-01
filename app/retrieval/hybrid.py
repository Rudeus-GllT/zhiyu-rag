"""Multi-query dense + BM25 recall fused with reciprocal rank fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from app.config import Settings
from app.core.types import QueryPlan, RetrievedChunk
from app.core.utils import dedupe, reciprocal_rank_fusion, tokenize
from app.llm.embeddings import BaseEmbedder
from app.storage.bm25_index import BM25Index
from app.storage.db import Database
from app.storage.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStats:
    queries: list[str] = field(default_factory=list)
    dense_hits: int = 0
    bm25_hits: int = 0
    fused_candidates: int = 0
    returned: int = 0
    doc_filter: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "queries": self.queries,
            "dense_hits": self.dense_hits,
            "bm25_hits": self.bm25_hits,
            "fused_candidates": self.fused_candidates,
            "returned": self.returned,
            "doc_filter": self.doc_filter,
        }


class HybridRetriever:
    def __init__(
        self,
        db: Database,
        vectors: BaseVectorStore,
        bm25: BM25Index,
        embedder: BaseEmbedder,
        settings: Settings,
    ):
        self.db = db
        self.vectors = vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings

    def retrieve(
        self,
        plan: QueryPlan,
        *,
        top_k: int | None = None,
        extra_queries: Sequence[str] = (),
        doc_ids: Sequence[str] | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalStats]:
        top_k = top_k or self.settings.retrieve_top_k
        queries = dedupe([*plan.all_queries(), *extra_queries])[:6]
        stats = RetrievalStats(queries=queries, doc_filter=list(doc_ids or plan.doc_ids or []))
        if not queries:
            return [], stats
        doc_filter = list(dict.fromkeys(doc_ids or plan.doc_ids or []))
        embeddings = self.embedder.embed(queries)

        rankings: list[list[str]] = []
        weights: list[float] = []
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        sources_map: dict[str, set[str]] = {}

        for query_index, query in enumerate(queries):
            dense_hits = self.vectors.search(
                embeddings[query_index],
                self.settings.dense_top_k,
                doc_ids=doc_filter or None,
            )
            stats.dense_hits += len(dense_hits)
            dense_ranking: list[str] = []
            for hit in dense_hits:
                dense_ranking.append(hit.chunk_id)
                dense_scores[hit.chunk_id] = max(
                    dense_scores.get(hit.chunk_id, 0.0), hit.score
                )
                sources_map.setdefault(hit.chunk_id, set()).add(f"dense:{query_index + 1}")
            if dense_ranking:
                rankings.append(dense_ranking)
                weights.append(1.0)

            bm25_hits = self.bm25.search(query, self.settings.bm25_top_k)
            stats.bm25_hits += len(bm25_hits)
            bm25_ranking: list[str] = []
            for chunk_id, score in bm25_hits:
                bm25_ranking.append(chunk_id)
                bm25_scores[chunk_id] = max(bm25_scores.get(chunk_id, 0.0), score)
                sources_map.setdefault(chunk_id, set()).add(f"bm25:{query_index + 1}")
            if bm25_ranking:
                rankings.append(bm25_ranking)
                weights.append(0.85)

        if not rankings:
            return [], stats
        fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k, weights=weights)
        stats.fused_candidates = len(fused)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        limit = max(top_k * 4, 32)
        candidate_ids = [chunk_id for chunk_id, _ in ordered[:limit]]
        chunks = self.db.get_chunks(candidate_ids)
        results: list[RetrievedChunk] = []
        for chunk_id, rrf_score in ordered[:limit]:
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue
            if doc_filter and chunk.doc_id not in doc_filter:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=rrf_score,
                    rrf_score=rrf_score,
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    bm25_score=bm25_scores.get(chunk_id, 0.0),
                    sources=sorted(sources_map.get(chunk_id, set())),
                )
            )
        # A tiny exact-match prior helps rare names / identifiers.
        for result in results:
            heading_tokens = set(tokenize(result.chunk.heading))
            result.score += 0.0005 * len(set(plan.keywords) & heading_tokens)
        results.sort(key=lambda r: r.score, reverse=True)
        stats.returned = len(results)
        return results[:limit], stats
