"""Dependency container built once at application startup."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.orchestrator import AgentService
from app.config import Settings
from app.core.cache import ResponseCache
from app.eval.runner import EvalRunner
from app.indexing.manager import IndexManager
from app.llm.service import LLMService
from app.retrieval.context import ContextBuilder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.verifier import CitationVerifier
from app.storage.db import Database

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: Settings
    db: Database
    llm: LLMService
    index: IndexManager
    retriever: HybridRetriever
    reranker: Reranker
    context_builder: ContextBuilder
    verifier: CitationVerifier
    cache: ResponseCache
    agent: AgentService
    eval_runner: EvalRunner


def build_context(settings: Settings) -> AppContext:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    llm = LLMService(settings, db=db)
    index = IndexManager(settings, db, llm)
    index.ensure_ready()
    retriever = HybridRetriever(db, index.vectors, index.bm25, llm.embedder, settings)
    reranker = Reranker(llm, settings)
    context_builder = ContextBuilder(db, settings)
    verifier = CitationVerifier()
    cache = ResponseCache(db, llm.embedder, settings)
    agent = AgentService(
        settings=settings,
        db=db,
        index=index,
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        verifier=verifier,
        cache=cache,
    )
    eval_runner = EvalRunner(settings, db, agent, llm)
    logger.info(
        "上下文就绪: llm=%s embedding=%s vector=%s docs=%d chunks=%d",
        llm.llm.name,
        llm.embedder.name,
        index.vectors.name,
        db.doc_count(),
        db.chunk_count(),
    )
    return AppContext(
        settings=settings,
        db=db,
        llm=llm,
        index=index,
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        verifier=verifier,
        cache=cache,
        agent=agent,
        eval_runner=eval_runner,
    )
