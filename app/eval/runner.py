"""Evaluation runner persisted to SQLite and driven in a background thread.

Datasets are JSONL files under ``samples/eval``; each line:

    {"id":"q1","question":"...","ground_truth":"...",
     "relevant_sources":[{"doc_name":"x.md","evidence":"..."}],
     "tags":["retrieval"]}

Retrieval metrics are computed against chunks resolved from the labels, not
against the model's own citations, so the numbers are not self-fulfilling.
Generation metrics use citation accuracy (faithfulness proxy), token F1 and a
question/answer lexical-overlap relevancy proxy.  An optional RAGAS adapter
can be dropped in later without changing the API.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.agent.orchestrator import AgentService
from app.config import Settings
from app.core.types import Chunk
from app.eval.metrics import (
    answer_relevancy_heuristic,
    average_precision,
    context_recall,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    token_f1,
)
from app.llm.service import LLMService
from app.storage.db import Database

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(
        self, settings: Settings, db: Database, agent: AgentService, llm: LLMService
    ):
        self.settings = settings
        self.db = db
        self.agent = agent
        self.llm = llm
        self._threads: dict[str, threading.Thread] = {}

    # ------------------------------------------------------------- datasets
    def list_datasets(self) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        if not self.settings.eval_dir.exists():
            return datasets
        for path in sorted(self.settings.eval_dir.glob("*.jsonl")):
            try:
                count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            except Exception:
                count = 0
            datasets.append(
                {
                    "name": path.stem,
                    "file": path.name,
                    "count": count,
                    "description": self._dataset_description(path),
                }
            )
        return datasets

    @staticmethod
    def _dataset_description(path: Path) -> str:
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            data = json.loads(first)
            return str(data.get("description") or "")
        except Exception:
            return ""

    def load_dataset(self, name: str, limit: int | None = None) -> list[dict[str, Any]]:
        safe = Path(name).name
        path = self.settings.eval_dir / f"{safe}.jsonl"
        if not path.exists():
            raise ValueError(f"评测集不存在: {name}")
        cases: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc
            if item.get("question"):
                cases.append(item)
        if limit:
            cases = cases[: max(1, int(limit))]
        return cases

    # --------------------------------------------------------------- runner
    def start(self, dataset: str, limit: int | None = None) -> str:
        cases = self.load_dataset(dataset, limit)
        if not cases:
            raise ValueError("评测集为空")
        run_id = "e_" + uuid.uuid4().hex[:14]
        self.db.create_eval_run(
            {
                "id": run_id,
                "dataset": dataset,
                "status": "running",
                "total": len(cases),
                "config": {
                    "top_k": self.settings.retrieve_top_k,
                    "rerank_top_n": self.settings.rerank_top_n,
                    "cache": False,
                    "embedding": self.llm.embedder.name,
                    "llm": self.llm.llm.name,
                },
            }
        )
        thread = threading.Thread(
            target=self._execute,
            args=(run_id, dataset, cases),
            name=f"eval-{run_id}",
            daemon=True,
        )
        self._threads[run_id] = thread
        thread.start()
        return run_id

    def _execute(self, run_id: str, dataset: str, cases: list[dict[str, Any]]) -> None:
        corpus = self.db.all_chunks()
        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            for index, case in enumerate(cases):
                question = str(case.get("question") or "")
                ground_truth = str(case.get("ground_truth") or "")
                labels = case.get("relevant_sources") or []
                relevant_chunk_ids, relevant_doc_ids = self._resolve_relevant(
                    corpus, labels
                )
                session_id = f"eval_{run_id}_{index}"
                try:
                    answer = self.agent.ask(
                        question,
                        session_id=session_id,
                        top_k=self.settings.retrieve_top_k,
                        client="eval",
                        use_cache=False,
                    )
                    retrieved_ids = answer.retrieved_ids or [
                        s.chunk_id for s in answer.sources
                    ]
                    retrieved_docs = [s.doc_id for s in answer.sources]

                    if relevant_chunk_ids:
                        hit = hit_at_k(retrieved_ids, relevant_chunk_ids, 5)
                        mrr = mrr_at_k(retrieved_ids, relevant_chunk_ids, 5)
                        ndcg = ndcg_at_k(retrieved_ids, relevant_chunk_ids, 5)
                        precision = average_precision(retrieved_ids, relevant_chunk_ids, 5)
                        recall = context_recall(retrieved_ids, relevant_chunk_ids, 5)
                    else:
                        relevant_docs = set(relevant_doc_ids)
                        hit = 1.0 if any(doc in relevant_docs for doc in retrieved_docs[:5]) else 0.0
                        mrr = next(
                            (1.0 / rank for rank, doc in enumerate(retrieved_docs[:5], start=1) if doc in relevant_docs),
                            0.0,
                        )
                        ndcg = hit
                        precision = hit
                        recall = hit

                    report = answer.citation_report
                    faithfulness = report.accuracy if report.total else 0.0
                    relevancy = answer_relevancy_heuristic(question, answer.answer, ground_truth)
                    f1 = token_f1(answer.answer, ground_truth) if ground_truth else 0.0
                    results.append(
                        {
                            "id": case.get("id") or index + 1,
                            "question": question,
                            "answer": answer.answer,
                            "ground_truth": ground_truth,
                            "metrics": {
                                "hit@5": round(hit, 4),
                                "mrr@5": round(mrr, 4),
                                "ndcg@5": round(ndcg, 4),
                                "context_precision": round(precision, 4),
                                "context_recall": round(recall, 4),
                                "faithfulness": round(faithfulness, 4),
                                "answer_relevancy": round(relevancy, 4),
                                "token_f1": round(f1, 4),
                                "citation_coverage": round(report.coverage, 4),
                                "latency_ms": answer.latency_ms,
                            },
                            "sources": [
                                {
                                    "doc_name": s.doc_name,
                                    "heading": s.heading,
                                    "page": s.page_start,
                                    "score": s.score,
                                }
                                for s in answer.sources
                            ],
                            "confidence": answer.confidence,
                            "retry_used": answer.retry_used,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    logger.exception("评测用例失败: %s", exc)
                    results.append(
                        {
                            "id": case.get("id") or index + 1,
                            "question": question,
                            "answer": "",
                            "ground_truth": ground_truth,
                            "metrics": {},
                            "sources": [],
                            "confidence": 0.0,
                            "retry_used": False,
                            "error": str(exc),
                        }
                    )
                finally:
                    try:
                        self.db.delete_session(session_id)
                    except Exception:
                        pass
                self.db.update_eval_run(
                    run_id, progress=index + 1, cases=results, status="running"
                )

            metrics = self._aggregate(results, time.perf_counter() - started)
            self.db.update_eval_run(
                run_id,
                status="completed",
                progress=len(cases),
                metrics=metrics,
                cases=results,
            )
        except Exception as exc:
            logger.exception("评测运行失败: %s", exc)
            self.db.update_eval_run(
                run_id, status="failed", error=str(exc)[:1000], cases=results
            )

    @staticmethod
    def _resolve_relevant(
        corpus: list[Chunk], labels: list[dict[str, Any]]
    ) -> tuple[set[str], set[str]]:
        relevant_chunks: set[str] = set()
        relevant_docs: set[str] = set()
        for label in labels:
            doc_name = str(label.get("doc_name") or "").strip().lower()
            evidence = str(label.get("evidence") or "").strip()
            evidence_norm = "".join(evidence.split()).lower()
            for chunk in corpus:
                name_match = (
                    not doc_name
                    or doc_name == chunk.doc_name.lower()
                    or doc_name in chunk.doc_name.lower()
                    or chunk.doc_name.lower() in doc_name
                )
                if not name_match:
                    continue
                relevant_docs.add(chunk.doc_id)
                if not evidence:
                    relevant_chunks.add(chunk.id)
                    continue
                text_norm = "".join(chunk.text.split()).lower()
                if evidence_norm and evidence_norm in text_norm:
                    relevant_chunks.add(chunk.id)
        if not relevant_chunks and relevant_docs:
            # Keep doc-level fallback in the caller via relevant_docs.
            pass
        return relevant_chunks, relevant_docs

    @staticmethod
    def _aggregate(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        keys = [
            "hit@5", "mrr@5", "ndcg@5", "context_precision", "context_recall",
            "faithfulness", "answer_relevancy", "token_f1", "citation_coverage",
        ]
        metrics: dict[str, Any] = {}
        valid = [r for r in results if r.get("metrics")]
        for key in keys:
            values = [float(r["metrics"][key]) for r in valid if key in r["metrics"]]
            metrics[key] = round(sum(values) / len(values), 4) if values else 0.0
        latencies = [float(r["metrics"]["latency_ms"]) for r in valid if "latency_ms" in r["metrics"]]
        metrics["latency_p50_ms"] = round(percentile(latencies, 50), 1)
        metrics["latency_p95_ms"] = round(percentile(latencies, 95), 1)
        metrics["cases"] = len(results)
        metrics["failed"] = sum(1 for r in results if r.get("error"))
        metrics["pass_rate"] = round(
            sum(
                1
                for r in valid
                if r["metrics"].get("hit@5", 0) >= 0.5
                and r["metrics"].get("faithfulness", 0) >= 0.5
            )
            / max(1, len(valid)),
            4,
        )
        metrics["elapsed_seconds"] = round(elapsed, 2)
        return metrics
