"""Pure-Python retrieval / answer metrics (Hit, MRR, nDCG, AP, F1)."""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

from app.core.utils import tokenize


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def mrr_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def context_recall(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / len(relevant)


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(tokenize(prediction, with_bigrams=True))
    truth_tokens = set(tokenize(ground_truth, with_bigrams=True))
    if not pred_tokens or not truth_tokens:
        return 0.0
    overlap = len(pred_tokens & truth_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.asarray(list(a), dtype=np.float32)
    vb = np.asarray(list(b), dtype=np.float32)
    if va.size == 0 or vb.size == 0:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), p))


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def answer_relevancy_heuristic(
    question: str, answer: str, ground_truth: str = ""
) -> float:
    """A cheap proxy for RAGAS answer-relevancy without an LLM judge."""
    if not answer.strip():
        return 0.0
    f1 = token_f1(answer, ground_truth) if ground_truth else 0.0
    question_terms = set(tokenize(question, with_bigrams=True))
    answer_terms = set(tokenize(answer, with_bigrams=True))
    overlap = len(question_terms & answer_terms) / max(1, len(question_terms))
    length_penalty = 0.0 if len(answer) < 8 else 1.0
    return max(0.0, min(1.0, 0.45 * f1 + 0.45 * overlap + 0.10 * length_penalty))
