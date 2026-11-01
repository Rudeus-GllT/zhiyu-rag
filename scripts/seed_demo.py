"""Seed (or re-seed) the sample knowledge base and print a summary."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
os.chdir(ROOT)

from app.config import settings  # noqa: E402
from app.context import build_context  # noqa: E402

settings.auto_seed = False
ctx = build_context(settings)
results = ctx.index.seed_samples(force=True)
for result in results:
    print(f"{result.status:10s} {result.name:34s} chunks={result.chunk_count:3d} {result.message}")
print("-" * 70)
print("index:", ctx.index.stats())
for doc in ctx.db.list_documents():
    print(f"{doc['name']:34s} {doc['status']:6s} chunks={doc['chunk_count']:3d} type={doc['metadata'].get('source_type')}")
