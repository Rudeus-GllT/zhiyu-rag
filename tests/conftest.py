from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.context import build_context  # noqa: E402


@pytest.fixture(scope="session")
def pipeline():
    # Keep runtime data inside the workspace; the DSH sandbox cannot write to
    # the platform temp directory on this machine.
    data_dir = ROOT / "data" / "pytest-runtime"
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        data_dir=data_dir,
        samples_dir=ROOT / "samples",
        auto_seed=False,
        vector_backend="local",
        llm_provider="mock",
        llm_api_key="",
        embedding_provider="hash",
        embedding_dim=128,
        chunk_size=320,
        chunk_overlap=40,
        enable_vlm=False,
        llm_vision_model="",
        llm_vision_base_url="",
        llm_vision_api_key="",
        ocr_enabled=False,
        enable_llm_query_analysis=False,
        enable_llm_rerank=False,
        cache_enabled=True,
        semantic_cache_enabled=False,
    )
    ctx = build_context(settings)
    yield ctx
