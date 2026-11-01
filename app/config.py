"""Application configuration.

All settings can be overridden through environment variables or a local
``.env`` file.  The default profile is deliberately zero-config: with no API
key the app runs a deterministic local LLM (extractive) and a local hash
embedding, so every feature remains demonstrable offline.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- app
    app_name: str = "知屿"
    app_subtitle: str = "个人多模态知识库智能体"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    samples_dir: Path = PROJECT_ROOT / "samples"
    web_dir: Path = PROJECT_ROOT / "app" / "web"
    auto_seed: bool = True
    cors_origins: str = "*"

    # ---------------------------------------------------------------- llm
    # provider: auto | openai | mock  (openai = any OpenAI-compatible endpoint)
    llm_provider: str = "auto"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_vision_model: str = ""
    llm_vision_base_url: str = ""
    llm_vision_api_key: str = ""
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.15
    enable_llm_rerank: bool = True
    enable_llm_query_analysis: bool = True
    enable_llm_citation_judge: bool = False

    # --------------------------------------------------------- embeddings
    # provider: auto | hash | openai
    embedding_provider: str = "auto"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 10

    # -------------------------------------------------------------- index
    vector_backend: str = "auto"  # auto | chroma | local
    chroma_collection: str = "zhiyu_chunks"
    chunk_size: int = 720
    chunk_overlap: int = 120
    max_chunk_chars: int = 1400
    min_chunk_chars: int = 40
    heading_boost: float = 0.08

    # ---------------------------------------------------------- retrieval
    dense_top_k: int = 14
    bm25_top_k: int = 14
    retrieve_top_k: int = 8
    rerank_top_n: int = 6
    max_subqueries: int = 3
    context_token_budget: int = 3200
    max_context_chunks: int = 8
    neighbor_expansion: int = 1
    rrf_k: int = 60
    min_confidence: float = 0.46
    retrieval_retry_confidence: float = 0.34
    min_relevance: float = 0.20

    # -------------------------------------------------------------- cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.94
    semantic_cache_max_rows: int = 400

    # ------------------------------------------------------------- vision
    enable_vlm: bool = True
    vlm_max_images_per_doc: int = 6
    ocr_enabled: bool = True
    ocr_lang: str = "ch"
    tesseract_cmd: str = ""
    max_image_pixels: int = 12_000_000

    # ----------------------------------------------------------- uploads
    max_upload_mb: int = 30
    allowed_extensions: str = ".pdf,.md,.markdown,.txt,.png,.jpg,.jpeg,.webp,.gif,.docx,.csv,.json"

    # -------------------------------------------------------------- auth
    dashboard_token: str = ""

    # --------------------------------------------------------------- eval
    eval_dir: Path = PROJECT_ROOT / "samples" / "eval"

    # ------------------------------------------------------------ helpers
    @property
    def db_path(self) -> Path:
        return self.data_dir / "zhiyu.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}

    @property
    def llm_key(self) -> str:
        return self.llm_api_key.strip()

    @property
    def vision_base_url(self) -> str:
        return (self.llm_vision_base_url or self.llm_base_url).rstrip("/")

    @property
    def vision_key(self) -> str:
        return (self.llm_vision_api_key or self.llm_api_key).strip()

    @property
    def embed_key(self) -> str:
        return (self.embedding_api_key or self.llm_api_key).strip()

    @property
    def embed_base_url(self) -> str:
        return (self.embedding_base_url or self.llm_base_url).rstrip("/")

    @property
    def allowed_suffixes(self) -> set[str]:
        return {x.strip().lower() for x in self.allowed_extensions.split(",") if x.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [x.strip() for x in self.cors_origins.split(",") if x.strip()]

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.uploads_dir,
            self.chroma_dir,
            self.extracted_dir,
            self.index_dir,
            self.eval_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
