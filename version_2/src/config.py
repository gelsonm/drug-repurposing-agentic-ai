"""
Configuration management for Drug Repurposing v2.
Loads from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM — Primary: OpenAI ────────────────────────────────────────────
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.1)
    openai_max_tokens: int = Field(default=4096)

    # ── LLM — Fallback: Groq (free tier, fast) ───────────────────────────
    # Groq provides OpenAI-compatible API — used when OpenAI key is missing/fails
    # Free models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b
    # Get key at: console.groq.com (free, no credit card)
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

    # ── Optional API keys ─────────────────────────────────────────────────
    disgenet_api_key: str = Field(default="")
    ncbi_api_key: str = Field(default="")

    # ── App ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    demo_mode: bool = Field(default=False)

    # ── Paths ─────────────────────────────────────────────────────────────
    chroma_db_path: Path = Field(default=Path("./data/vector_store"))
    kg_data_path: Path = Field(default=Path("./data/kg"))

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_chunk_size: int = Field(default=512)
    rag_chunk_overlap: int = Field(default=64)
    rag_top_k: int = Field(default=10)
    rag_bm25_weight: float = Field(default=0.4)
    rag_dense_weight: float = Field(default=0.6)

    # ── Scoring weights ───────────────────────────────────────────────────
    weight_kg_proximity: float = Field(default=0.20)
    weight_moa_alignment: float = Field(default=0.25)
    weight_literature: float = Field(default=0.20)
    weight_pathway_plausibility: float = Field(default=0.25)
    weight_adversarial_adjustment: float = Field(default=0.10)

    # ── Pipeline ──────────────────────────────────────────────────────────
    max_candidates: int = Field(default=10)
    top_n_for_deep_analysis: int = Field(default=5)
    request_timeout: int = Field(default=30)
    max_retries: int = Field(default=3)

    def ensure_dirs(self) -> None:
        for p in [self.chroma_db_path, self.kg_data_path]:
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
