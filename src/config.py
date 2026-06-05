"""
Configuration management for Drug Repurposing Agents.
Loads settings from environment variables via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    groq_temperature: float = Field(default=0.1)
    groq_max_tokens: int = Field(default=4096)

    # Optional API keys
    ncbi_api_key: str = Field(default="")
    openfda_api_key: str = Field(default="")

    # App
    demo_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Paths
    chroma_db_path: Path = Field(default=Path("./data/chroma"))
    kg_data_path: Path = Field(default=Path("./data/kg"))
    cache_path: Path = Field(default=Path("./data/cache"))

    # API settings
    request_timeout: int = Field(default=30)
    max_retries: int = Field(default=3)

    # RAG settings
    rag_top_k: int = Field(default=8)
    rag_chunk_size: int = Field(default=512)
    rag_chunk_overlap: int = Field(default=64)

    # Scoring weights (must sum to 1.0)
    weight_mechanistic: float = Field(default=0.35)
    weight_literature: float = Field(default=0.30)
    weight_clinical: float = Field(default=0.20)
    weight_data_confidence: float = Field(default=0.15)

    def ensure_dirs(self) -> None:
        """Create all required data directories."""
        for path in [self.chroma_db_path, self.kg_data_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
