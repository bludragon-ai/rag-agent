"""
Centralized configuration via pydantic-settings.

All settings are loaded from environment variables (or .env file).
Defaults are tuned for a reasonable out-of-the-box experience.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"


class Settings(BaseSettings):
    """Application settings — populated from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    vector_store: VectorStoreBackend = VectorStoreBackend.CHROMA
    chroma_persist_dir: str = "data/vectorstore"

    # Chunking
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=2000)

    # Retrieval
    retrieval_top_k: int = Field(default=4, ge=1, le=20)

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8501
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Singleton accessor — cached after first call."""
    return Settings()
