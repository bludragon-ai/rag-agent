"""Tests for configuration module."""

import os
from unittest.mock import patch

from src.core.config import Settings, LLMProvider, VectorStoreBackend


def test_default_settings():
    """Settings should have sensible defaults."""
    settings = Settings()
    assert settings.llm_provider == LLMProvider.OPENAI
    assert settings.vector_store == VectorStoreBackend.CHROMA
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_top_k == 4


def test_settings_from_env():
    """Settings should pick up environment variables."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "CHUNK_SIZE": "500"}):
        settings = Settings()
        assert settings.llm_provider == LLMProvider.ANTHROPIC
        assert settings.chunk_size == 500


def test_vector_store_enum():
    """Both vector store backends should be valid."""
    assert VectorStoreBackend("chroma") == VectorStoreBackend.CHROMA
    assert VectorStoreBackend("faiss") == VectorStoreBackend.FAISS
