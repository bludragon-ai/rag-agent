"""Embeddings factory — returns the right embeddings based on config."""

from __future__ import annotations

import logging

from src.core.config import EmbeddingProvider, Settings

logger = logging.getLogger(__name__)


def build_embeddings(settings: Settings):
    """Return an embeddings instance based on settings.embedding_provider."""

    if settings.embedding_provider == EmbeddingProvider.HUGGINGFACE:
        logger.info("Using HuggingFace embeddings: %s", settings.embedding_model)
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        logger.info("Using OpenAI embeddings")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=settings.openai_api_key)

    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
