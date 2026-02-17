"""
Embedding model factory.

Supports HuggingFace (local, no API key) and OpenAI embeddings.
Controlled via the EMBEDDING_PROVIDER env var.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.core.config import Settings, EmbeddingProvider


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model used for document indexing and queries."""
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    # Default: HuggingFace (local, no API key needed)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
    )
