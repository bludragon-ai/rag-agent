"""
Embedding model factory.

Uses OpenAI embeddings by default. The embedding model is independent
of the chat LLM provider — you can use Ollama for chat but still
embed with OpenAI, for example.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.core.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model used for document indexing and queries."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
