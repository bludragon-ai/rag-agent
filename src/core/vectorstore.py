"""
Vector store factory and management.

Supports ChromaDB (persistent, default) and FAISS (in-memory with
optional save/load). Both are local — no external services required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.core.config import Settings, VectorStoreBackend

logger = logging.getLogger(__name__)


def build_vectorstore(
    settings: Settings,
    embeddings: Embeddings,
) -> VectorStore:
    """Create or load the configured vector store."""

    match settings.vector_store:
        case VectorStoreBackend.CHROMA:
            import chromadb
            from langchain_community.vectorstores import Chroma

            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(path=str(persist_dir))
            return Chroma(
                client=client,
                collection_name="rag_documents",
                embedding_function=embeddings,
            )

        case VectorStoreBackend.FAISS:
            from langchain_community.vectorstores import FAISS

            faiss_path = Path(settings.chroma_persist_dir) / "faiss_index"
            if faiss_path.exists():
                logger.info("Loading existing FAISS index from %s", faiss_path)
                return FAISS.load_local(
                    str(faiss_path),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )

            # Return a fresh (empty) FAISS store — will be populated on first ingest
            logger.info("Creating new FAISS index")
            return FAISS.from_documents(
                [Document(page_content="placeholder", metadata={"source": "init"})],
                embeddings,
            )

        case _:
            raise ValueError(f"Unsupported vector store: {settings.vector_store}")


def save_faiss_index(store: VectorStore, settings: Settings) -> None:
    """Persist a FAISS index to disk (no-op for Chroma, which auto-persists)."""
    if settings.vector_store == VectorStoreBackend.FAISS:
        faiss_path = Path(settings.chroma_persist_dir) / "faiss_index"
        faiss_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(faiss_path))
        logger.info("FAISS index saved to %s", faiss_path)
