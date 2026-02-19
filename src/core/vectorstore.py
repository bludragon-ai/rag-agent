"""Vector store factory — Chroma or FAISS."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.vectorstores import VectorStore

from src.core.config import Settings, VectorStoreBackend

logger = logging.getLogger(__name__)


def build_vectorstore(settings: Settings, embeddings) -> VectorStore:
    """Return a vector store, loading existing data if available."""

    if settings.vector_store == VectorStoreBackend.CHROMA:
        from langchain_community.vectorstores import Chroma
        persist_dir = settings.chroma_persist_dir
        logger.info("Using Chroma vector store at: %s", persist_dir)
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    if settings.vector_store == VectorStoreBackend.FAISS:
        from langchain_community.vectorstores import FAISS
        index_path = Path("data/vectorstore/faiss_index")
        if index_path.exists():
            logger.info("Loading existing FAISS index from: %s", index_path)
            return FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        logger.info("Creating new FAISS index")
        return FAISS.from_texts(["__placeholder__"], embeddings)

    raise ValueError(f"Unknown vector store backend: {settings.vector_store}")


def save_faiss_index(vectorstore, settings: Settings) -> None:
    """Persist a FAISS index to disk."""
    if settings.vector_store != VectorStoreBackend.FAISS:
        return
    index_path = Path("data/vectorstore/faiss_index")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info("FAISS index saved to: %s", index_path)
