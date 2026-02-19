"""Document ingestion — load, chunk, embed, store."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore

from src.core.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def ingest_documents(
    file_paths: list[Path],
    vectorstore: VectorStore,
    settings: Settings,
) -> int:
    """
    Load, chunk, and index documents into the vector store.
    Returns total number of chunks added.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks = []

    for path in file_paths:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", path.name)
            continue

        try:
            if ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(path))
            else:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(path), encoding="utf-8")

            docs = loader.load()
            chunks = splitter.split_documents(docs)

            # Stamp metadata
            for chunk in chunks:
                chunk.metadata["filename"] = path.name
                chunk.metadata["source"] = str(path)

            all_chunks.extend(chunks)
            logger.info("Ingested %s → %d chunks", path.name, len(chunks))

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", path.name, exc)

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        logger.info("Added %d total chunks to vector store", len(all_chunks))

    return len(all_chunks)
