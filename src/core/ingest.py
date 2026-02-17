"""
Document ingestion pipeline.

Handles loading, splitting, and indexing documents into the vector store.
Supports PDF, TXT, and Markdown files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import Settings

logger = logging.getLogger(__name__)

# Supported file extensions → loader class paths
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path: Path) -> list[Document]:
    """
    Load a single document and return a list of LangChain Document objects.

    Each loader attaches source metadata automatically.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(file_path)).load()

    elif suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(file_path), encoding="utf-8").load()

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def split_documents(
    documents: list[Document],
    settings: Settings,
) -> list[Document]:
    """Split documents into chunks optimized for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d document(s) into %d chunks", len(documents), len(chunks))
    return chunks


def ingest_documents(
    file_paths: list[Path],
    vectorstore: VectorStore,
    settings: Settings,
) -> int:
    """
    Full ingestion pipeline: load → split → embed → store.

    Returns the number of chunks indexed.
    """
    all_chunks: list[Document] = []

    for path in file_paths:
        try:
            docs = load_document(path)
            # Enrich metadata with the original filename
            for doc in docs:
                doc.metadata["filename"] = path.name
            chunks = split_documents(docs, settings)
            all_chunks.extend(chunks)
            logger.info("Processed %s → %d chunks", path.name, len(chunks))
        except Exception:
            logger.exception("Failed to process %s", path.name)

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        logger.info("Indexed %d total chunks into vector store", len(all_chunks))

    return len(all_chunks)
