"""Tests for document ingestion."""

import tempfile
from pathlib import Path

import pytest

from src.core.ingest import load_document, split_documents, SUPPORTED_EXTENSIONS
from src.core.config import Settings


def test_supported_extensions():
    """All expected file types should be supported."""
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS


def test_load_text_document():
    """Should load a plain text file into Document objects."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello, this is a test document.\nWith multiple lines.")
        f.flush()
        docs = load_document(Path(f.name))

    assert len(docs) == 1
    assert "Hello" in docs[0].page_content


def test_load_markdown_document():
    """Should load a markdown file."""
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Title\n\nSome content here.")
        f.flush()
        docs = load_document(Path(f.name))

    assert len(docs) == 1
    assert "Title" in docs[0].page_content


def test_unsupported_extension():
    """Should raise ValueError for unsupported file types."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(Path("file.xyz"))


def test_split_documents():
    """Should split documents into chunks."""
    from langchain_core.documents import Document

    docs = [Document(page_content="word " * 500, metadata={"source": "test"})]
    settings = Settings(chunk_size=200, chunk_overlap=50)
    chunks = split_documents(docs, settings)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 250  # some tolerance for splitting
