"""Tests for the RAG chain utilities."""

from langchain_core.documents import Document

from src.core.chain import format_documents


def test_format_documents_basic():
    """Should format documents with source headers."""
    docs = [
        Document(page_content="Content A", metadata={"filename": "doc1.pdf", "page": 0}),
        Document(page_content="Content B", metadata={"filename": "doc2.md"}),
    ]
    result = format_documents(docs)

    assert "[Source 1: doc1.pdf, p.1]" in result
    assert "[Source 2: doc2.md]" in result
    assert "Content A" in result
    assert "Content B" in result


def test_format_documents_empty():
    """Should handle empty document list."""
    assert format_documents([]) == ""
