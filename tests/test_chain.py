"""Smoke tests for the RAG chain utilities."""

from __future__ import annotations

from langchain_core.documents import Document

from src.core.chain import format_documents, RAGResponse


def test_format_documents_basic():
    docs = [
        Document(page_content="Hello world", metadata={"filename": "test.pdf", "page": 0}),
        Document(page_content="Second chunk", metadata={"filename": "other.txt"}),
    ]
    result = format_documents(docs)
    assert "test.pdf" in result
    assert "p.1" in result
    assert "Hello world" in result
    assert "Second chunk" in result


def test_format_documents_empty():
    result = format_documents([])
    assert result == ""


def test_rag_response_dataclass():
    doc = Document(page_content="test", metadata={})
    response = RAGResponse(answer="The answer is 42.", sources=[doc])
    assert response.answer == "The answer is 42."
    assert len(response.sources) == 1
