"""Smoke tests for document ingestion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.core.ingest import ingest_documents, SUPPORTED_EXTENSIONS
from src.core.config import get_settings


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS


def test_ingest_txt_file():
    settings = get_settings()
    vectorstore = MagicMock()
    vectorstore.add_documents = MagicMock()

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("This is a test document. " * 50)
        tmp_path = Path(f.name)

    chunks_added = ingest_documents([tmp_path], vectorstore, settings)

    assert chunks_added > 0
    vectorstore.add_documents.assert_called_once()
    tmp_path.unlink()


def test_ingest_unsupported_extension():
    settings = get_settings()
    vectorstore = MagicMock()
    vectorstore.add_documents = MagicMock()

    fake_path = Path("document.xyz")
    chunks_added = ingest_documents([fake_path], vectorstore, settings)

    assert chunks_added == 0
    vectorstore.add_documents.assert_not_called()
