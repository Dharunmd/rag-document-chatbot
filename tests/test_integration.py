"""Integration test for indexing + query (requires GOOGLE_API_KEY)."""

import os
from pathlib import Path

import pytest

from config import SAMPLE_DOCS_DIR


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY required for live RAG test",
)
def test_index_and_ask_sample_document():
    from src.pipeline import ask_document, index_document

    sample = SAMPLE_DOCS_DIR / "project_report_sample.txt"
    index_result = index_document(str(sample))

    assert index_result.info["total_chunks"] > 0

    query_result = ask_document("What embedding model is used in this project?")
    assert query_result.answer
    assert "mini" in query_result.answer.lower() or len(query_result.sources) > 0
