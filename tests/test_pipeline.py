"""Basic checks for the RAG pipeline modules."""

from pathlib import Path

from config import SAMPLE_DOCS_DIR


def test_sample_document_exists():
    sample = SAMPLE_DOCS_DIR / "project_report_sample.txt"
    assert sample.exists(), "Sample document missing for demo mode"


def test_document_loader_splits_text():
    from src.document_loader import get_document_info, load_document

    sample = SAMPLE_DOCS_DIR / "project_report_sample.txt"
    chunks = load_document(str(sample))
    info = get_document_info(chunks)

    assert len(chunks) > 0
    assert info["total_chunks"] == len(chunks)
    assert info["total_characters"] > 0


def test_embedding_model_loads():
    from src.embeddings import load_embedding_model

    model = load_embedding_model()
    vector = model.embed_query("retrieval augmented generation")
    assert len(vector) > 0
