from dataclasses import dataclass, field
from time import perf_counter

from src.document_loader import (
    get_document_info,
    load_document,
)


@dataclass
class IndexResult:
    info: dict
    elapsed_seconds: float


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def index_document(file_path: str) -> IndexResult:
    """Load, chunk, embed, and persist a document."""

    from src.vector_store import create_vector_store

    start = perf_counter()

    chunks = load_document(file_path)

    create_vector_store(chunks)

    info = get_document_info(chunks)

    return IndexResult(
        info=info,
        elapsed_seconds=perf_counter() - start,
    )


def ask_document(question: str) -> QueryResult:
    """Run retrieval-augmented generation."""

    from src.retriever import get_retriever
    from src.llm_chain import create_qa_chain

    start = perf_counter()

    retriever = get_retriever()

    qa_chain = create_qa_chain(retriever)

    response = qa_chain.invoke(
        {"query": question}
    )

    sources = []

    for doc in response.get(
        "source_documents",
        []
    ):
        sources.append(
            {
                "content": doc.page_content,
                "metadata": getattr(
                    doc,
                    "metadata",
                    {},
                ),
            }
        )

    return QueryResult(
        answer=response["result"],
        sources=sources,
        elapsed_seconds=(
            perf_counter() - start
        ),
    )