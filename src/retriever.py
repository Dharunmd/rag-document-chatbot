from config import RETRIEVER_FETCH_K, RETRIEVER_K
from src.vector_store import load_vector_store


def get_retriever(k=None, fetch_k=None):
    """Return an MMR retriever over the indexed document chunks."""

    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k or RETRIEVER_K,
            "fetch_k": fetch_k or RETRIEVER_FETCH_K,
        },
    )
