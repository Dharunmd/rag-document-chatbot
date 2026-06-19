from src.vector_store import load_vector_store


def get_retriever(k=4):
    """
    Create and return a retriever.
    """

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10
        }
    )

    return retriever