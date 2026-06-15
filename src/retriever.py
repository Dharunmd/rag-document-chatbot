from src.vector_store import load_vector_store

def get_retriever(k=4):

    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        "skills",
        k=2
    )

    print("\nRetrieved Documents:")
    print(docs)

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )