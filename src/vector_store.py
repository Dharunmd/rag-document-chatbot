# src/vector_store.py

from langchain_chroma import Chroma

from src.embeddings import load_embedding_model

VECTOR_DB_PATH = "vector_db"


def create_vector_store(chunks):
    """
    Create and persist a Chroma vector database.
    """

    print("🔄 Creating vector store...")

    embeddings = load_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    print(
        f"✅ Vector store created with "
        f"{len(chunks)} chunks"
    )

    return vector_store


def load_vector_store():
    """
    Load an existing vector database.
    """

    embeddings = load_embedding_model()

    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_store