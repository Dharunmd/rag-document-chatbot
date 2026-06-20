import shutil

from langchain_chroma import Chroma

from config import VECTOR_DB_PATH
from src.embeddings import load_embedding_model


def clear_vector_store():
    """Remove the persisted Chroma index before indexing a new document."""
    import chromadb

    chromadb.api.client.SharedSystemClient.clear_system_cache()

    if VECTOR_DB_PATH.exists():
        shutil.rmtree(VECTOR_DB_PATH)
        print("Removed previous vector store")


def create_vector_store(chunks):
    """Build and persist a Chroma collection from document chunks."""

    clear_vector_store()
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    embeddings = load_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )

    print(f"Indexed {len(chunks)} chunks into Chroma")
    return vector_store


def load_vector_store():
    """Load the persisted Chroma collection."""

    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            "No indexed document found. Upload and process a file first."
        )

    embeddings = load_embedding_model()

    return Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
    )
