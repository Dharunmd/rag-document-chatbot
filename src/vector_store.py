# src/vector_store.py

import os
import shutil

from langchain_chroma import Chroma
from src.embeddings import load_embedding_model

VECTOR_DB_PATH = "vector_db"


def clear_vector_store():
    """
    Remove existing vector database.
    """

    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)
        print("🗑️ Old vector store removed")


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
    Load existing vector database.
    """

    embeddings = load_embedding_model()

    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vector_store