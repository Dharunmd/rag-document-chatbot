from langchain_chroma import Chroma
from src.embeddings import load_embedding_model

PERSIST_DIR = "vector_db"


def create_vector_store(chunks):

    embeddings = load_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    return vector_store


def load_vector_store():

    embeddings = load_embedding_model()

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )