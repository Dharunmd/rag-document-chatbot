from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def load_embedding_model():
    """Load the sentence-transformer used for document indexing."""

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
