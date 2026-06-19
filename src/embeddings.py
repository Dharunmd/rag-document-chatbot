# src/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the embedding model.
    """

    print("🔄 Loading embedding model...")
    print(f"   Model: {MODEL_NAME}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("✅ Embedding model loaded successfully!")

    return embedding_model


def embed_text(
    text: str,
    embedding_model
):
    """
    Generate embedding for a single text.
    """

    return embedding_model.embed_query(text)


def embed_documents(
    texts: list[str],
    embedding_model
):
    """
    Generate embeddings for multiple texts.
    """

    return embedding_model.embed_documents(texts)


def show_embedding_demo(
    embedding_model
):
    """
    Demonstrate semantic similarity.
    """

    print("\n🧪 Embedding Similarity Demo")
    print("-" * 40)

    sentences = [
        "machine learning algorithms",
        "deep learning neural networks",
        "cooking pasta recipe"
    ]

    vectors = [
        embed_text(
            sentence,
            embedding_model
        )
        for sentence in sentences
    ]

    v1 = np.array(vectors[0])
    v2 = np.array(vectors[1])
    v3 = np.array(vectors[2])

    sim_12 = np.dot(v1, v2)
    sim_13 = np.dot(v1, v3)

    print(
        f"Similarity (AI vs AI): "
        f"{sim_12:.4f}"
    )

    print(
        f"Similarity (AI vs Cooking): "
        f"{sim_13:.4f}"
    )

    print(
        f"Embedding Dimensions: "
        f"{len(vectors[0])}"
    )