# src/embeddings.py

from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


def load_embedding_model():
    """
    Load the sentence transformer embedding model.
    
    Returns:
        HuggingFaceEmbeddings model instance
    """

    print("🔄 Loading embedding model...")
    print("   Model: all-MiniLM-L6-v2")
    print("   This may take a minute on first run (downloading model)...")

    # Load the embedding model
    # model_name: which pretrained model to use
    # model_kwargs: run on CPU (device:'cpu')
    # encode_kwargs: normalize vectors to unit length
    #                this makes cosine similarity calculations faster
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("✅ Embedding model loaded successfully!")
    return embedding_model


def embed_text(text: str, embedding_model) -> list:
    """
    Convert a single piece of text into a vector.
    
    Args:
        text: The text to embed
        embedding_model: Loaded embedding model
        
    Returns:
        List of floats representing the vector
    """

    # embed_query is used for single text embedding
    # it returns a list of 384 floating point numbers
    vector = embedding_model.embed_query(text)
    return vector


def show_embedding_demo(embedding_model):
    """
    Demo function to visually understand embeddings.
    Shows how similar texts have similar vectors.
    
    Args:
        embedding_model: Loaded embedding model
    """

    print("\n🧪 Embedding Demo — Similarity Test")
    print("=" * 45)

    # Three sentences — two similar, one different
    sentences = [
        "machine learning algorithms",   # similar to sentence 2
        "deep learning neural networks",  # similar to sentence 1
        "cooking pasta recipe"            # completely different
    ]

    # Embed all three sentences
    vectors = [embed_text(s, embedding_model) for s in sentences]

    # Convert to numpy arrays for math operations
    v1 = np.array(vectors[0])
    v2 = np.array(vectors[1])
    v3 = np.array(vectors[2])

    # Cosine similarity — measures angle between vectors
    # Result: 1.0 = identical meaning, 0.0 = completely different
    # Formula: dot product of two normalized vectors
    sim_12 = np.dot(v1, v2)  # similarity between sentence 1 and 2
    sim_13 = np.dot(v1, v3)  # similarity between sentence 1 and 3

    print(f"\n📝 Sentence 1: '{sentences[0]}'")
    print(f"📝 Sentence 2: '{sentences[1]}'")
    print(f"📝 Sentence 3: '{sentences[2]}'")
    print(f"\n📊 Similarity Score (1.0 = same, 0.0 = different):")
    print(f"   Sentence 1 vs 2 (both AI topics): {sim_12:.4f} ← should be HIGH")
    print(f"   Sentence 1 vs 3 (AI vs cooking):  {sim_13:.4f} ← should be LOW")
    print(f"\n💡 Vector size: {len(vectors[0])} dimensions")
    print(f"💡 First 5 numbers of vector 1: {vectors[0][:5]}")