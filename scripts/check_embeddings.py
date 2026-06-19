"""Manual helper script for checking embedding model loading."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import load_embedding_model


def main():
    model = load_embedding_model()

    samples = [
        "machine learning algorithms",
        "deep learning neural networks",
        "cooking pasta recipe",
    ]

    vectors = [model.embed_query(text) for text in samples]

    print("\nEmbedding check")
    print("-" * 30)
    print(f"Dimensions: {len(vectors[0])}")
    print(f"AI vs AI similarity: {sum(a * b for a, b in zip(vectors[0], vectors[1])):.4f}")
    print(f"AI vs cooking similarity: {sum(a * b for a, b in zip(vectors[0], vectors[2])):.4f}")


if __name__ == "__main__":
    main()
