# test_embeddings.py

from src.embeddings import load_embedding_model, embed_text, show_embedding_demo

# Step 1: Load the model
embedding_model = load_embedding_model()

# Step 2: Run the similarity demo
show_embedding_demo(embedding_model)

# Step 3: Test with your own text
print("\n🎯 Testing with custom text:")
my_text = "What are Dharun's technical skills?"
vector = embed_text(my_text, embedding_model)
print(f"   Text: '{my_text}'")
print(f"   Vector dimensions: {len(vector)}")
print(f"   First 3 values: {vector[:3]}")