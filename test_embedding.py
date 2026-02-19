from sentence_transformers import SentenceTransformer
import numpy as np

# load a free local model
model = SentenceTransformer("all-MiniLM-L6-v2")

# create embedding
embedding = model.encode("hello")

print("✅ Embedding length:", len(embedding))
print("✅ First 10 values:", embedding[:10])  # preview
