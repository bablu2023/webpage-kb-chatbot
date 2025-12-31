import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1️⃣ Load chunks
with open("chunks.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = text.split("\n--- CHUNK ")
chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

print("🧩 Total chunks:", len(chunks))

# 2️⃣ Create TF-IDF embeddings
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

embeddings = vectorizer.fit_transform(chunks).toarray().astype("float32")

print("📐 Embedding shape:", embeddings.shape)

# 3️⃣ Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("📦 Stored vectors:", index.ntotal)

# 4️⃣ Save everything
faiss.write_index(index, "kb_index.faiss")

with open("kb_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF embeddings & FAISS index created successfully")
