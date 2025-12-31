import faiss
import pickle
import numpy as np

# Load FAISS index
index = faiss.read_index("kb_index.faiss")

# Load chunks
with open("kb_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("🤖 Webpage Knowledge-Base Chatbot")
print("Type 'exit' to quit")

while True:
    query = input("\nAsk a question: ").strip()

    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Vectorize query
    query_vector = vectorizer.transform([query]).toarray().astype("float32")

    # Search in FAISS
    k = 3
    distances, indices = index.search(query_vector, k)

    print("\n📄 Answer (from webpage content):\n")

    for rank, idx in enumerate(indices[0], start=1):
        print(f"[Result {rank}]")
        print(chunks[idx][:600])
        print("-" * 70)
