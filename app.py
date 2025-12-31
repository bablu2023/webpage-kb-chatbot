import streamlit as st
import pickle
import faiss
import numpy as np
import os

st.set_page_config(page_title="Webpage KB Chatbot")

st.title("🌐 Webpage Knowledge-Base Chatbot")

# ---- File checks ----
required_files = ["chunks.txt", "tfidf_vectorizer.pkl"]
for f in required_files:
    if not os.path.exists(f):
        st.error(f"Missing file: {f}")
        st.stop()

st.success("Required files found. Building index...")

# ---- Load data ----
with open("chunks.txt", "r", encoding="utf-8") as f:
    raw = f.read()

chunks = raw.split("\n--- CHUNK ")
chunks = [c.strip() for c in chunks if len(c.strip()) > 50]

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---- Build embeddings & FAISS INDEX (CLOUD SAFE) ----
embeddings = vectorizer.transform(chunks).toarray().astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

st.success("FAISS index built successfully ✅")

# ---- UI ----
query = st.text_input("Ask a question")

if query:
    q_vec = vectorizer.transform([query]).toarray().astype("float32")
    distances, indices = index.search(q_vec, k=3)

    st.subheader("Answer (from webpage content)")

    for i, idx in enumerate(indices[0], start=1):
        st.markdown(f"**Result {i}:**")
        st.write(chunks[idx][:600])
        st.markdown("---")
