import streamlit as st
import faiss
import pickle
import numpy as np

# Load vector database
index = faiss.read_index("kb_index.faiss")

with open("kb_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Webpage KB Chatbot", layout="centered")

st.title("🌐 Webpage Knowledge-Base Chatbot")
st.write("Ask questions based on the webpage content")

query = st.text_input("Enter your question")

if query:
    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    distances, indices = index.search(query_vec, k=3)

    st.subheader("Answer (from webpage):")

    for i, idx in enumerate(indices[0], start=1):
        st.markdown(f"**Result {i}:**")
        st.write(chunks[idx][:600])
        st.markdown("---")
