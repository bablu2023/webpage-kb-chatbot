import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Webpage Knowledge-Base Chatbot", layout="centered")
st.title("🌐 Webpage Knowledge-Base Chatbot")

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def read_webpage(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_faiss_index(chunks):
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(chunks).toarray().astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, vectorizer


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.vectorizer = None
    st.session_state.chunks = None

# --------------------------------------------------
# TRAINING SECTION
# --------------------------------------------------
st.subheader("🔁 Train on New Webpage")

url = st.text_input(
    "Enter webpage URL",
    placeholder="https://en.wikipedia.org/wiki/Machine_learning"
)

if st.button("🚀 Train on New Page"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Reading webpage and building knowledge base..."):
            try:
                text = read_webpage(url)
                chunks = chunk_text(text)

                index, vectorizer = build_faiss_index(chunks)

                st.session_state.index = index
                st.session_state.vectorizer = vectorizer
                st.session_state.chunks = chunks

                st.success(f"Training completed ✅ ({len(chunks)} chunks indexed)")
            except Exception as e:
                st.error(f"Error while training: {e}")

# --------------------------------------------------
# CHAT SECTION (LEVEL 1 FIX)
# --------------------------------------------------
st.subheader("💬 Ask a Question")

query = st.text_input("Type your question here")

if query:
    if st.session_state.index is None:
        st.warning("Please train on a webpage first.")
    else:
        # Vectorize query
        query_vector = (
            st.session_state.vectorizer
            .transform([query])
            .toarray()
            .astype("float32")
        )

        # Search top-1 result only
        distances, indices = st.session_state.index.search(query_vector, k=1)

        best_chunk = st.session_state.chunks[indices[0][0]]

        st.markdown("### 📄 Answer (Most Relevant Section)")
        st.write(best_chunk)
